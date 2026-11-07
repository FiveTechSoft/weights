"""Compiler for WSP-Lang architecture files (.wsparch).

An architecture is plain data: a config, a graph of tensor primitives and explicit init passes.
Compiling it needs no model class, so changing the text changes the network. Expressions are parsed
with `ast` and walked through an allowlist; nothing from a source file is ever evaluated as Python.

Grammar, by section (see programs/arch/nanogpt.wsparch):
    config:        name = expression
    graph(input):  var = op(args) | a, b = split(x, 2) | var = other | output var
                   scope path:  (prefixes tensor names)
                   repeat i in expression:  (use {i} in scope paths)
    init:          pass: selector = fn(args)   (kind.attr or match "glob"; applied in graph order)
                   tie tensor = tensor         (first name becomes an alias of the second)

Init passes run in order and consume the global RNG in graph order. Same-seed parity with another
implementation therefore requires declaring every draw it makes, including construction defaults.
Loss is fixed to cross-entropy with ignore_index=-1 and is not yet declared by the language.
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

MAX_PARAMETERS = 2_000_000_000
MAX_REPEAT = 4096
MAX_INSTRUCTIONS = 200_000

NAME_RE = re.compile(r"^[A-Za-z_]\w*$")
PATH_RE = re.compile(r"^[A-Za-z_0-9{}]+(\.[A-Za-z_0-9{}]+)*$")
BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv}
LITERALS = {"true": True, "false": False}

# N: tensor name, T: float tensor var, I: integer-id tensor var, i/f/b: int/float/bool expression
OPS = {
    "embedding": "NIii",
    "positions": "I",
    "add": "TT",
    "dropout": "Tf",
    "layernorm": "NTib",
    "linear": "NTiib",
    "split": "Ti",
    "causal_attention": "TTTif",
    "gelu": "T",
}
SELECTORS = {"embedding.weight", "linear.weight", "linear.bias", "layernorm.weight", "layernorm.bias"}


class ArchError(ValueError):
    """Invalid architecture source; the message carries the source line when one applies."""


def _fail(line, message):
    raise ArchError(f"line {line}: {message}")


# ---------------------------------------------------------------------------
# Source layout
# ---------------------------------------------------------------------------

@dataclass
class Node:
    text: str
    line: int
    children: list


def _parse_tree(text):
    root = Node("", 0, [])
    stack = [(-1, root)]
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prefix = raw[:len(raw) - len(raw.lstrip())]
        if "\t" in prefix:
            _fail(number, "indent must use spaces")
        indent = len(prefix)
        while indent < stack[-1][0]:
            stack.pop()
        if indent > stack[-1][0]:
            parent = stack[-1][1]
            if parent is not root and not parent.text.endswith(":"):
                _fail(number, "unexpected indent")
            node = Node(stripped, number, [])
            parent.children.append(node)
            stack.append((indent, node))
        else:
            parent = stack[-2][1]
            node = Node(stripped, number, [])
            parent.children.append(node)
            stack[-1] = (indent, node)
    return root


def _parse_expr(text, line):
    try:
        return ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        _fail(line, f"invalid expression: {text.strip()!r}")


def _eval(node, env, line):
    """Evaluates an arithmetic expression; every node type outside this list is rejected."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in LITERALS:
            return LITERALS[node.id]
        if node.id not in env:
            _fail(line, f"undefined name {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        return BINOPS[type(node.op)](_eval(node.left, env, line), _eval(node.right, env, line))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, env, line)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt"
            and len(node.args) == 1 and not node.keywords):
        return math.sqrt(_eval(node.args[0], env, line))
    _fail(line, f"expression element not allowed: {ast.dump(node)[:60]}")


def _section(arch, prefix):
    found = [n for n in arch.children if n.text.startswith(prefix)]
    if len(found) != 1:
        _fail(arch.line, f"arch needs exactly one {prefix!r} section")
    return found[0]


def _config(section, overrides):
    declared = {n.text.split("=", 1)[0].strip() for n in section.children}
    unknown = set(overrides) - declared
    if unknown:
        raise ArchError(f"line {section.line}: unknown config override(s): {sorted(unknown)}")
    cfg = {}
    for node in section.children:
        if "=" not in node.text:
            _fail(node.line, f"expected 'name = expression': {node.text!r}")
        name, expr = (p.strip() for p in node.text.split("=", 1))
        if not NAME_RE.match(name):
            _fail(node.line, f"invalid config name {name!r}")
        cfg[name] = overrides[name] if name in overrides else _eval(_parse_expr(expr, node.line), cfg, node.line)
    return cfg


# ---------------------------------------------------------------------------
# Graph flattening and shape checking
# ---------------------------------------------------------------------------

@dataclass
class TensorDecl:
    name: str
    selector: str
    shape: tuple
    fan_in: int = 0


class _GraphBuilder:
    def __init__(self, cfg, input_var):
        self.cfg = cfg
        self.input_var = input_var
        self.vars = {input_var: ("ids", None)}  # name -> (kind, last dim)
        self.tensors = {}
        self.code = []
        self.output = None
        self.max_seq = None
        self.parameter_count = 0

    def walk(self, nodes, scope, env):
        for node in nodes:
            self._statement(node, scope, env)

    def _statement(self, node, scope, env):
        text = node.text
        if text.startswith("scope "):
            path = text[len("scope "):].rstrip(":").strip()
            if not PATH_RE.match(path):
                _fail(node.line, f"invalid scope path {path!r}")
            for key, value in env.items():
                path = path.replace("{" + key + "}", str(value))
            if "{" in path:
                _fail(node.line, f"undefined loop variable in scope {path!r}")
            self.walk(node.children, f"{scope}{path}.", env)
        elif text.startswith("repeat "):
            match = re.match(r"^repeat\s+(\w+)\s+in\s+(.+):$", text)
            if not match:
                _fail(node.line, "expected 'repeat i in expression:'")
            count = self._int(_parse_expr(match.group(2), node.line), env, node.line)
            if count > MAX_REPEAT:
                _fail(node.line, f"repeat count {count} exceeds limit {MAX_REPEAT}")
            for i in range(count):
                self.walk(node.children, scope, {**env, match.group(1): i})
        elif text.startswith("output "):
            name = text[len("output "):].strip()
            if name not in self.vars:
                _fail(node.line, f"undefined variable {name!r}")
            self.output = name
        elif "=" in text:
            self._assignment(node, scope, env)
        else:
            _fail(node.line, f"cannot parse statement: {text!r}")

    def _int(self, expr, env, line):
        value = _eval(expr, {**self.cfg, **env}, line)
        if isinstance(value, bool) or not (isinstance(value, int) or float(value).is_integer()):
            _fail(line, f"expected an integer, got {value!r}")
        return int(value)

    def _assignment(self, node, scope, env):
        lhs, rhs = (p.strip() for p in node.text.split("=", 1))
        outs = [v.strip() for v in lhs.split(",")]
        for out in outs:
            if not NAME_RE.match(out):
                _fail(node.line, f"invalid variable name {out!r}")
            if out in self.cfg:
                _fail(node.line, f"variable {out!r} would shadow a config value")
        expr = _parse_expr(rhs, node.line)
        if isinstance(expr, ast.Name):
            if expr.id not in self.vars:
                _fail(node.line, f"undefined variable {expr.id!r}")
            self.vars[outs[0]] = self.vars[expr.id]
            self.code.append(("alias", outs, [expr.id]))
        elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and not expr.keywords:
            self._call(expr, outs, scope, env, node.line)
        else:
            _fail(node.line, f"right side must be op(args) or a variable: {rhs!r}")
        if len(self.code) > MAX_INSTRUCTIONS:
            _fail(node.line, f"expanded graph exceeds limit of {MAX_INSTRUCTIONS} instructions")

    def _arguments(self, signature, args, scope, env, op, line):
        if len(args) != len(signature):
            _fail(line, f"{op} takes {len(signature)} arguments, got {len(args)}")
        values = []
        for kind, arg in zip(signature, args):
            if kind == "N":
                if not (isinstance(arg, ast.Name) and NAME_RE.match(arg.id)):
                    _fail(line, f"{op} needs a tensor name as first argument")
                values.append(f"{scope}{arg.id}")
            elif kind in "TI":
                if not isinstance(arg, ast.Name) or arg.id not in self.vars:
                    _fail(line, f"{op}: undefined variable {ast.unparse(arg)!r}")
                wanted = ("ids", "positions") if kind == "I" else ("float",)
                if self.vars[arg.id][0] not in wanted:
                    _fail(line, f"{op}: {arg.id!r} has the wrong tensor kind")
                values.append(arg.id)
            elif kind == "i":
                values.append(self._int(arg, env, line))
            else:
                value = _eval(arg, {**self.cfg, **env}, line)
                if kind == "b" and not isinstance(value, bool):
                    _fail(line, f"{op}: expected true or false")
                values.append(value)
        return values

    def _declare(self, name, selector, shape, line, fan_in=0):
        if name in self.tensors:
            _fail(line, f"duplicate tensor {name!r}")
        if any(s <= 0 for s in shape):
            _fail(line, f"tensor {name!r} has non-positive shape {shape}")
        self.parameter_count += math.prod(shape)
        if self.parameter_count > MAX_PARAMETERS:
            _fail(line, f"parameter count exceeds limit of {MAX_PARAMETERS}")
        self.tensors[name] = TensorDecl(name, selector, shape, fan_in)

    def _call(self, call, outs, scope, env, line):
        op = call.func.id
        if op not in OPS:
            _fail(line, f"unknown operation {op!r}")
        args = self._arguments(OPS[op], call.args, scope, env, op, line)
        if op != "split" and len(outs) != 1:
            _fail(line, f"{op} produces one value, got {len(outs)} targets")
        dim_of = lambda v: self.vars[v][1]  # noqa: E731
        if op == "embedding":
            name, ids, count, dim = args
            self._declare(f"{name}.weight", "embedding.weight", (count, dim), line)
            if self.vars[ids][0] == "positions":
                self.max_seq = count
            kind, out_dim, inputs, extra = "float", dim, [ids], (f"{name}.weight",)
        elif op == "positions":
            kind, out_dim, inputs, extra = "positions", None, [args[0]], ()
        elif op == "add":
            if dim_of(args[0]) != dim_of(args[1]):
                _fail(line, f"add: dimensions differ ({dim_of(args[0])} vs {dim_of(args[1])})")
            kind, out_dim, inputs, extra = "float", dim_of(args[0]), args, ()
        elif op == "dropout":
            if not 0.0 <= args[1] < 1.0:
                _fail(line, f"dropout probability out of range: {args[1]}")
            kind, out_dim, inputs, extra = "float", dim_of(args[0]), [args[0]], (float(args[1]),)
        elif op == "gelu":
            kind, out_dim, inputs, extra = "float", dim_of(args[0]), args, ()
        elif op == "layernorm":
            name, x, dim, bias = args
            if dim_of(x) != dim:
                _fail(line, f"layernorm: input dimension {dim_of(x)} is not {dim}")
            self._declare(f"{name}.weight", "layernorm.weight", (dim,), line)
            if bias:
                self._declare(f"{name}.bias", "layernorm.bias", (dim,), line)
            kind, out_dim, inputs, extra = "float", dim, [x], (name, bias)
        elif op == "linear":
            name, x, dim_in, dim_out, bias = args
            if dim_of(x) != dim_in:
                _fail(line, f"linear: input dimension {dim_of(x)} is not {dim_in}")
            self._declare(f"{name}.weight", "linear.weight", (dim_out, dim_in), line)
            if bias:
                self._declare(f"{name}.bias", "linear.bias", (dim_out,), line, fan_in=dim_in)
            kind, out_dim, inputs, extra = "float", dim_out, [x], (name, bias)
        elif op == "split":
            x, parts = args
            if parts != len(outs):
                _fail(line, f"split into {parts} parts needs {parts} targets, got {len(outs)}")
            if dim_of(x) % parts:
                _fail(line, f"split: dimension {dim_of(x)} is not divisible by {parts}")
            kind, out_dim, inputs, extra = "float", dim_of(x) // parts, [x], (dim_of(x) // parts,)
        else:  # causal_attention
            q, k, v, heads, p = args
            if not dim_of(q) == dim_of(k) == dim_of(v):
                _fail(line, "causal_attention: q, k, v dimensions differ")
            if heads <= 0 or dim_of(q) % heads:
                _fail(line, f"causal_attention: {heads} heads do not divide dimension {dim_of(q)}")
            kind, out_dim, inputs, extra = "float", dim_of(q), [q, k, v], (heads, float(p))
        for out in outs:
            self.vars[out] = (kind, out_dim)
        self.code.append((op, outs, inputs, *extra))


# ---------------------------------------------------------------------------
# Init passes
# ---------------------------------------------------------------------------

def _init_rule(node, env):
    lhs, rhs = (p.strip() for p in node.text.split("=", 1))
    match = re.match(r'^match\s+"([^"]+)"$', lhs)
    if not match and lhs not in SELECTORS:
        _fail(node.line, f"unknown init selector {lhs!r}")
    call = _parse_expr(rhs, node.line)
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and not call.keywords):
        _fail(node.line, f"init value must be fn(args): {rhs!r}")
    return {"line": node.line, "selector": None if match else lhs, "glob": match.group(1) if match else None,
            "fn": call.func.id, "args": [_eval(a, env, node.line) for a in call.args]}


def _apply_init(rule, decl, tensor):
    fn, args, line = rule["fn"], rule["args"], rule["line"]
    wanted = {"normal": 2, "kaiming_uniform": 1, "fan_in_uniform": 0, "zeros": 0, "ones": 0}
    if fn not in wanted:
        _fail(line, f"unknown init function {fn!r}")
    if len(args) != wanted[fn]:
        _fail(line, f"{fn} takes {wanted[fn]} arguments, got {len(args)}")
    if fn == "normal":
        nn.init.normal_(tensor, mean=args[0], std=args[1])
    elif fn == "zeros":
        nn.init.zeros_(tensor)
    elif fn == "ones":
        nn.init.ones_(tensor)
    elif fn == "kaiming_uniform":
        if tensor.dim() != 2:
            _fail(line, f"kaiming_uniform needs a matrix, {decl.name} has shape {decl.shape}")
        nn.init.kaiming_uniform_(tensor, a=args[0])
    else:
        if not decl.fan_in:
            _fail(line, f"fan_in_uniform only applies to linear bias, not {decl.name}")
        bound = 1 / math.sqrt(decl.fan_in)
        nn.init.uniform_(tensor, -bound, bound)


# ---------------------------------------------------------------------------
# Runtime module
# ---------------------------------------------------------------------------

class ParamScope(nn.Module):
    """Empty container: gives parameters their dotted names, holds no behaviour."""


class CompiledArch(nn.Module):
    def __init__(self, name, cfg, builder, init_section, source_sha256, seed):
        super().__init__()
        self.arch_name = name
        self.source_sha256 = source_sha256
        self.config = SimpleNamespace(**cfg, block_size=builder.max_seq,
                                      vocab_size=builder.vars[builder.output][1])
        self._code = builder.code
        self._input = builder.input_var
        self._output = builder.output
        self._decls = builder.tensors
        self._tensors = {}
        self._aliases = []
        for decl in self._decls.values():
            self._set_tensor(decl.name, nn.Parameter(torch.empty(decl.shape)))
        torch.manual_seed(seed)
        self._run_init(init_section, cfg)

    def _scope_for(self, name):
        scope = self
        parts = name.split(".")
        for part in parts[:-1]:
            child = getattr(scope, "_modules").get(part)
            if child is None:
                child = ParamScope()
                scope.add_module(part, child)
            scope = child
        return scope, parts[-1]

    def _set_tensor(self, name, parameter):
        scope, leaf = self._scope_for(name)
        scope.register_parameter(leaf, parameter)
        self._tensors[name] = parameter

    @torch.no_grad()
    def _run_init(self, section, cfg):
        initialized = set()
        for node in section.children:
            if node.text == "pass:":
                rules = [_init_rule(child, cfg) for child in node.children]
                used = set()
                for decl in self._decls.values():
                    for i, rule in enumerate(rules):
                        hit = (rule["selector"] == decl.selector if rule["selector"]
                               else fnmatch.fnmatchcase(decl.name, rule["glob"]))
                        if hit:
                            _apply_init(rule, decl, self._tensors[decl.name])
                            initialized.add(id(self._tensors[decl.name]))
                            used.add(i)
                            break
                for i, rule in enumerate(rules):
                    if rule["glob"] and i not in used:
                        _fail(rule["line"], f"match {rule['glob']!r} selects no tensor")
            elif node.text.startswith("tie "):
                self._tie(node)
            else:
                _fail(node.line, f"expected 'pass:' or 'tie a = b', got {node.text!r}")
        missing = [n for n, p in self._tensors.items() if id(p) not in initialized]
        if missing:
            _fail(section.line, f"uninitialized tensors: {missing[:5]}")

    def _tie(self, node):
        if "=" not in node.text:
            _fail(node.line, "expected 'tie a = b'")
        alias, target = (p.strip() for p in node.text[len("tie "):].split("=", 1))
        for name in (alias, target):
            if name not in self._tensors:
                _fail(node.line, f"tie: unknown tensor {name!r}")
        if self._tensors[alias].shape != self._tensors[target].shape:
            _fail(node.line, f"tie: shapes differ for {alias} and {target}")
        self._set_tensor(alias, self._tensors[target])
        self._aliases.append((alias, target))

    def _layer_params(self, name, bias):
        return self._tensors[f"{name}.weight"], self._tensors[f"{name}.bias"] if bias else None

    def input_of(self, tensor_scope, idx):
        """Value entering the linear layer named `tensor_scope`, stopping execution there."""
        return self._execute(idx, last_only=False, stop_before=tensor_scope)

    def _execute(self, idx, last_only, stop_before=None):
        env = {self._input: idx}
        tensors = self._tensors
        for instr in self._code:
            op, outs, inputs = instr[:3]
            extra = instr[3:]
            if op == "alias":
                env[outs[0]] = env[inputs[0]]
            elif op == "embedding":
                env[outs[0]] = F.embedding(env[inputs[0]], tensors[extra[0]])
            elif op == "positions":
                env[outs[0]] = torch.arange(0, env[inputs[0]].size(1), dtype=torch.long, device=idx.device)
            elif op == "add":
                env[outs[0]] = env[inputs[0]] + env[inputs[1]]
            elif op == "dropout":
                env[outs[0]] = F.dropout(env[inputs[0]], extra[0], self.training)
            elif op == "gelu":
                env[outs[0]] = F.gelu(env[inputs[0]])
            elif op == "layernorm":
                weight, bias = self._layer_params(*extra)
                env[outs[0]] = F.layer_norm(env[inputs[0]], weight.shape, weight, bias, 1e-5)
            elif op == "linear":
                if extra[0] == stop_before:
                    return env[inputs[0]]
                weight, bias = self._layer_params(*extra)
                x = env[inputs[0]]
                if last_only and outs[0] == self._output and instr is self._code[-1]:
                    x = x[:, [-1], :]  # same shortcut as upstream at inference
                env[outs[0]] = F.linear(x, weight, bias)
            elif op == "split":
                for out, part in zip(outs, env[inputs[0]].split(extra[0], dim=2)):
                    env[out] = part
            else:  # causal_attention
                heads, p = extra
                B, T, C = env[inputs[0]].size()
                q, k, v = (env[name].view(B, T, heads, C // heads).transpose(1, 2) for name in inputs)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                   dropout_p=p if self.training else 0, is_causal=True)
                env[outs[0]] = y.transpose(1, 2).contiguous().view(B, T, C)
        return env[self._output]

    def forward(self, idx, targets=None):
        limit = self.config.block_size
        if limit is not None and idx.size(1) > limit:
            raise ValueError(f"sequence length {idx.size(1)} exceeds block size {limit}")
        logits = self._execute(idx, last_only=targets is None)
        if targets is None:
            return logits[:, [-1], :], None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Same sampling as upstream's generate(); temperature=0 selects the argmax."""
        limit = self.config.block_size
        for _ in range(max_new_tokens):
            context = idx if limit is None or idx.size(1) <= limit else idx[:, -limit:]
            logits = self(context)[0][:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def export_state_dict(self):
        """Independent copies, since safetensors refuses tensors that share memory."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.export_state_dict(), str(path),
                  metadata={"format": "pt", "wsparch": self.arch_name, "wsparch_sha256": self.source_sha256,
                            **(metadata or {})})

    def load(self, path):
        sd = load_file(str(path))
        for alias, target in self._aliases:
            if alias in sd and target in sd and not torch.equal(sd[alias], sd[target]):
                raise RuntimeError(f"checkpoint breaks declared tie: {alias} != {target}")
        self.load_state_dict(sd, strict=True)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def parse_config_value(text):
    """Converts a command-style override such as '3', '0.1' or 'false' to a config value."""
    expr = _parse_expr(str(text), 0)
    return _eval(expr, {}, 0)


def compile_source(source, overrides=None, seed=1337):
    """Compiles architecture text into a module with weights from its init passes."""
    root = _parse_tree(source)
    if len(root.children) != 1 or not root.children[0].text.startswith("arch "):
        raise ArchError("source must contain exactly one 'arch name:' block")
    arch = root.children[0]
    name = arch.text[len("arch "):].rstrip(":").strip()
    cfg = _config(_section(arch, "config:"), overrides or {})
    graph = _section(arch, "graph(")
    header = re.match(r"^graph\((\w+)\):$", graph.text)
    if not header:
        _fail(graph.line, "expected 'graph(input):'")
    builder = _GraphBuilder(cfg, header.group(1))
    builder.walk(graph.children, "", {})
    if builder.output is None:
        _fail(graph.line, "graph has no output statement")
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return CompiledArch(name, cfg, builder, _section(arch, "init:"), sha, seed)


def load_arch_file(path, overrides=None, seed=1337):
    return compile_source(Path(path).read_text(encoding="utf-8"), overrides=overrides, seed=seed)
