"""Decompiles nanoGPT weights into a WSPLang n-gram program (`.wspngram`) that compiles back to the same function.

Two decompilers:
  - decompile_ngram(): structural. Reads weights produced by wsp_synth.compile_ngram and recovers every table
    entry from the weights alone: no seed, no table and no training data. It knows the compiler's layout, as a
    disassembler knows an instruction set, and rejects weights that do not follow it.
  - behavioral_table(): for any model, including gradient-trained ones. Averages the model's own predictions over
    every occurrence of each short context in a text. The result is the best n-gram description of what the model
    does at that context length; whatever the model computes from older context is lost, and this is measured.

Program format, one statement per line:
    ngram vocab=65 order=4
    chars "<json string>"
    unit "<context as json string>" <vocab floats>    # top: e +2.31, a +1.10, ...
Units are in priority order: compiling with fewer units keeps the first ones.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from engine import wsp_synth

FORMAT_HEADER = "# WSPLang n-gram program: compile with step synthesize method=ngram table=<this file>"


class DecompileError(ValueError):
    """The weights do not follow the layout this decompiler reads."""


class UnitTable:
    """Ordered n-gram corrections; same interface as WittenBellTable for wsp_synth.compile_ngram."""

    def __init__(self, chars, order, units, provenance="closed_form"):
        if any(ch.isspace() for ch in provenance) or not provenance:
            raise DecompileError(f"provenance must be one word, got {provenance!r}")
        self.provenance = provenance
        self.chars = chars
        self.vocab = len(chars)
        self.order = order
        self._units = [(tuple(int(t) for t in c), np.asarray(v, dtype=np.float64)) for c, v in units]
        self._index = {c: v for c, v in self._units}
        if len(self._index) != len(self._units):
            raise DecompileError("duplicate context in unit list")
        for context, _ in self._units:
            if context and context[1:] not in self._index:
                raise DecompileError(f"unit {context} has no parent {context[1:]}: list is not suffix-closed")
            if len(context) >= order:
                raise DecompileError(f"unit {context} is longer than order {order} allows")

    @classmethod
    def from_table(cls, table, chars, limit=None):
        return cls(chars, table.order, list(table.units(limit=limit)))

    def num_units(self):
        return len(self._units)

    def units(self, limit=None):
        return iter(self._units[:limit])

    def log_prob(self, context):
        context = tuple(context)[-(self.order - 1):] if self.order > 1 else ()
        total = np.zeros(self.vocab)
        for k in range(len(context) + 1):
            suffix = context[len(context) - k:]
            if suffix not in self._index:
                break
            total = total + self._index[suffix]
        total = total - total.max()
        return total - np.log(np.exp(total).sum())


def _context_text(chars, context):
    return json.dumps("".join(chars[t] for t in context), ensure_ascii=False)


def format_program(table, source=None):
    lines = [FORMAT_HEADER]
    if source:
        lines.append(f"# source: {source}")
    lines.append(f"ngram vocab={table.vocab} order={table.order}")
    lines.append(f"provenance {getattr(table, 'provenance', 'closed_form')}")
    lines.append(f"chars {json.dumps(table.chars, ensure_ascii=False)}")
    for context, values in table.units():
        centered = values - values.mean()
        top = np.argsort(-centered)[:3]
        hint = ", ".join(f"{json.dumps(table.chars[t], ensure_ascii=False)} {centered[t]:+.2f}" for t in top)
        numbers = " ".join(f"{v:.9g}" for v in values)
        lines.append(f"unit {_context_text(table.chars, context)} {numbers}    # top: {hint}")
    return "\n".join(lines) + "\n"


def parse_program(text):
    decoder = json.JSONDecoder()
    vocab = order = chars = None
    provenance = "closed_form"
    units = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, rest = line.partition(" ")
        if keyword == "ngram":
            fields = dict(part.split("=", 1) for part in rest.split())
            vocab, order = int(fields["vocab"]), int(fields["order"])
        elif keyword == "chars":
            chars = json.loads(rest)
        elif keyword == "provenance":
            provenance = rest.strip()
        elif keyword == "unit":
            if chars is None:
                raise DecompileError(f"line {number}: unit before chars")
            word, end = decoder.raw_decode(rest)
            numbers = rest[end:].split("#", 1)[0].split()
            if len(numbers) != len(chars):
                raise DecompileError(f"line {number}: {len(numbers)} values, vocabulary has {len(chars)}")
            index = {ch: i for i, ch in enumerate(chars)}
            units.append((tuple(index[ch] for ch in word), np.array([float(v) for v in numbers])))
        else:
            raise DecompileError(f"line {number}: unknown statement {keyword!r}")
    if vocab is None or chars is None or vocab != len(chars):
        raise DecompileError("program needs 'ngram vocab=.. order=..' and a matching 'chars' line")
    return UnitTable(chars, order, units, provenance=provenance)


def _nonzero(array):
    return array.size > 0 and np.abs(array).max() > 0


def _check(condition, message):
    if not condition:
        raise DecompileError(message)


@torch.no_grad()
def decompile_ngram(model, chars):
    """Recovers the table compiled by wsp_synth.compile_ngram from the weights alone."""
    cfg = model.config
    d, heads, layers = cfg.d, cfg.heads, cfg.layers
    vocab, head_dim, code_dim = len(chars), cfg.d // cfg.heads, len(chars) - 1
    get = lambda name: model.get_parameter(name).detach().double().cpu().numpy()  # noqa: E731

    wte = get("transformer.wte.weight")
    _check(wte.shape == (vocab, d), f"wte has shape {wte.shape}, expected {(vocab, d)}")
    logit_dirs = wte.T / wsp_synth.TOKEN_NORM
    _check(np.allclose(logit_dirs.T @ logit_dirs, np.eye(vocab), atol=1e-3),
           "token embeddings are not an orthonormal readout basis")
    ballast = get("transformer.wpe.weight").mean(0)
    _check(abs(np.linalg.norm(ballast) / wsp_synth.BALLAST - 1) < 1e-2, "no ballast direction in position embeddings")
    ballast /= np.linalg.norm(ballast)

    attn_proj = get("transformer.h.0.attn.c_proj.weight")
    values = get("transformer.h.0.attn.c_attn.weight")[2 * d:]
    slot_dirs, codes = [], None
    for j in range(heads):
        block = attn_proj[:, j * head_dim:j * head_dim + code_dim] / wsp_synth.SLOT_NORM
        if not _nonzero(block):
            break
        _check(np.allclose(block.T @ block, np.eye(code_dim), atol=1e-3), f"head {j} does not write an orthonormal slot")
        _check(not _nonzero(attn_proj[:, j * head_dim + code_dim:(j + 1) * head_dim]),
               f"head {j} writes outside its token slot")
        slot_dirs.append(block)
        head_codes = values[j * head_dim:j * head_dim + code_dim] @ logit_dirs  # code_dim x vocab, scaled
        head_codes = head_codes.T / np.linalg.norm(head_codes.T, axis=1, keepdims=True)
        codes = head_codes if codes is None else codes
        _check(np.allclose(head_codes, codes, atol=1e-6), f"head {j} uses different token codes")
    _check(slot_dirs, "no attention head writes a token slot")
    _check(not _nonzero(attn_proj[:, len(slot_dirs) * head_dim:]),
           "extra attention heads: window or fitted layouts are not supported")
    order = len(slot_dirs) + 1

    activation = float(torch.nn.functional.gelu(torch.tensor(wsp_synth.UNIT_GAIN / 2, dtype=torch.float64)))
    units, ended = [], False
    for layer in range(layers):
        c_fc = get(f"transformer.h.{layer}.mlp.c_fc.weight")
        c_proj = get(f"transformer.h.{layer}.mlp.c_proj.weight")
        for unit in range(c_fc.shape[0]):
            row = c_fc[unit]
            if not _nonzero(row):
                ended = True
                continue
            _check(not ended, f"layer {layer} unit {unit} follows an empty unit")
            context = []
            for k, slot in enumerate(slot_dirs):
                part = row @ slot  # input_gain * code
                if np.linalg.norm(part) < 1e-6 * max(1.0, np.linalg.norm(row)):
                    break
                cosine = codes @ part / np.linalg.norm(part)
                token = int(np.argmax(cosine))
                _check(cosine[token] > 0.999, f"layer {layer} unit {unit} slot {k} matches no token")
                context.append(token)
            context = tuple(reversed(context))
            residue = row - ballast * (row @ ballast) - sum(s @ (s.T @ row) for s in slot_dirs)
            _check(np.linalg.norm(residue) < 1e-3 * np.linalg.norm(row), f"layer {layer} unit {unit} is not a table unit")
            correction = logit_dirs.T @ c_proj[:, unit] * activation
            if len(context) == 1:
                correction[context[0]] += wsp_synth.TOKEN_NORM
            units.append((context, correction))
    _check(units, "no table units found")
    return UnitTable(chars, order, units)


@torch.no_grad()
def behavioral_table(model, ids, chars, order, device=None, batch=64):
    """The model's average prediction after each context of length < order, over every occurrence in `ids`.

    Each window is scored as a whole, so the model sees up to its block size of history; positions whose context
    would start before the window are skipped. Units are ranked like WittenBellTable: frequency, then length.
    """
    vocab = len(chars)
    device = device or next(model.parameters()).device
    block = model.config.block_size
    ids = np.asarray(ids, dtype=np.int64)
    count = (len(ids) - 1) // block
    windows = torch.tensor(np.stack([ids[i * block:i * block + block] for i in range(count)]))

    keys = [[] for _ in range(order)]
    probs = []
    for start in range(0, count, batch):
        x = windows[start:start + batch].to(device)
        p = torch.softmax(model(x, x)[0].double(), -1).cpu()
        for length in range(order):
            positions = torch.arange(max(length - 1, 0), block)
            key = torch.zeros(x.shape[0], len(positions), dtype=torch.int64)
            for back in range(length):
                key = key * vocab + x[:, positions - back].cpu()
            keys[length].append(key.flatten())
        probs.append(p)

    full = torch.cat([p for p in probs]).view(-1, block, vocab)
    log_p, counts = {}, {}
    for length in range(order):
        start = max(length - 1, 0)
        rows = full[:, start:].reshape(-1, vocab)
        unique, inverse, freq = torch.unique(torch.cat(keys[length]), return_inverse=True, return_counts=True)
        sums = torch.zeros(len(unique), vocab, dtype=torch.float64).index_add_(0, inverse, rows)
        means = (sums / freq[:, None]).clamp_min(1e-12).log().numpy()
        for key, row, n in zip(unique.tolist(), means, freq.tolist()):
            context = []
            for _ in range(length):
                context.append(key % vocab)
                key //= vocab
            context = tuple(context)  # least significant digit is the oldest token, so this is oldest first
            log_p[context] = row
            counts[context] = n

    ranked = sorted(log_p, key=lambda c: (-counts[c], len(c), c))
    units = [(c, log_p[c] - (log_p[c[1:]] if c else 0.0)) for c in ranked]
    table = UnitTable(chars, order, units)
    table.counts = counts
    return table


def retained_fraction(nll, reference_nll, floor_nll):
    """Share of the reference model's gain over `floor_nll` (e.g. unigram entropy) that `nll` keeps."""
    return (floor_nll - nll) / (floor_nll - reference_nll) if not math.isclose(floor_nll, reference_nll) else float("nan")
