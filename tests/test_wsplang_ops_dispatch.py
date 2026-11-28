"""test_wsplang_ops_dispatch.py — structural and smoke tests for the extracted dispatch layer.

The dispatch chain for execute_plan_v4 lives in engine/wsplang_ops.py. These tests pin:
  - that every handler named by the chain exists, has the canonical signature and is
    present in the ops module (so a refactor can't silently drop or misname a branch);
  - that each op in the chain reaches its handler and produces a well-formed result —
    a handler must either return a completed step record or raise a benign error that
    the loop converts to status=error (missing artifacts/args). Anything else — an
    undefined name, a wrong attribute on the context — fails the test.
"""
import ast
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.wsplang_ops as ops_module  # noqa: E402
import engine.wsplang as wsplang  # noqa: E402

OPS_SRC = (ROOT / "engine" / "wsplang_ops.py").read_text(encoding="utf-8")
KNOWN_STATUSES = {"ok", "simulated", "unsupported_backend", "stub_result",
                  "security_rejection", "assertion_failed", "error",
                  "skipped (non-interactive op)"}


def dispatch_tests():
    """The if/elif tests in the dispatch, in order, plus the single else body."""
    tree = ast.parse(OPS_SRC)
    dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
    tests = []
    node = dispatch.body[1]
    while isinstance(node, ast.If):
        tests.append(ast.unparse(node.test))
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
        else:
            break
    return tests


def dispatch_ops():
    """Every op reachable by the chain (expanded from compound ops), deduped."""
    ops = set()
    for t in dispatch_tests():
        m = re.match(r"op == ['\"]([^'\"]+)['\"] and", t)
        if m:
            ops.add(m.group(1))
            continue
        m = re.match(r"op == ['\"]([^'\"]+)['\"]", t)
        if m:
            ops.add(m.group(1))
            continue
        m = re.match(r"op in \((.*)\)$", t)
        if m:
            for v in ast.parse(m.group(1), mode="eval").body.elts:
                ops.add(v.value)
    return sorted(ops)


class TestDispatchStructure(unittest.TestCase):
    def test_dispatch_chain_has_59_branches(self):
        self.assertEqual(len(dispatch_tests()), 59)

    def test_every_handler_named_by_the_chain_exists_with_signature(self):
        tree = ast.parse(OPS_SRC)
        handlers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(len(handlers), 61, "handlers, helpers and dispatch")

        # each test in the chain calls exactly one handler
        dispatch_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
        called = []
        node = dispatch_fn.body[1]
        while isinstance(node, ast.If):
            calls = [s.value for s in node.body if isinstance(s, ast.Return) and s.value is not None]
            self.assertEqual(len(calls), 1, f"if/elif at {ast.unparse(node.test)!r} has one call")
            handler_names = [c.func.id for c in calls if isinstance(c, ast.Call)]
            self.assertEqual(len(handler_names), 1)
            called.append(handler_names[0])
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                node = node.orelse[0]
            else:
                break
        self.assertEqual(len(called), 59)
        for name in called:
            handler = handlers.get(name)
            self.assertIsNotNone(handler, f"{name} missing from module")
            args = [a.arg for a in handler.args.posonlyargs + handler.args.args]
            self.assertEqual(args, ["ctx", "op", "positional", "named", "res"],
                             f"{name} signature drifted from the dispatch contract")

    def test_else_branch_skips_non_interactive_ops(self):
        tree = ast.parse(OPS_SRC)
        dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
        # final else in the dispatch body
        node = dispatch.body[1]
        while isinstance(node, ast.If) and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
        else_src = "\n".join(ast.unparse(stmt) for stmt in node.orelse)
        self.assertIn("skipped (non-interactive op", else_src)


class TestDispatchSmoke(unittest.TestCase):
    def _minimal_context(self, tmpdir):
        tmp = Path(tmpdir)
        return ops_module.PlanContext(
            budgets={},
            max_modified_weights=None,
            max_delta_norm=None,
            allow_write=False,
            written_paths=set(),
            target_path=tmp,
            is_safetensors=True,
            default_gguf=Path("/nonexistent-gguf"),
            default_safetensors=tmp / "default-safetensors",
            nanowsp_default=tmp / "nanowsp" / "model.safetensors",
            nanowsp2_default=tmp / "nanowsp2" / "model.safetensors",
            nanowsp3_default=tmp / "nanowsp3" / "model.safetensors",
            tiny_default=tmp / "tinyllama" / "model.safetensors",
            qwen08_default=tmp / "qwen08" / "model.safetensors",
            out_dir=None,
            model_path=None,
            dry_run=True,
            executed_steps=[],
        )

    def test_every_op_reaches_its_handler_and_produces_a_step_result(self):
        prog = {"model": [("x", "model:test")], "steps": [], "__root__": []}
        for op in dispatch_ops():
            with self.subTest(op=op):
                plan = {"pasos": [{"op": op, "args": ""}], "budgets": {}, "theoretical_framework": "test"}
                with tempfile.TemporaryDirectory() as td:
                    results = wsplang.execute_plan_v4(plan, prog, model_path=td, dry_run=True)
                self.assertEqual(len(results), 1, f"{op}: expected one step record")
                res = results[0]
                self.assertEqual(res["op"], op)
                self.assertIn(res["status"], KNOWN_STATUSES, f"{op}: unexpected status {res['status']!r}")


if __name__ == "__main__":
    unittest.main()
