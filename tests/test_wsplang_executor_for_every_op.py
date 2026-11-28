"""Generalisation of test_wsplang_phantom_ops: every op the validator accepts must reach a real handler.

dispatch() falls back to "skipped (non-interactive op)" when no elif matches; an op listed in
IMPLEMENTED_OPS with no handler validates and then silently does nothing (the distributed_memit
bug of 2026-11-10, which this file checks for the whole catalogue, not one name).

If torch is missing the module is imported with a stub: the handlers only touch torch inside
function bodies (torch.save / torch.load), so dispatch routing can be tested without it. The
stub is removed from sys.modules afterwards so later imports of torch still fail loudly
instead of silently binding to the stub.
"""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.wsplang as w  # noqa: E402

_stubbed_torch = False
try:
    import engine.wsplang_ops as ops  # noqa: E402
except ImportError:
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    _stubbed_torch = True
    import engine.wsplang_ops as ops  # noqa: E402
if _stubbed_torch:
    sys.modules.pop("torch", None)


class _Ctx:
    """Minimal stand-in: a phantom op must not depend on ctx to reveal itself."""


class TestExecutorForEveryOp(unittest.TestCase):
    def test_every_implemented_op_reaches_a_handler(self):
        phantoms = []
        for op in sorted(w.IMPLEMENTED_OPS):
            res = {"op": op, "args": "", "status": "ok"}
            try:
                ops.dispatch(_Ctx(), op, [], {}, res)
            except BaseException:
                continue  # a handler exists and raised on the dummy ctx: routing is fine
            if str(res.get("status", "")).startswith("skipped"):
                phantoms.append(f"{op}: {res['status']}")
        self.assertFalse(phantoms, f"IMPLEMENTED_OPS with no dispatch handler: {phantoms}")

    def test_implemented_and_skipped_are_disjoint(self):
        self.assertFalse(w.IMPLEMENTED_OPS & set(ops.SKIPPED_OPS),
                         sorted(w.IMPLEMENTED_OPS & set(ops.SKIPPED_OPS)))

    def test_stub_ops_are_implemented_and_routed(self):
        self.assertFalse(w.STUB_OPS - w.IMPLEMENTED_OPS, sorted(w.STUB_OPS - w.IMPLEMENTED_OPS))
        for op in sorted(w.STUB_OPS):
            res = {"op": op, "args": "", "status": "ok"}
            try:
                ops.dispatch(_Ctx(), op, [], {}, res)
            except BaseException:
                continue
            self.assertFalse(str(res.get("status", "")).startswith("skipped"),
                             f"stub {op} has no handler either")


if __name__ == "__main__":
    unittest.main()
