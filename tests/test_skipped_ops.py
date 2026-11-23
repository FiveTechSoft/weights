"""The in-engine skip registry: every declared-but-not-executed op gets a reason."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.wsplang as wsplang
import engine.wsplang_ops as ops_module


class TestSkippedOpsRegistry(unittest.TestCase):

    def test_every_registry_entry_is_an_allowed_op_not_dispatched(self):
        from tests.test_wsplang_ops_dispatch import dispatch_ops
        dispatched = set(dispatch_ops())
        for op in ops_module.SKIPPED_OPS:
            self.assertIn(op, wsplang.OPS_ALLOWED, f"{op} not in OPS_ALLOWED")
            self.assertNotIn(op, dispatched, f"{op} is dispatched but registered as skipped")

    def test_every_declared_but_not_dispatched_op_is_registered(self):
        from tests.test_wsplang_ops_dispatch import dispatch_ops
        dispatched = set(dispatch_ops())
        unhandled = wsplang.OPS_ALLOWED - dispatched
        registered = set(ops_module.SKIPPED_OPS)
        self.assertEqual(unhandled, registered,
                         f"unregistered ops: {sorted(unhandled - registered)}; "
                         f"stale registry entries: {sorted(registered - unhandled)}")

    def test_each_skipped_op_reports_reason_at_runtime(self):
        prog = {"model": [("x", "model:test")], "steps": [], "__root__": []}
        for op, reason in ops_module.SKIPPED_OPS.items():
            with self.subTest(op=op):
                plan = {"pasos": [{"op": op, "args": ""}], "budgets": {}, "theoretical_framework": "test"}
                with tempfile.TemporaryDirectory() as td:
                    results = wsplang.execute_plan_v4(plan, prog, model_path=td, dry_run=True)
                self.assertEqual(len(results), 1)
                res = results[0]
                self.assertEqual(res["status"], "skipped (non-interactive op: " + reason + ")")


if __name__ == "__main__":
    unittest.main()
