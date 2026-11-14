"""An op the runtime advertises must run; one it cannot run must be rejected by the validator.
distributed_memit was listed as implemented with no executor: validated, then 'skipped' in silence (2026-11-10)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine.wsplang as wsplang


class TestPhantomOps(unittest.TestCase):
    def test_distributed_memit_is_rejected_not_skipped(self):
        prog = wsplang.parse_args_lines("experiment t:\nsteps:\n    step distributed_memit layers=1 rank=2\n")
        _, errors = wsplang.validate(prog)
        self.assertTrue(any("distributed_memit" in e for e in errors), errors)
        self.assertNotIn("distributed_memit", wsplang.IMPLEMENTED_OPS)


if __name__ == "__main__":
    unittest.main()
