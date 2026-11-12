"""lab/probes/cc_weights/verdict.py: the run directories come from the environment so the recompilation with the
corrected vector (PLAN.md, 2026-11-12) is judged by the same code as the first one, on its own files."""
import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "cc_weights"))


def load(env):
    old = {k: os.environ.get(k) for k in ("CCW_CC", "CCW_RUNS", "CCW_SEALED")}
    for k in old:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        import verdict
        return importlib.reload(verdict)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


class TestDirs(unittest.TestCase):
    def test_defaults_are_the_first_compilation(self):
        v = load({})
        self.assertEqual(v.CC, ROOT / "lab" / "runs" / "cc")
        self.assertEqual(v.W, ROOT / "lab" / "runs" / "cc_weights")
        self.assertEqual(v.SEALED, ROOT / "lab" / "runs" / "sealed_ccw")

    def test_environment_moves_all_three(self):
        v = load({"CCW_CC": "lab/runs/cc_v2", "CCW_RUNS": "lab/runs/cc_v2/weights", "CCW_SEALED": "lab/runs/sealed_ccw_v2"})
        self.assertEqual(v.CC, ROOT / "lab" / "runs" / "cc_v2")
        self.assertEqual(v.W, ROOT / "lab" / "runs" / "cc_v2" / "weights")
        self.assertEqual(v.SEALED, ROOT / "lab" / "runs" / "sealed_ccw_v2")


if __name__ == "__main__":
    unittest.main()
