import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class TestMakeKey(unittest.TestCase):
    def test_matches_the_recipe_used_by_tests_nanowsp_and_is_deterministic(self):
        np.random.seed(42)
        expect = np.random.randn(64).astype(np.float32)
        expect = expect / np.linalg.norm(expect)
        with tempfile.TemporaryDirectory() as d:
            outs = []
            for i in range(2):
                out = Path(d) / f"k{i}.npz"
                r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_nanowsp_key.py"), "--out", str(out)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                outs.append(np.load(out)["u"])
        self.assertTrue(np.array_equal(outs[0], expect))
        self.assertTrue(np.array_equal(outs[0], outs[1]))
        self.assertAlmostEqual(float(np.linalg.norm(outs[0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
