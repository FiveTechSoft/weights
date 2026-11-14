"""lab/probes/delta_patterns/gguf_q8_delta.py: re-quantizing a bf16 base to Q8_0 must reproduce gguf-py's bytes, and the
value-head alignment must undo a column permutation of heads."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "delta_patterns"))

import gguf_q8_delta as g  # noqa: E402


class TestQ8(unittest.TestCase):
    def test_both_recipes_match_gguf_py_and_dequantize_back(self):
        import gguf
        x = (np.random.default_rng(0).standard_normal((64, 256)) * 0.02).astype(np.float32)
        ref = np.asarray(gguf.quants.quantize(x, gguf.GGMLQuantizationType.Q8_0)).reshape(-1, 34)
        self.assertTrue((g.q8_numpy(x) == ref).all())
        self.assertGreater((g.q8_cref(x) == ref).all(axis=1).mean(), 0.99)
        self.assertLess(np.abs(g.deq8(ref).reshape(64, 256) - x).max(), 1e-3)

    def test_head_order_undoes_a_head_permutation(self):
        rng = np.random.default_rng(1)
        B = rng.standard_normal((16, 6 * 8)).astype(np.float32)
        perm = [3, 0, 5, 1, 4, 2]
        M = np.concatenate([B[:, j * 8:(j + 1) * 8] for j in perm], axis=1)
        self.assertTrue(np.array_equal(B[:, g.head_order(M, B, 8)], M))


if __name__ == "__main__":
    unittest.main()
