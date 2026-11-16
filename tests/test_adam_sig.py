"""lab/probes/adam_sig/adam_sig.py: Adam / weight-decay footprint in embedding rows (Chaudhuri, X thread 14 Nov 2026;
lab/probes/adam_sig/PLAN.md). Row classes on D = donor - base: exactly zero; pure decay (D = -c w, one c per row, small
residual); single-step (|D_ij| nearly constant across j); the rest. Tested on synthetic rows where the class is known."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestRowClasses(unittest.TestCase):
    def setUp(self):
        g = np.random.default_rng(0)
        self.W = g.normal(size=(6, 64)).astype(np.float32)
        D = np.zeros_like(self.W)
        D[1] = -0.001 * self.W[1]                         # pure decay, c = 0.001
        D[2] = 0.0003 * g.choice([-1.0, 1.0], size=64)    # single step, |D| constant
        D[3] = 0.0003 * g.choice([-1.0, 1.0], size=64) + 0.00001 * g.normal(size=64)   # single step, bf16-ish jitter
        D[4] = g.normal(size=64) * 0.01                   # ordinary change
        D[5] = -0.001 * self.W[5] + 0.0003 * g.choice([-1.0, 1.0], size=64)            # decay + one step: "rest"
        self.D = D

    def test_classes(self):
        from lab.probes.adam_sig.adam_sig import classify
        c = classify(self.W, self.D)
        self.assertEqual(c["cls"].tolist(), ["zero", "decay", "step", "step", "other", "other"])
        self.assertAlmostEqual(float(c["decay_c"][1]), 0.001, places=6)
        self.assertAlmostEqual(float(c["step_mag"][2]), 0.0003, places=6)

    def test_chunked_equals_whole(self):
        from lab.probes.adam_sig.adam_sig import classify, classify_chunked
        a = classify(self.W, self.D)
        b = classify_chunked(self.W, self.D, chunk=4)
        self.assertEqual(a["cls"].tolist(), b["cls"].tolist())
        np.testing.assert_allclose(a["norm"], b["norm"])

    def test_two_methods_agree_on_zero_count(self):
        from lab.probes.adam_sig.adam_sig import zero_rows_a, zero_rows_b
        self.assertEqual(zero_rows_a(self.D), zero_rows_b(self.D))
        self.assertEqual(zero_rows_a(self.D), 1)


if __name__ == "__main__":
    unittest.main()
