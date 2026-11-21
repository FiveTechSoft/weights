"""lab/probes/discrete_writer/writer.py (roadmap gap 20, lab/probes/discrete_writer/PLAN.md): a discrete per-row writer
in the bf16 alphabet (sign flip, x2, x1/2 per entry), exact separable threshold solve and a coupled greedy solve on
activations. Written before the code."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.probes.discrete_writer import writer as dw  # noqa: E402


class Bf16MovesTest(unittest.TestCase):
    def test_to_bf16_rounds_to_nearest_even_and_moves_are_exact(self):
        x = np.array([1.0, 0.1, -3.0, 1e-3], dtype=np.float32)
        b = dw.to_bf16(x)
        self.assertEqual(b.dtype, np.float32)
        self.assertTrue(np.all(dw.to_bf16(b) == b))                     # idempotent
        self.assertAlmostEqual(float(b[1]), 0.10009765625, places=12)   # 0.1 in bf16
        m = dw.moves(b)                                                  # (3, n): flip, x2, half
        self.assertEqual(m.shape, (3, 4))
        np.testing.assert_array_equal(m[0], -b)
        np.testing.assert_array_equal(m[1], 2 * b)
        np.testing.assert_array_equal(m[2], b / 2)
        self.assertTrue(np.all(dw.to_bf16(m) == m))                     # every move is a valid bf16 value

    def test_bit_flips_counts_hamming_distance_of_bf16_codes(self):
        a = dw.to_bf16(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        b = np.array([-1.0, 2.0, 1.0], dtype=np.float32)
        self.assertEqual(dw.bit_flips(a, b), 1 + 8)                      # sign bit; x2 takes the exponent 127 -> 128 (8 bits)
        c = dw.to_bf16(np.array([3.0], dtype=np.float32))                # exponent 128 -> 129 flips one bit
        self.assertEqual(dw.bit_flips(c, 2 * c), 1)


class ThresholdTest(unittest.TestCase):
    def test_recovers_a_target_made_of_admissible_moves(self):
        rng = np.random.default_rng(0)
        w = dw.to_bf16(rng.normal(size=64).astype(np.float32) * 0.01)
        code = rng.integers(0, 4, size=64)                                # 0 none, 1 flip, 2 x2, 3 half
        new = np.where(code == 1, -w, np.where(code == 2, 2 * w, np.where(code == 3, w / 2, w))).astype(np.float32)
        t = new - w
        d, chosen, info = dw.discrete_row_threshold(w, t, lam=0.0)
        np.testing.assert_array_equal(chosen, code)
        np.testing.assert_allclose(d, t, atol=0)
        self.assertEqual(info["changed"], int((code != 0).sum()))

    def test_lambda_large_changes_nothing_and_zero_changes_every_beneficial_entry(self):
        rng = np.random.default_rng(1)
        w = dw.to_bf16(rng.normal(size=256).astype(np.float32) * 0.01)
        t = (rng.normal(size=256) * 0.004).astype(np.float32)
        d, chosen, info = dw.discrete_row_threshold(w, t, lam=1e9)
        self.assertEqual(info["changed"], 0)
        self.assertTrue(np.all(d == 0))
        d0, chosen0, info0 = dw.discrete_row_threshold(w, t, lam=0.0)
        gain = dw.gains(w, t)                                             # t^2 - min_m (d_m - t)^2 per entry
        self.assertEqual(info0["changed"], int((gain > 0).sum()))
        # each chosen move lowers the per-entry loss, and no better move exists
        m = dw.moves(w) - w
        for j in range(256):
            if chosen0[j]:
                self.assertLess((d0[j] - t[j]) ** 2, t[j] ** 2)
                self.assertLessEqual((d0[j] - t[j]) ** 2, ((m[:, j] - t[j]) ** 2).min() + 1e-12)

    def test_lambda_for_fraction_hits_the_requested_share(self):
        rng = np.random.default_rng(2)
        w = dw.to_bf16(rng.normal(size=(8, 512)).astype(np.float32) * 0.01)
        t = (rng.normal(size=(8, 512)) * 0.01).astype(np.float32)
        lam = dw.lam_for_fraction(w, t, 0.05)
        d, chosen, info = dw.discrete_threshold(w, t, lam)
        self.assertAlmostEqual(info["changed"] / w.size, 0.05, delta=0.01)


class CoupledTest(unittest.TestCase):
    def test_greedy_never_increases_the_objective(self):
        rng = np.random.default_rng(3)
        n, p = 40, 12
        w = dw.to_bf16(rng.normal(size=n).astype(np.float32) * 0.01)
        t = (rng.normal(size=n) * 0.003).astype(np.float32)
        X = rng.normal(size=(p, n)).astype(np.float32)
        d, chosen, trace = dw.discrete_row_anneal(w, t, X, lam=0.0, sweeps=3, seed=0)
        self.assertGreaterEqual(len(trace), 1)
        for a, b in zip(trace, trace[1:]):
            self.assertLessEqual(b, a + 1e-9)
        obj = float(((X @ (d - t)) ** 2).sum()) + 0.0 * int((chosen != 0).sum())
        self.assertAlmostEqual(obj, trace[-1], places=4)
        self.assertLessEqual(trace[-1], float(((X @ (-t)) ** 2).sum()) + 1e-9)   # no worse than doing nothing


if __name__ == "__main__":
    unittest.main()
