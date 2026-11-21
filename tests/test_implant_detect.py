import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab" / "probes" / "implant_detect"))
import detect as dt  # noqa: E402


class RobustZ(unittest.TestCase):
    def test_outlier_scores_high(self):
        x = np.r_[np.random.default_rng(0).normal(1, 0.1, 999), 5.0]
        z = dt.robust_z(x, *dt.med_mad(x))
        self.assertGreater(z[-1], 20)
        self.assertLess(np.abs(z[:-1]).max(), 6)

    def test_zero_mad_flags_any_change(self):
        ref = np.zeros(100)
        med, mad = dt.med_mad(ref)
        z = dt.robust_z(np.array([0.0, 1e-6]), med, mad)
        self.assertEqual(z[0], 0)
        self.assertTrue(np.isinf(z[1]))


class Threshold(unittest.TestCase):
    def test_at_most_one_flag_on_calibration(self):
        s = np.array([0.1, 3.0, 2.0, 9.0, 1.0])
        tau = dt.tau_from(s)
        self.assertEqual(int((s > tau).sum()), 1)

    def test_all_equal_flags_none(self):
        self.assertEqual(int((np.zeros(5) > dt.tau_from(np.zeros(5))).sum()), 0)


class Implants(unittest.TestCase):
    def test_rewrite_keeps_norm(self):
        g = np.random.default_rng(1)
        v = g.normal(size=64).astype(np.float32)
        w = dt.rewrite(v, g)
        self.assertAlmostEqual(float(np.linalg.norm(w)), float(np.linalg.norm(v)), places=4)
        self.assertLess(abs(float(w @ v)) / float(v @ v), 0.6)

    def test_push_has_requested_norm(self):
        g = np.random.default_rng(2)
        v = np.ones(64, dtype=np.float32)
        w = dt.push(v, 0.5, g)
        self.assertAlmostEqual(float(np.linalg.norm(w - v)), 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
