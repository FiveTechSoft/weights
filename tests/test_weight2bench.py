"""lab/probes/weight2bench (plan fixed first): weight-only features of an edit vs its measured benchmark change.
Written before features.py / run.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.probes.weight2bench import features as F  # noqa: E402


class MagnitudeTest(unittest.TestCase):
    def test_equal_orthogonal_rows(self):
        D = np.eye(4, 8, dtype=np.float32) * 2.0          # 4 layers, norms 2, orthogonal
        m = F.mag_features(D)
        self.assertAlmostEqual(m["mag_total"], 4.0, places=5)
        self.assertAlmostEqual(m["mag_mean"], 2.0, places=5)
        self.assertAlmostEqual(m["mag_max"], 2.0, places=5)
        self.assertAlmostEqual(m["participation"], 1.0, places=5)
        self.assertAlmostEqual(m["eff_rank"], 4.0, places=4)

    def test_one_dominant_layer(self):
        D = np.zeros((5, 6), dtype=np.float32); D[3, 0] = 3.0; D[0, 1] = 1.0
        m = F.mag_features(D)
        self.assertAlmostEqual(m["argmax_depth"], 4 / 5, places=6)   # layer index 3 of 5 -> (3+1)/5
        self.assertLess(m["participation"], 0.5)
        self.assertLess(m["eff_rank"], 2.0)


class CosineTest(unittest.TestCase):
    def test_aligned_and_orthogonal(self):
        r_close = np.array([1.0, 0, 0, 0]); r_open = np.array([0, 1.0, 0, 0])
        D = np.array([[2.0, 0, 0, 0], [0, 0, 1.0, 0], [0.5, 0, 0, 0]])
        c = F.cos_features(D, r_close, r_open, late=2)
        self.assertAlmostEqual(c["cos_close_mean"], 2 / 3, places=6)
        self.assertAlmostEqual(c["cos_close_late8"], 0.5, places=6)   # last 2 layers: 0 and 1
        self.assertAlmostEqual(c["cos_close_last"], 1.0, places=6)
        self.assertAlmostEqual(c["cos_open_mean"], 0.0, places=6)
        self.assertAlmostEqual(c["cos_gap"], 2 / 3, places=6)

    def test_cos_between_uses_common_layers(self):
        A = {1: np.array([1.0, 0]), 2: np.array([0, 1.0]), 3: np.array([1.0, 1.0])}
        B = {1: np.array([1.0, 0]), 2: np.array([1.0, 0])}
        self.assertAlmostEqual(F.cos_between(A, B), 0.5, places=6)

    def test_readout_applies_final_norm(self):
        np.testing.assert_allclose(F.readout(np.array([1.0, 2.0]), np.array([3.0, 0.5])), [3.0, 1.0])


class StatsTest(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(F.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=9)
        self.assertAlmostEqual(F.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, places=9)
        # ties: x ranks (1.5, 1.5, 3, 4), y ranks (1, 2, 3, 4) -> Pearson of the ranks = 0.9487
        self.assertAlmostEqual(F.spearman([1, 1, 2, 3], [1, 2, 3, 4]), 0.9486833, places=5)
        self.assertTrue(np.isnan(F.spearman([1, 1], [2, 3])))

    def test_auc(self):
        self.assertAlmostEqual(F.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(F.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)
        self.assertAlmostEqual(F.auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)
        self.assertAlmostEqual(F.auc([0.9, 0.5, 0.5, 0.1], [1, 0, 1, 0]), 0.875)   # 1 + 1 + 0.5 (tie) + 1 over 4 pairs


class LoadersTest(unittest.TestCase):
    def test_cols_block_gives_top_direction_and_frobenius_magnitude(self):
        rng = np.random.default_rng(0)
        u = rng.normal(size=16); u /= np.linalg.norm(u)
        V = np.outer(u, rng.normal(size=5)).astype(np.float32)      # rank-1 block, 5 columns
        d = F.cols_direction(V)
        self.assertAlmostEqual(abs(float(d @ u)) / np.linalg.norm(d), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(d)), float(np.linalg.norm(V)), places=5)
        self.assertGreaterEqual(float(d @ V.mean(axis=1)), 0.0)     # sign fixed toward the mean column

    def test_compiled_direction_is_v_scaled_by_u_norm(self):
        u = np.array([3.0, 4.0]); v = np.array([0.0, 1.0, 0.0])
        d = F.compiled_direction(u, v)
        np.testing.assert_allclose(d, [0.0, 5.0, 0.0])

    def test_targets_from_run_summaries(self):
        base = {"summary": {"gsm8k": {"n": 100, "acc": 0.50, "acc_independent": 0.49, "cut_at_max": 51, "mean_tokens": 1600},
                            "math500": {"n": 50, "acc": 0.06, "acc_independent": 0.06, "cut_at_max": 45, "mean_tokens": 2000}}}
        arm = {"summary": {"gsm8k": {"n": 100, "acc": 0.83, "acc_independent": 0.84, "cut_at_max": 8, "mean_tokens": 800},
                           "math500": {"n": 50, "acc": 0.34, "acc_independent": 0.28, "cut_at_max": 17, "mean_tokens": 1400}}}
        t = F.targets(arm, base)
        self.assertEqual(t["d_gsm"], 33); self.assertEqual(t["d_gsm_g2"], 35)
        self.assertEqual(t["d_math"], 14); self.assertEqual(t["d_cut"], (8 - 51) + (17 - 45))
        pooled_base = (1600 * 100 + 2000 * 50) / 150; pooled_arm = (800 * 100 + 1400 * 50) / 150
        self.assertAlmostEqual(t["d_tok"], 100 * (pooled_arm / pooled_base - 1), places=6)
        self.assertEqual(t["n"], 150)

    def test_targets_math_only_run(self):
        base = {"summary": {"math500": {"n": 64, "acc": 0.125, "acc_independent": 0.125, "cut_at_max": 50, "mean_tokens": 1900}}}
        arm = {"summary": {"math500": {"n": 64, "acc": 0.5, "acc_independent": 0.5, "cut_at_max": 10, "mean_tokens": 900}}}
        t = F.targets(arm, base)
        self.assertIsNone(t["d_gsm"]); self.assertEqual(t["d_math"], 24); self.assertEqual(t["n"], 64)

    def test_row_builder_reads_paired_files(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, acc in (("base", 0.5), ("arm", 0.6)):
                (d / f"{name}.json").write_text(json.dumps({"summary": {"gsm8k": {"n": 10, "acc": acc, "acc_independent": acc,
                                                                                   "cut_at_max": 0, "mean_tokens": 100}}}), encoding="utf-8")
            row = F.target_row(str(d / "arm.json"), str(d / "base.json"))
            self.assertEqual(row["d_gsm"], 1)
            self.assertEqual(row["n"], 10)


if __name__ == "__main__":
    unittest.main()
