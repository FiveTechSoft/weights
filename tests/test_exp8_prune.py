"""Pruning of the Exp 8 edit (lab/probes/exp8_prune/PLAN.md): keep the k largest entries by |value| per layer of the
recovered edit (data/exp8_edit_recovered.npz), or their complement, and report how much of the edit survives.
Written before prune.py."""
import os
import tempfile
import unittest

import numpy as np

from lab.probes.exp8_prune.prune import prune_npz, stats, topk_mask


class TopkMaskTest(unittest.TestCase):
    def test_keeps_exactly_k_largest_by_abs(self):
        rng = np.random.default_rng(0)
        V = rng.normal(size=(50, 8)).astype(np.float32)
        m = topk_mask(V, 37)
        self.assertEqual(m.shape, V.shape)
        self.assertEqual(int(m.sum()), 37)
        self.assertGreaterEqual(np.abs(V[m]).min(), np.abs(V[~m]).max())

    def test_k_at_least_size_keeps_everything(self):
        V = np.ones((3, 4), dtype=np.float32)
        self.assertEqual(int(topk_mask(V, 12).sum()), 12)


class StatsTest(unittest.TestCase):
    def test_energy_and_push_fraction_on_rank_one(self):
        u = np.array([3.0, 0.0, 4.0])  # norm 5
        v = np.array([1.0, 2.0])
        V = np.outer(u, v)  # entries: [3,6],[0,0],[4,8]
        m = np.zeros_like(V, dtype=bool); m[2, 1] = True  # keep the 8 only
        s = stats(V, m)
        self.assertEqual(s["kept"], 1)
        self.assertAlmostEqual(s["energy_frac"], 64 / 125, places=6)
        # push along u that survives: ||u^T (V*m)|| / ||u^T V|| = (4*8 / 5) / (5*sqrt(5))
        self.assertAlmostEqual(s["push_frac"], (32 / 5) / (5 * np.sqrt(5)), places=6)
        self.assertAlmostEqual(s["cos_to_full"], 8 / np.sqrt(125), places=6)


class PruneNpzTest(unittest.TestCase):
    def test_kept_plus_complement_is_the_original_and_cols_untouched(self):
        rng = np.random.default_rng(1)
        src = {"cols_0": np.array([0, 2, 5]), "vec_0": rng.normal(size=(10, 3)).astype(np.float32),
               "cols_1": np.array([1, 4, 7]), "vec_1": rng.normal(size=(10, 3)).astype(np.float32)}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "src.npz"); np.savez(p, **src)
            k_path = os.path.join(d, "k.npz"); c_path = os.path.join(d, "c.npz")
            sk = prune_npz(p, 7, k_path)
            sc = prune_npz(p, 7, c_path, complement=True)
            with np.load(k_path) as zk, np.load(c_path) as zc:  # Windows: close before the tempdir is removed
                for i in (0, 1):
                    np.testing.assert_array_equal(zk[f"cols_{i}"], src[f"cols_{i}"])
                    np.testing.assert_array_equal(zc[f"cols_{i}"], src[f"cols_{i}"])
                    np.testing.assert_allclose(zk[f"vec_{i}"] + zc[f"vec_{i}"], src[f"vec_{i}"])
                    self.assertEqual(int((zk[f"vec_{i}"] != 0).sum()), 7)
                    self.assertEqual(sk[f"layer_{i}"]["kept"], 7)
                    self.assertEqual(sc[f"layer_{i}"]["kept"], 23)
                    self.assertAlmostEqual(sk[f"layer_{i}"]["energy_frac"] + sc[f"layer_{i}"]["energy_frac"], 1.0,
                                           places=5)


if __name__ == "__main__":
    unittest.main()
