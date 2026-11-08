"""H8 (lab/probes/delta_patterns/h8_plan.md): decomposition of C_200 along C_k, and the library of summed C."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h8"))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h7"))
sys.path.insert(0, str(ROOT / "lab" / "probes" / "h6"))


def _flat(D):
    return torch.cat([D[n].double().flatten() for n in sorted(D)])


class TestDecompose(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(3)
        self.Ck = {"a": torch.randn(3, 4, generator=g), "b": torch.randn(5, generator=g)}
        self.CT = {"a": torch.randn(3, 4, generator=g), "b": torch.randn(5, generator=g)}

    def test_parts_add_up_and_off_is_orthogonal(self):
        import h8
        P, O, a, share = h8.decompose(self.Ck, self.CT)
        for n in self.CT:
            self.assertTrue(torch.allclose(P[n] + O[n], self.CT[n], atol=1e-6))
        d = _flat(self.Ck)
        self.assertAlmostEqual(float(_flat(O) @ d), 0.0, places=5)
        self.assertAlmostEqual(share, float(_flat(O).pow(2).sum() / _flat(self.CT).pow(2).sum()), places=6)

    def test_planted_scale_is_recovered(self):
        import h8
        CT = {n: 2.5 * x for n, x in self.Ck.items()}
        P, O, a, share = h8.decompose(self.Ck, CT)
        self.assertAlmostEqual(a, 2.5 * float(_flat(self.Ck).norm()), places=4)
        self.assertAlmostEqual(share, 0.0, places=8)


class TestLibrary(unittest.TestCase):
    def test_retention(self):
        import h8
        self.assertAlmostEqual(h8.retention(0.5, 0.2, 0.6), 0.75)

    def test_add_coefs_sums_per_matrix_and_scales(self):
        import h8
        A = {"x": torch.ones(2, 2), "y": torch.zeros(3)}
        B = {"x": 2 * torch.ones(2, 2), "y": torch.ones(3)}
        S = h8.add_coefs([A, B], 0.5)
        self.assertTrue(torch.allclose(S["x"], 1.5 * torch.ones(2, 2)))
        self.assertTrue(torch.allclose(S["y"], 0.5 * torch.ones(3)))
        self.assertTrue(torch.equal(A["x"], torch.ones(2, 2)))

    def test_pick_lambda_tie_goes_to_smaller(self):
        import h8
        self.assertEqual(h8.pick_lambda({0.5: 3.0, 0.75: 3.0, 1: 2.0}), 0.5)
        self.assertEqual(h8.pick_lambda({0.5: 1.0, 0.75: 3.0, 1: 2.0}), 0.75)

    def test_ties_merge_trim_elect_disjoint_mean(self):
        import h8
        A = {"m": torch.tensor([4.0, -1.0, 2.0, 0.1])}
        B = {"m": torch.tensor([2.0, 3.0, -6.0, 0.2])}
        C = {"m": torch.tensor([-1.0, 1.0, -2.0, 0.3])}
        # no trim: coord0 sum +5 -> mean(4,2)=3; coord1 sum +3 -> mean(3,1)=2; coord2 sum -6 -> mean(-6,-2)=-4;
        # coord3 all positive -> 0.2
        M = h8.ties_merge([A, B, C], density=1.0)
        self.assertTrue(torch.allclose(M["m"], torch.tensor([3.0, 2.0, -4.0, 0.2])))
        # density 0.5 keeps top 2 of 4 per task: A {4,2}, B {-6,3}, C {-2,-1 or 1 tie}
        M = h8.ties_merge([A, B, C], density=0.5)
        self.assertAlmostEqual(float(M["m"][3]), 0.0)
        self.assertAlmostEqual(float(M["m"][0]), 4.0)
        self.assertAlmostEqual(float(M["m"][2]), -4.0)

    def test_ties_merge_is_global_over_matrices(self):
        import h8
        A = {"a": torch.tensor([10.0, 0.0]), "b": torch.tensor([0.1, 0.2])}
        M = h8.ties_merge([A], density=0.25)
        self.assertTrue(torch.allclose(M["a"], torch.tensor([10.0, 0.0])))
        self.assertTrue(torch.allclose(M["b"], torch.zeros(2)))

    def test_pick_grid_ties_to_smaller(self):
        import h8
        self.assertEqual(h8.pick_grid({(0.5, 2): 5.0, (0.2, 3): 5.0, (0.2, 1): 4.0}), (0.2, 3))
        self.assertEqual(h8.pick_grid({(0.2, 3): 5.0, (0.2, 2): 5.0}), (0.2, 2))

    def test_pairs_are_21(self):
        import h8
        self.assertEqual(len(h8.pairs(["t%d" % i for i in range(7)])), 21)


if __name__ == "__main__":
    unittest.main()
