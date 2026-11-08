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

    def test_pairs_are_21(self):
        import h8
        self.assertEqual(len(h8.pairs(["t%d" % i for i in range(7)])), 21)


if __name__ == "__main__":
    unittest.main()
