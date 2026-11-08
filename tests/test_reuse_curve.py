import importlib.util
import pathlib
import unittest

import torch

_p = pathlib.Path(__file__).resolve().parents[1] / "lab/probes/delta_patterns/reuse_curve.py"
_s = importlib.util.spec_from_file_location("reuse_curve", _p)
rc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rc)


class ReuseCurve(unittest.TestCase):
    def test_rank_two_delta_is_captured_by_rank_two(self):
        g = torch.Generator().manual_seed(0)
        d = torch.randn(40, 2, generator=g) @ torch.randn(2, 30, generator=g)
        c = rc.curve(d)
        self.assertLessEqual(c["rank_for_90pct"], 2)
        self.assertAlmostEqual(c["energy_rank_16"], 1.0, places=4)

    def test_diffuse_delta_needs_many_ranks(self):
        d = torch.randn(64, 64, generator=torch.Generator().manual_seed(1))
        c = rc.curve(d)
        self.assertGreater(c["rank_for_90pct"], 30)
        self.assertLess(c["energy_rank_16"], 0.8)  # a square Gaussian keeps about 0.62 in 16 of 64 ranks

    def test_orientation_does_not_matter(self):
        d = torch.randn(20, 50, generator=torch.Generator().manual_seed(2))
        a, b = rc.curve(d), rc.curve(d.T.contiguous())
        self.assertEqual(a["rank_for_80pct"], b["rank_for_80pct"])


if __name__ == "__main__":
    unittest.main()
