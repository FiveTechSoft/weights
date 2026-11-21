import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lab" / "probes" / "cc_neurons"))
import decompose as dc  # noqa: E402


class Concentration(unittest.TestCase):
    def test_one_neuron_carries_everything(self):
        W = torch.eye(4, 8)
        da = torch.tensor([3.0, 0, 0, 0, 0, 0, 0, 0])
        r = dc.concentration(W, da, W @ da)
        self.assertAlmostEqual(r["share"], 1.0, places=6)
        self.assertEqual(r["k50"], 1)
        self.assertEqual(r["k90"], 1)
        self.assertAlmostEqual(r["top"][1], 1.0, places=6)

    def test_spread_evenly(self):
        W = torch.cat([torch.eye(4), torch.eye(4)], 1)        # 8 neurons, two per output axis
        da = torch.ones(8)
        r = dc.concentration(W, da, W @ da)
        self.assertEqual(r["k50"], 4)
        self.assertEqual(r["k90"], 8)
        self.assertAlmostEqual(r["top"][1], 1 / 8, places=6)

    def test_share_is_mlp_part_of_target(self):
        W = torch.eye(2, 3)
        da = torch.tensor([1.0, 0, 0])
        target = torch.tensor([1.0, 1.0])                      # half of |v|^2 is not the MLP
        r = dc.concentration(W, da, target)
        self.assertAlmostEqual(r["share"], 0.5, places=6)
        self.assertEqual(r["k50"], 1)

    def test_contributions_sum_to_share(self):
        g = torch.Generator().manual_seed(0)
        W, da, v = torch.randn(16, 40, generator=g), torch.randn(40, generator=g), torch.randn(16, generator=g)
        c = dc.contributions(W, da, v)
        self.assertAlmostEqual(float(c.sum()), float((W @ da) @ v / (v @ v)), places=4)


class Increments(unittest.TestCase):
    def test_increment_is_attention_plus_mlp(self):
        D = torch.tensor([[0.0, 0], [1, 1], [3, 2]])
        attn = {1: torch.tensor([0.5, 0.5])}
        mlp = {1: torch.tensor([1.5, 0.5])}
        err = dc.additivity_error(D, attn, mlp)
        self.assertAlmostEqual(err[1], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
