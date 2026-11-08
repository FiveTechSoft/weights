import importlib.util
import pathlib
import unittest

import torch

_p = pathlib.Path(__file__).resolve().parents[1] / "lab/probes/delta_patterns/layer_probe.py"
_s = importlib.util.spec_from_file_location("layer_probe", _p)
lp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(lp)


class Metrics(unittest.TestCase):
    def test_reproduces_and_nothing(self):
        base, mimo = torch.zeros(10), torch.arange(10.0)
        self.assertAlmostEqual(lp.metrics(mimo, base, mimo)["captured"], 1.0)
        self.assertAlmostEqual(lp.metrics(mimo, base, mimo)["resid"], 0.0)
        self.assertAlmostEqual(lp.metrics(base, base, mimo)["captured"], 0.0)
        self.assertAlmostEqual(lp.metrics(base, base, mimo)["resid"], 1.0)

    def test_half_and_wrong_way(self):
        base, mimo = torch.zeros(10), torch.arange(10.0)
        self.assertAlmostEqual(lp.metrics(mimo / 2, base, mimo)["captured"], 0.5)
        self.assertLess(lp.metrics(-mimo, base, mimo)["captured"], 0)

    def test_orthogonal_change_captures_nothing(self):
        base = torch.zeros(4)
        shift = torch.tensor([1.0, 0, 0, 0])
        other = torch.tensor([0, 1.0, 0, 0])
        self.assertAlmostEqual(lp.metrics(other, base, shift)["captured"], 0.0)


if __name__ == "__main__":
    unittest.main()
