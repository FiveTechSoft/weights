"""lab/probes/close_probe/scale_delta.py: base + alpha * (source - base), tensor by tensor."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))


class TestMix(unittest.TestCase):
    def test_alpha_endpoints_and_half(self):
        import scale_delta as sd
        b = torch.tensor([1.0, 2.0]).bfloat16()
        s = torch.tensor([3.0, 6.0]).bfloat16()
        self.assertTrue(torch.equal(sd.mix(b, s, 0.0), b))
        self.assertTrue(torch.equal(sd.mix(b, s, 1.0), s))
        self.assertTrue(torch.equal(sd.mix(b, s, 0.5), torch.tensor([2.0, 4.0]).bfloat16()))
        self.assertEqual(sd.mix(b, s, 0.5).dtype, torch.bfloat16)

    def test_missing_source_keeps_base(self):
        import scale_delta as sd
        b = torch.ones(3)
        self.assertTrue(torch.equal(sd.mix(b, None, 0.5), b))


if __name__ == "__main__":
    unittest.main()
