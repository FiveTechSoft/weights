"""lab/probes/close_probe/delta_to_lora.py: a fine-tune's delta as a PEFT LoRA (truncated SVD per matrix)."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))


class TestLowrank(unittest.TestCase):
    def test_exact_on_low_rank_delta(self):
        import delta_to_lora as dl
        g = torch.Generator().manual_seed(0)
        D = torch.randn(40, 3, generator=g) @ torch.randn(3, 30, generator=g)
        A, B, kept = dl.lowrank(D, 8)
        self.assertEqual(tuple(A.shape), (8, 30))
        self.assertEqual(tuple(B.shape), (40, 8))
        self.assertTrue(torch.allclose(B @ A, D, atol=1e-4))
        self.assertGreater(kept, 0.9999)

    def test_peft_names(self):
        import delta_to_lora as dl
        a, b = dl.peft_names("model.language_model.layers.3.mlp.down_proj.weight")
        self.assertEqual(a, "base_model.model.model.language_model.layers.3.mlp.down_proj.lora_A.weight")
        self.assertEqual(b, "base_model.model.model.language_model.layers.3.mlp.down_proj.lora_B.weight")

    def test_wanted_filters_by_kind(self):
        import delta_to_lora as dl
        mlp = "model.language_model.layers.3.mlp.down_proj.weight"
        att = "model.language_model.layers.3.linear_attn.out_proj.weight"
        norm = "model.language_model.layers.3.input_layernorm.weight"
        self.assertTrue(dl.wanted(mlp, None) and dl.wanted(att, None))
        self.assertFalse(dl.wanted(norm, None))
        self.assertTrue(dl.wanted(mlp, {"mlp"}))
        self.assertFalse(dl.wanted(att, {"mlp"}))


if __name__ == "__main__":
    unittest.main()
