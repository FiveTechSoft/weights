"""lab/probes/close_probe/placebo_lora.py: a norm-matched random placebo of a PEFT LoRA adapter. Every adapted matrix
gets a change B'A' of the same rank and the same Frobenius norm as the real B A, in random directions; the config is
copied unchanged (same alpha / scaling). Used to rule out "any perturbation of this size makes the base close"."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))


def _adapter(d, g):
    from safetensors.torch import save_file
    T = {}
    for l, (o, i) in enumerate(((40, 24), (24, 40))):
        s = f"base_model.model.model.layers.{l}.mlp.down_proj"
        T[s + ".lora_A.weight"] = torch.randn(4, i, generator=g).to(torch.bfloat16)
        T[s + ".lora_B.weight"] = (torch.randn(o, 4, generator=g) * (l + 1)).to(torch.bfloat16)
    save_file(T, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 4, "target_modules": ["down_proj"]}))
    return T


class TestPlacebo(unittest.TestCase):
    def test_same_norm_and_rank_random_direction(self):
        import placebo_lora as pl
        from safetensors.torch import load_file
        g = torch.Generator().manual_seed(0)
        with tempfile.TemporaryDirectory() as t:
            src, dst = Path(t) / "a", Path(t) / "b"
            src.mkdir()
            T = _adapter(src, g)
            pl.make(src, dst, seed=1)
            P = load_file(str(dst / "adapter_model.safetensors"))
            self.assertEqual(set(P), set(T))
            self.assertEqual(json.loads((dst / "adapter_config.json").read_text()),
                             json.loads((src / "adapter_config.json").read_text()))
            for k in T:
                if k.endswith("lora_A.weight"):
                    kb = k.replace("lora_A", "lora_B")
                    self.assertEqual(P[k].shape, T[k].shape)
                    self.assertEqual(P[kb].dtype, T[kb].dtype)
                    real = T[kb].float() @ T[k].float()
                    fake = P[kb].float() @ P[k].float()
                    self.assertAlmostEqual(fake.norm().item() / real.norm().item(), 1.0, delta=0.01)
                    cos = (fake * real).sum() / (fake.norm() * real.norm())
                    self.assertLess(abs(cos.item()), 0.3)

    def test_refuses_to_overwrite(self):
        import placebo_lora as pl
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "a"
            src.mkdir()
            _adapter(src, torch.Generator().manual_seed(0))
            with self.assertRaises(SystemExit):
                pl.make(src, src, seed=1)


if __name__ == "__main__":
    unittest.main()
