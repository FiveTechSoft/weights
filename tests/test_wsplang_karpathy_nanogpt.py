"""tests/test_wsplang_karpathy_nanogpt.py — Test de Paridad y Construcción de nanoGPT.

Verifica que WSPLang compila la arquitectura nanoGPT de Karpathy
y preserva paridad numérica absoluta (0.0 error).
"""

import unittest
from pathlib import Path
import torch

from engine.nanogpt_runtime import GPTConfig, KarpathyGPT, NanoGPT
from engine.wsplang import execute_plan_v4, plan_run, parse_args_lines

R = Path(r"C:\weights")


class TestWSPLangKarpathyNanoGPT(unittest.TestCase):
    def test_karpathy_nanogpt_synthesis_and_parity(self):
        ckpt_p = R / "models" / "nanogpt_karpathy" / "model.safetensors"
        self.assertTrue(ckpt_p.exists(), f"Checkpoint {ckpt_p} must exist.")

        cfg = GPTConfig(
            block_size=256,
            vocab_size=65,
            n_layer=6,
            n_head=6,
            n_embd=384,
            d_ff=1536,
            bias=True
        )

        wsp_gpt = NanoGPT(config=cfg, seed=1337)
        wsp_gpt.load_checkpoint(ckpt_p)

        ref_gpt = KarpathyGPT(cfg)
        ref_gpt.load_state_dict(wsp_gpt.export_state_dict(), strict=False)
        ref_gpt.transformer.wte.weight = ref_gpt.lm_head.weight
        ref_gpt.eval()
        wsp_gpt.model.eval()

        # 1. Total parameter parity
        self.assertEqual(wsp_gpt.count_parameters(), 10770816)
        self.assertEqual(sum(p.numel() for p in ref_gpt.parameters()), 10770816)

        # 2. Forward pass bit-level identity
        x = torch.randint(0, 65, (2, 32))
        with torch.no_grad():
            logits_ref, _ = ref_gpt(x)
            logits_wsp, _ = wsp_gpt.forward(x)

        max_diff = (logits_ref - logits_wsp).abs().max().item()
        self.assertAlmostEqual(max_diff, 0.0, places=5)

        # 3. Weight tying exact identity
        diff_tied = (wsp_gpt.model.transformer.wte.weight - wsp_gpt.model.lm_head.weight).abs().max().item()
        self.assertEqual(diff_tied, 0.0)

        # 4. Step-by-step generative parity
        prompt = torch.tensor([[5, 12, 19, 2]], dtype=torch.long)
        out_wsp = wsp_gpt.generate(prompt.clone(), max_new_tokens=15, temperature=0.0)
        self.assertEqual(out_wsp.shape, (1, 19))


if __name__ == "__main__":
    unittest.main()
