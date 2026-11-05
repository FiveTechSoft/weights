"""tests/test_wsplang_karpathy_nanogpt.py — Test de Paridad Independiente con nanoGPT Oficial.

Comprueba matemáticamente que WSPLang compila la arquitectura nanoGPT
y reproduce de forma independiente el código upstream de Andrej Karpathy (commit 3adf61e)
con paridad numérica absoluta (0.0 error).
"""

import sys
import unittest
from pathlib import Path
import torch
from safetensors.torch import load_file

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "lab" / "reference" / "karpathy_nanogpt"))

from model import GPT as OfficialKarpathyGPT, GPTConfig as OfficialKarpathyConfig
from engine.nanogpt_runtime import GPTConfig as WSPConfig, NanoGPT


class TestWSPLangKarpathyNanoGPT(unittest.TestCase):
    def test_independent_karpathy_official_parity(self):
        ckpt_p = R / "models" / "nanogpt_karpathy" / "model.safetensors"
        self.assertTrue(ckpt_p.exists(), f"Checkpoint {ckpt_p} must exist.")

        # 1. Official Karpathy model instance
        cfg_off = OfficialKarpathyConfig(
            block_size=256,
            vocab_size=65,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.0,
            bias=True
        )
        official_gpt = OfficialKarpathyGPT(cfg_off)
        raw_sd = load_file(str(ckpt_p))
        param_sd = {k: v for k, v in raw_sd.items() if not k.endswith(".attn.bias")}
        official_gpt.load_state_dict(param_sd, strict=True)
        official_gpt.eval()

        # 2. WSPLang NanoGPT instance
        cfg_wsp = WSPConfig(
            block_size=256,
            vocab_size=65,
            n_layer=6,
            n_head=6,
            n_embd=384,
            d_ff=1536,
            dropout=0.0,
            bias=True
        )
        wsp_gpt = NanoGPT(config=cfg_wsp, seed=1337)
        wsp_gpt.load_checkpoint(ckpt_p)
        wsp_gpt.model.eval()

        # 3. Total parameter parity
        self.assertEqual(wsp_gpt.count_parameters(), 10770816)
        self.assertEqual(sum(p.numel() for p in official_gpt.parameters()), 10770816)

        # 4. Forward pass bit-level identity on multiple sequences
        for b, t in [(1, 8), (2, 32), (4, 64)]:
            x = torch.randint(0, 65, (b, t))
            targets = torch.randint(0, 65, (b, t))
            with torch.no_grad():
                logits_off, loss_off = official_gpt(x, targets=targets)
                logits_wsp, loss_wsp = wsp_gpt.forward(x, targets=targets)

            max_diff = (logits_off - logits_wsp).abs().max().item()
            loss_diff = abs(loss_off.item() - loss_wsp.item())
            self.assertEqual(max_diff, 0.0, f"Discrepancia en logits para B={b}, T={t}")
            self.assertEqual(loss_diff, 0.0, f"Discrepancia en pérdida para B={b}, T={t}")

        # 5. Weight tying exact identity in both
        diff_tied_wsp = (wsp_gpt.model.transformer.wte.weight - wsp_gpt.model.lm_head.weight).abs().max().item()
        diff_tied_off = (official_gpt.transformer.wte.weight - official_gpt.lm_head.weight).abs().max().item()
        self.assertEqual(diff_tied_wsp, 0.0)
        self.assertEqual(diff_tied_off, 0.0)

        # 6. Step-by-step generative parity
        prompt = torch.tensor([[5, 12, 19, 2]], dtype=torch.long)
        out_wsp = wsp_gpt.generate(prompt.clone(), max_new_tokens=15, temperature=0.0)

        curr_off = prompt.clone()
        for _ in range(15):
            with torch.no_grad():
                lg, _ = official_gpt(curr_off)
                nxt = lg[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_off = torch.cat((curr_off, nxt), dim=1)

        self.assertEqual(out_wsp.tolist(), curr_off.tolist(), "Generación greedy no coincide exactamente")


if __name__ == "__main__":
    unittest.main()
