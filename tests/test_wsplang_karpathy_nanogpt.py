"""Paridad del checkpoint que produce programs/build_karpathy_nanogpt.wspl frente a nanoGPT upstream.

El test ejecuta el programa de verdad contra un directorio temporal, sin depender de models/ (ignorado
por git), y compara con model.py de karpathy/nanoGPT (commit 3adf61e). El programa solo instancia la
clase fija de engine/nanogpt_runtime.py: esto valida W0 y runtime, no síntesis ni pesos entrenados.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "lab" / "reference" / "karpathy_nanogpt"))

from model import GPT as OfficialKarpathyGPT, GPTConfig as OfficialKarpathyConfig
from engine.nanogpt_runtime import GPTConfig as WSPConfig, NanoGPT
from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate

PROGRAM = R / "programs" / "build_karpathy_nanogpt.wspl"
PROGRAM_CKPT = "models/nanogpt_karpathy/model.safetensors"


class TestWSPLangKarpathyNanoGPT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = Path(cls._tmp.name) / "model.safetensors"
        code = PROGRAM.read_text(encoding="utf-8")
        assert PROGRAM_CKPT in code
        prog = parse_args_lines(code.replace(PROGRAM_CKPT, cls.ckpt.as_posix()))
        steps, errors = validate(prog)
        assert not errors, errors
        cls.results = execute_plan_v4(plan_run(prog, steps), prog, dry_run=False)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.cfg = dict(block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=True)

    def test_program_runs_every_step(self):
        self.assertEqual([r["status"] for r in self.results], ["ok"] * len(self.results))

    def test_artifact_is_upstream_w0_for_same_seed(self):
        torch.manual_seed(1337)
        official = OfficialKarpathyGPT(OfficialKarpathyConfig(**self.cfg))
        sd = load_file(str(self.ckpt))
        for name, p in official.named_parameters():
            self.assertTrue(torch.equal(sd[name], p.detach()), name)

    def test_forward_and_generation_parity(self):
        official_gpt = OfficialKarpathyGPT(OfficialKarpathyConfig(**self.cfg))
        official_gpt.load_state_dict(load_file(str(self.ckpt)), strict=True)
        official_gpt.eval()

        wsp_gpt = NanoGPT(config=WSPConfig(**self.cfg), seed=1337)
        wsp_gpt.load_checkpoint(self.ckpt)
        wsp_gpt.model.eval()

        expected_params = sum(p.numel() for p in official_gpt.parameters())
        self.assertEqual(wsp_gpt.count_parameters(), expected_params)

        for b, t in [(1, 8), (2, 32), (4, 64)]:
            x = torch.randint(0, 65, (b, t))
            targets = torch.randint(0, 65, (b, t))
            with torch.no_grad():
                logits_off, loss_off = official_gpt(x, targets=targets)
                logits_wsp, loss_wsp = wsp_gpt.forward(x, targets=targets)
            self.assertEqual((logits_off - logits_wsp).abs().max().item(), 0.0, f"logits B={b}, T={t}")
            self.assertEqual(abs(loss_off.item() - loss_wsp.item()), 0.0, f"loss B={b}, T={t}")

        self.assertIs(wsp_gpt.model.transformer.wte.weight, wsp_gpt.model.lm_head.weight)

        prompt = torch.tensor([[5, 12, 19, 2]], dtype=torch.long)
        out_wsp = wsp_gpt.generate(prompt.clone(), max_new_tokens=15, temperature=0.0)
        curr_off = prompt.clone()
        for _ in range(15):
            with torch.no_grad():
                lg, _ = official_gpt(curr_off)
            curr_off = torch.cat((curr_off, lg[:, -1, :].argmax(dim=-1, keepdim=True)), dim=1)
        self.assertEqual(out_wsp.tolist(), curr_off.tolist())


if __name__ == "__main__":
    unittest.main()
