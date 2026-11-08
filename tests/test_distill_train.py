"""lab/probes/close_probe/distill_train.py: LoRA training of a frozen model streamed layer by layer.

Checked on the real Qwen3.5-0.8B (CPU): the streamed + checkpointed + chunked-loss gradient equals the plain one, the
LoRA starts as the identity edit, and the export is the PEFT layout delta_to_lora writes."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lab" / "probes" / "close_probe"))
M08 = ROOT / "hf_cache" / "qwen35-08b-hf"


@unittest.skipUnless(M08.exists(), "needs hf_cache/qwen35-08b-hf")
class TestDistillTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForCausalLM
        cls.model = AutoModelForCausalLM.from_pretrained(str(M08), dtype=torch.float32)
        g = torch.Generator().manual_seed(0)
        cls.seqs = [(torch.randint(0, 5000, (7,), generator=g).tolist(), torch.randint(0, 5000, (9,), generator=g).tolist()),
                    (torch.randint(0, 5000, (5,), generator=g).tolist(), torch.randint(0, 5000, (4,), generator=g).tolist())]

    def setUp(self):
        import distill_train as dt
        self.dt = dt
        self.loras = dt.attach(self.model, rank=4, kinds=("down_proj",), layers=[0, 3], device="cpu", seed=1)

    def tearDown(self):
        self.dt.detach(self.model)

    def test_lora_starts_as_identity(self):
        ids = torch.tensor([self.seqs[0][0] + self.seqs[0][1]])
        with torch.no_grad():
            a = self.model(input_ids=ids).logits
            self.dt.detach(self.model)
            b = self.model(input_ids=ids).logits
        self.loras = self.dt.attach(self.model, rank=4, kinds=("down_proj",), layers=[0, 3], device="cpu", seed=1)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_streamed_chunked_gradient_equals_plain(self):
        for l in self.loras.values():                 # B != 0 so every LoRA parameter gets a gradient
            torch.nn.init.normal_(l.B, std=0.01)
        params = [p for l in self.loras.values() for p in (l.A, l.B)]
        # plain: full logits, mean CE over completion tokens of both sequences
        tot, n = 0.0, 0
        for p_ids, c_ids in self.seqs:
            ids = torch.tensor([p_ids + c_ids])
            logits = self.model(input_ids=ids).logits[0, :-1]
            tgt = ids[0, 1:]
            m = torch.zeros(len(tgt), dtype=torch.bool)
            m[len(p_ids) - 1:] = True
            tot = tot + torch.nn.functional.cross_entropy(logits[m], tgt[m], reduction="sum")
            n += int(m.sum())
        g_plain = torch.autograd.grad(tot / n, params)
        for p in params:
            p.grad = None
        loss = self.dt.batch_backward(self.model, self.seqs, device="cpu", chunk=3)
        self.assertAlmostEqual(loss, float(tot / n), places=4)
        for gp, p in zip(g_plain, params):
            self.assertTrue(torch.allclose(gp, p.grad, atol=1e-5, rtol=1e-3), float((gp - p.grad).abs().max()))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a free GPU")
    def test_streaming_through_the_gpu_gives_the_cpu_gradient(self):
        for l in self.loras.values():
            torch.nn.init.normal_(l.B, std=0.01)
        params = [p for l in self.loras.values() for p in (l.A, l.B)]
        cpu_loss = self.dt.batch_backward(self.model, self.seqs, device="cpu", chunk=3)
        g_cpu = [p.grad.clone() for p in params]
        state = {k: (l.A.detach().clone(), l.B.detach().clone()) for k, l in self.loras.items()}
        self.dt.detach(self.model)
        gpu = self.dt.attach(self.model, rank=4, kinds=("down_proj",), layers=[0, 3], device="cuda", seed=1)
        with torch.no_grad():
            for k, l in gpu.items():
                l.A.copy_(state[k][0])
                l.B.copy_(state[k][1])
        loss = self.dt.batch_backward(self.model, self.seqs, device="cuda", chunk=3)
        self.assertAlmostEqual(loss, cpu_loss, places=3)
        g_gpu = [p.grad.cpu() for l in gpu.values() for p in (l.A, l.B)]
        for a, b in zip(g_cpu, g_gpu):
            self.assertTrue(torch.allclose(a, b, atol=1e-4, rtol=1e-2), float((a - b).abs().max()))
        self.assertTrue(all(p.device.type == "cpu" for n, p in self.model.named_parameters() if ".A" not in n
                            and ".B" not in n))
        self.loras = gpu

    def test_export_is_peft_layout(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "lora"
            self.dt.export(self.loras, out, rank=4)
            from safetensors.torch import load_file
            T = load_file(str(out / "adapter_model.safetensors"))
            k = "base_model.model.model.language_model.layers.3.mlp.down_proj.lora_A.weight"
            self.assertIn(k, T)
            self.assertEqual(len(T), 4)
            cfg = json.loads((out / "adapter_config.json").read_text())
            self.assertEqual((cfg["r"], cfg["lora_alpha"], cfg["target_modules"]), (4, 4, ["down_proj"]))


if __name__ == "__main__":
    unittest.main()
