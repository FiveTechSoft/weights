"""Shared shape and measurement for the path 1 probes (see als_criterion.md)."""
import sys
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R / "lab")]

import llmness as lm  # noqa: E402

ARCH = R / "programs" / "arch" / "nanogpt.wsparch"
SHAPE = dict(block=64, layers=2, heads=4, d=64, ff=256, dropout=0.0, bias=False)
L2_SEEDS = range(5)


class DeviceModel(lm.NanoGPTCharModel):
    def logits(self, x):
        device = next(self.module.parameters()).device
        return self.module(x.to(device), torch.zeros_like(x).to(device))[0].float().cpu()


def measure(name, model, data, text, device):
    model = model.eval().to(device)
    adapter = DeviceModel(model, data.chars)
    val = lm.score_nll(adapter, text[int(len(text) * 0.9):], seq=model.config.block_size, batch=64)
    baseline_train, test_text = lm.split_corpus(text, 0.1, 8000)
    gate = lm.evaluate(adapter, baseline_train, test_text, ladder=())
    gate["GEN"].pop("greedy_sample", None)
    seeds = [lm.check_long_context(adapter, test_text, seed=s) for s in L2_SEEDS]
    passes = sum(r["verdict"] == "PASSES" for r in seeds)
    gains = [r["mean_gain_nats"] for r in seeds]
    print(f"{name}: val={val:.4f} verdict={gate['verdict']} L2 {passes}/{len(seeds)} "
          f"gain {min(gains):+.4f}..{max(gains):+.4f}", flush=True)
    return {"val_nll_full": round(val, 4), "parameters": sum(p.numel() for p in model.parameters()),
            "synthesis": getattr(model, "synthesis_info", None), "gate": gate,
            "l2_seeds_passed": passes, "l2_seeds_total": len(seeds), "l2_gains": gains}
