"""Measures long-context synthesis at the exact 6L/384/1536 shape, against the table of the same size.

Repeats the gate's L2 check over several sampling seeds, because a small margin may depend on which
positions were drawn. Writes lab/runs/wsp_synth_long_verification.json.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

R = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R), str(R / "lab")]

import llmness as lm  # noqa: E402
from engine import wsp_synth, wsp_train  # noqa: E402

ARCH = R / "programs" / "arch" / "nanogpt.wsparch"
SHAPE = dict(block=256, layers=6, heads=6, d=384, ff=1536, dropout=0.0, bias=False)
L2_SEEDS = range(10)


class DeviceModel(lm.NanoGPTCharModel):
    def logits(self, x):
        device = next(self.module.parameters()).device
        return self.module(x.to(device), torch.zeros_like(x).to(device))[0].cpu()


def measure(name, model, data, text, device):
    assert all(p.grad is None for p in model.parameters()), "a gradient was computed"
    model = model.eval().to(device)
    adapter = DeviceModel(model, data.chars)
    val = lm.score_nll(adapter, text[int(len(text) * 0.9):], seq=model.config.block_size, batch=4)
    baseline_train, test_text = lm.split_corpus(text, 0.1, 8000)
    gate = lm.evaluate(adapter, baseline_train, test_text, ladder=())
    gate["GEN"].pop("greedy_sample", None)
    seeds = [lm.check_long_context(adapter, test_text, seed=s) for s in L2_SEEDS]
    passes = sum(r["verdict"] == "PASSES" for r in seeds)
    gains = [r["mean_gain_nats"] for r in seeds]
    print(f"{name}: val={val:.4f} verdict={gate['verdict']} L2 passes {passes}/{len(seeds)} seeds, "
          f"gain {min(gains):+.4f}..{max(gains):+.4f}, fits={model.synthesis_info.get('fits')}", flush=True)
    return {"val_nll_full": round(val, 4), "parameters": sum(p.numel() for p in model.parameters()),
            "synthesis": model.synthesis_info, "gate": gate, "l2_seeds_passed": passes,
            "l2_seeds_total": len(seeds), "l2_gains": gains}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(R / "lab" / "data" / "shakespeare_char" / "input.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(R / "lab" / "runs" / "wsp_synth_long_verification.json"))
    a = ap.parse_args()
    text = Path(a.corpus).read_text(encoding="utf-8")
    data = wsp_train.char_dataset(text)

    table = wsp_synth.WittenBellTable(data.train, len(data.chars), 4)
    runs = {"table_order4": measure("table_order4", wsp_synth.compile_ngram(ARCH, table, overrides=SHAPE, seed=1337),
                                    data, text, a.device)}
    for free in (1, 2):
        model = wsp_synth.compile_long_context(ARCH, data.train, data.chars, overrides=SHAPE, seed=1337, order=4,
                                               free_blocks=free, device=a.device)
        runs[f"long_free{free}"] = measure(f"long_free{free}", model, data, text, a.device)
    Path(a.out).write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
