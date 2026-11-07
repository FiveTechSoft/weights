"""Measures synthesized checkpoints: full validation NLL, table agreement and the LLM-ness gate.

Compiles without gradients inside this process, then evaluates on the validation split, which neither
the table nor the order selection touched. Order is selected on the tail of the training split.
Writes lab/runs/wsp_synth_verification.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R), str(R / "lab"), str(R / "lab" / "reference" / "karpathy_nanogpt")]

import llmness as lm  # noqa: E402
from engine import wsp_synth, wsp_train  # noqa: E402

ARCH = R / "programs" / "arch" / "nanogpt.wsparch"
LISTON = 1.7463  # best train-only add-k n-gram on full validation
GRADIENT_REFERENCE = 1.4573


class DeviceModel(lm.NanoGPTCharModel):
    def logits(self, x):
        device = next(self.module.parameters()).device
        return self.module(x.to(device), torch.zeros_like(x).to(device))[0].cpu()


def table_nll(table, ids):
    order = table.order
    total = 0.0
    for t in range(order - 1, len(ids)):
        total -= table.log_prob(tuple(int(v) for v in ids[t - order + 1:t]))[ids[t]]
    return total / (len(ids) - order + 1)


def evaluate(name, overrides, table, data, text, device):
    t0 = time.time()
    model = wsp_synth.compile_ngram(ARCH, table, overrides=overrides, seed=1337).eval().to(device)
    compile_seconds = time.time() - t0
    adapter = DeviceModel(model, data.chars)
    cut = int(len(text) * 0.9)
    val = lm.score_nll(adapter, text[cut:], seq=model.config.block_size, batch=4)
    report = lm.evaluate(adapter, *lm.split_corpus(text, 0.1, 8000))
    report["GEN"].pop("greedy_sample", None)
    print(f"{name}: val={val:.4f} verdict={report['verdict']} L1={report['L1']['verdict']} "
          f"L2={report['L2_long_context']['mean_gain_nats']:+.4f} GEN={report['GEN']['verdict']} "
          f"words={report['GEN']['known_word_rate']} units={model.synthesis_info['units_used']}", flush=True)
    return {"val_nll_full": round(val, 4), "compile_seconds": round(compile_seconds, 1),
            "parameters": sum(p.numel() for p in model.parameters()), "synthesis": model.synthesis_info,
            "all_grads_none": all(p.grad is None for p in model.parameters()), "gate": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(R / "lab" / "data" / "shakespeare_char" / "input.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(R / "lab" / "runs" / "wsp_synth_verification.json"))
    a = ap.parse_args()
    text = Path(a.corpus).read_text(encoding="utf-8")
    data = wsp_train.char_dataset(text)
    vocab = len(data.chars)

    # Order is chosen on the training tail; validation is only read after this choice.
    inner = int(len(data.train) * 0.9)
    selection = {}
    for order in (2, 3, 4, 5):
        selection[order] = round(table_nll(wsp_synth.WittenBellTable(data.train[:inner], vocab, order),
                                           data.train[inner:]), 4)
        print(f"inner train selection order {order}: {selection[order]}", flush=True)
    best_order = min(selection, key=selection.get)

    table_vals = {}
    for order in (2, 3, 4, 5):
        table_vals[order] = round(table_nll(wsp_synth.WittenBellTable(data.train, vocab, order), data.val), 4)
    print("table val nll", table_vals, flush=True)

    full = wsp_synth.WittenBellTable(data.train, vocab, best_order)
    wide = dict(block=256, layers=1, heads=6, d=384, dropout=0.0, bias=False)
    runs = {f"wide_order{best_order}": evaluate(f"wide_order{best_order}", wide, full, data, text, a.device)}
    upstream_shape = dict(block=256, layers=6, heads=6, d=384, dropout=0.0, bias=False, ff=1536)
    runs["upstream_shape_6L_384"] = evaluate("upstream_shape_6L_384", upstream_shape, full, data, text, a.device)

    result = {"selected_order": best_order, "selection_inner_train": selection, "table_val_nll": table_vals,
              "liston": LISTON, "gradient_reference": GRADIENT_REFERENCE, "runs": runs}
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
