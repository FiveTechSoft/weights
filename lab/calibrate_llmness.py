"""Calibrates the LLM-ness gate against controls with known answers; exits 1 on any surprise.

Negative: W0 (official nanoGPT at init) and count-based n-grams.
Positive: official nanoGPT trained by gradient descent; CopyOracle for L3 only.
The trained control uses lab/reference/karpathy_nanogpt/model.py, not the runtime under test.

"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R / "lab"), str(R / "lab" / "reference" / "karpathy_nanogpt")]

import llmness as lm  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

CFG = dict(block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=True)
EXPECTED = {"w0": "FAILS", "1-gram": "FAILS", "2-gram": "FAILS", "3-gram": "FAILS",
            "4-gram": "FAILS", "5-gram": "FAILS", "trained_nanogpt": "LANGUAGE_MODEL"}


def train_control(text, chars, steps, seed):
    torch.manual_seed(seed)
    return train(GPT(GPTConfig(vocab_size=len(set(chars)), **CFG)), text, chars, steps)


def train(model, text, chars, steps):
    """Continues from the current RNG state, so a same-seed model gets the same batches."""
    ids = np.asarray(lm.CharCodec(chars).encode(text), dtype=np.int64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    t0 = time.time()
    for s in range(steps):
        ix = torch.randint(len(ids) - CFG["block_size"] - 1, (12,))
        x = torch.stack([torch.from_numpy(ids[i:i + CFG["block_size"]]) for i in ix])
        y = torch.stack([torch.from_numpy(ids[i + 1:i + 1 + CFG["block_size"]]) for i in ix])
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        opt.step()
        if s % 500 == 0:
            print(f"  train step {s} loss {loss.item():.3f} t={time.time() - t0:.0f}s", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(R / "lab" / "data" / "shakespeare_char" / "input.txt"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-chars", type=int, default=8000)
    ap.add_argument("--out", default=str(R / "lab" / "runs" / "llmness_calibration.json"))
    a = ap.parse_args()

    text = Path(a.corpus).read_text(encoding="utf-8")
    chars = sorted(set(text))
    split = int(len(text) * 0.9)  # same split as nanoGPT prepare.py: the control never sees the tail
    baseline_train, test_text = lm.split_corpus(text, 0.1, a.eval_chars)

    results = {}
    torch.manual_seed(a.seed)
    w0 = lm.NanoGPTCharModel(GPT(GPTConfig(vocab_size=len(chars), **CFG)), chars)
    results["w0"] = lm.evaluate(w0, baseline_train, test_text)
    for order in (1, 2, 3, 4, 5):
        ngram = lm.NGramModel(baseline_train, order, vocab_chars=chars)
        ngram.max_seq = CFG["block_size"]
        results[f"{order}-gram"] = lm.evaluate(ngram, baseline_train, test_text, ladder=())
    trained = lm.NanoGPTCharModel(train_control(text[:split], chars, a.steps, a.seed), chars)
    results["trained_nanogpt"] = lm.evaluate(trained, baseline_train, test_text)
    oracle = lm.check_icl(lm.CopyOracle())

    surprises = [f"{k}: expected {v}, got {results[k]['verdict']}" for k, v in EXPECTED.items()
                 if results[k]["verdict"] != v]
    if oracle["verdict"] != "PASSES":
        surprises.append(f"copy_oracle L3: {oracle['verdict']}")

    for k, r in results.items():
        print(f"{k:16s} {r['verdict']:15s} nll={r['L1']['nll']:.4f} L1={r['L1']['verdict']:6s} "
              f"gain={r['L2_long_context']['mean_gain_nats']:+.4f} [{r['L2_long_context']['ci95_low']:+.4f}] "
              f"L2={r['L2_long_context']['verdict']:6s} words={r['GEN']['known_word_rate']:.2f} "
              f"distinct={r['GEN']['distinct_3grams_ratio']:.2f} GEN={r['GEN']['verdict']:6s} "
              f"icl_delta={r['L3_icl']['delta']:+.3f} L3={r['L3_icl']['verdict']}")
    print("copy_oracle L3", oracle)

    report = {"config": {**CFG, "steps": a.steps, "seed": a.seed, "corpus": Path(a.corpus).name},
              "expected": EXPECTED, "results": results, "copy_oracle_l3": oracle,
              "calibrated": not surprises, "surprises": surprises}
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("CALIBRATED" if not surprises else f"NOT CALIBRATED: {surprises}")
    sys.exit(0 if not surprises else 1)


if __name__ == "__main__":
    main()
