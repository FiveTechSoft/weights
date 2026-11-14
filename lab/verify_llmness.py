"""CLI for the LLM-ness gate (lab/llmness.py) on a trace or a nanoGPT checkpoint.

Examples:
  python lab/verify_llmness.py --trace lab/runs/nanowsp4_v1b145_trace.npz --corpus lab/data/corpus_n4b.txt
  python lab/verify_llmness.py --nanogpt-ckpt out.safetensors --n-head 6 --corpus shakespeare.txt

The last --eval-chars of the corpus are scored. Baselines fit before the last --holdout-frac, which must
match the model's own holdout: 0.1 for nanoGPT, 0.09 for nanowsp3, 0.05 for nanowsp4.
"""
import argparse
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R), str(R / "lab")]

import llmness as lm  # noqa: E402


def load_corpus(path, ascii_only):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return text.encode("ascii", "ignore").decode() if ascii_only else text


def load_nanogpt(path, n_head, chars):
    from safetensors.torch import load_file
    from engine.nanogpt_runtime import GPTConfig, NanoGPT
    sd = load_file(str(path))
    vocab, d = sd["transformer.wte.weight"].shape
    blocks = {k.split(".")[2] for k in sd if k.startswith("transformer.h.")}
    cfg = GPTConfig(vocab_size=vocab, n_embd=d, n_head=n_head, n_layer=len(blocks),
                    block_size=sd["transformer.wpe.weight"].shape[0],
                    d_ff=sd["transformer.h.0.mlp.c_fc.weight"].shape[0],
                    bias="transformer.h.0.mlp.c_fc.bias" in sd)
    if len(set(chars)) != vocab:
        raise ValueError(f"corpus has {len(set(chars))} symbols, checkpoint expects {vocab}")
    runtime = NanoGPT(cfg)
    runtime.load_checkpoint(path)
    return lm.NanoGPTCharModel(runtime.model, chars)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--trace", default=None)
    src.add_argument("--nanogpt-ckpt", default=None)
    ap.add_argument("--n-head", type=int, help="required with --nanogpt-ckpt; not recoverable from shapes")
    ap.add_argument("--snap-idx", type=int, default=-1)
    ap.add_argument("--corpus", default=str(R / "lab" / "data" / "corpus.txt"))
    ap.add_argument("--eval-chars", type=int, default=8000)
    ap.add_argument("--holdout-frac", type=float, default=0.1, help="tail excluded from model and baselines")
    ap.add_argument("--out", default=str(R / "rig_runs" / "llmness_verify.json"))
    a = ap.parse_args()

    if a.nanogpt_ckpt:
        if not a.n_head:
            ap.error("--nanogpt-ckpt requires --n-head")
        corpus = load_corpus(a.corpus, ascii_only=False)
        model, patient = load_nanogpt(a.nanogpt_ckpt, a.n_head, corpus), a.nanogpt_ckpt
    else:
        corpus = load_corpus(a.corpus, ascii_only=True)
        patient = a.trace or str(R / "lab" / "runs" / "nanowsp3_trace.npz")
        model = lm.NanoTraceModel(patient, a.snap_idx)

    report = {"patient": str(patient), "corpus": str(a.corpus), "eval_chars": a.eval_chars}
    report.update(lm.evaluate(model, *lm.split_corpus(corpus, a.holdout_frac, a.eval_chars)))
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
