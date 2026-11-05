"""Independent check of a checkpoint from programs/train_nanogpt_shakespeare_char.wspl.

Loads it strictly into upstream's model.py and measures validation NLL over every fixed window, as a
second method next to the random-batch estimate that selected the checkpoint during training.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

R = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R), str(R / "lab"), str(R / "lab" / "reference" / "karpathy_nanogpt")]

import llmness as lm  # noqa: E402
from engine import wsp_train  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

UPSTREAM_BEST_VAL = 1.4697  # README of karpathy/nanoGPT, same configuration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(R / "models" / "nanogpt_shakespeare_char" / "model.safetensors"))
    ap.add_argument("--corpus", default=str(R / "lab" / "data" / "shakespeare_char" / "input.txt"))
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--out", default=str(R / "lab" / "runs" / "wsp_train_verification.json"))
    a = ap.parse_args()

    sd = load_file(a.ckpt)
    with safe_open(a.ckpt, framework="pt") as f:
        meta = f.metadata()
    vocab, d = sd["transformer.wte.weight"].shape
    layers = len({k.split(".")[2] for k in sd if k.startswith("transformer.h.")})
    cfg = GPTConfig(vocab_size=vocab, n_embd=d, n_head=a.n_head, n_layer=layers, dropout=0.0,
                    block_size=sd["transformer.wpe.weight"].shape[0], bias="transformer.h.0.mlp.c_fc.bias" in sd)
    official = GPT(cfg)
    official.load_state_dict(sd, strict=True)

    text = Path(a.corpus).read_text(encoding="utf-8")
    data = wsp_train.char_dataset(text)
    if json.loads(meta["vocab_chars"]) != data.chars:
        raise ValueError("checkpoint vocabulary differs from corpus")
    val_text = text[int(len(text) * 0.9):]
    model = lm.NanoGPTCharModel(official, data.chars)
    full_val = lm.score_nll(model, val_text, seq=cfg.block_size, batch=8)

    torch.manual_seed(1337)
    prompt = torch.tensor([model.encode("ROMEO:\n")])
    with torch.no_grad():
        sample = model.decode(official.generate(prompt, 400, temperature=0.8, top_k=200)[0].tolist())

    estimate = float(meta["val_loss"])
    result = {"checkpoint": Path(a.ckpt).name, "provenance": meta["provenance"],
              "backward_calls": int(meta["backward_calls"]), "iter_num": int(meta["iter_num"]),
              "val_estimate_random_batches": round(estimate, 4), "val_full_fixed_windows": round(full_val, 4),
              "methods_differ_by": round(abs(estimate - full_val), 4),
              "upstream_reported_best_val": UPSTREAM_BEST_VAL,
              "delta_vs_upstream": round(estimate - UPSTREAM_BEST_VAL, 4), "sample": sample}
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
