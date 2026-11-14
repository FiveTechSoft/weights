"""Shared harness for the loop-fix experiment (small replica of "Sparks of Weight-Space Programming", exp. 8).

Prompts are 128-character windows of Shakespeare val. Dev prompts come from the first half of val, test prompts
from the second half; the editor only ever sees dev. A continuation "loops" when its last 200 characters are
exactly periodic with a period of 1-100 characters.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(R)]

from engine import wsp_arch, wsp_train  # noqa: E402

ARCH = R / "programs" / "arch" / "nanogpt.wsparch"
BASE = R / "models/nanogpt_shakespeare_char/model.safetensors"
SHAPE = dict(block=256, layers=6, heads=6, d=384, ff=1536, dropout=0.0, bias=False)
PROMPT_LEN, NEW_CHARS, TAIL, MAX_PERIOD = 128, 512, 200, 100
RUNS = R / "lab/runs/loopfix"
DEV_LOG = RUNS / "dev_calls.jsonl"
DEV_BUDGET = 50

TEXT = (R / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
DATA = wsp_train.char_dataset(TEXT)
STOI = {c: i for i, c in enumerate(DATA.chars)}
VAL = TEXT[int(len(TEXT) * 0.9):]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prompts(split, n, seed):
    import os
    if split != "dev" and os.environ.get("LOOPFIX_FINAL") != "1":
        raise PermissionError("test prompts are reserved for final_test.py")
    half = len(VAL) // 2
    lo, hi = (0, half - PROMPT_LEN) if split == "dev" else (half, len(VAL) - PROMPT_LEN)
    rng = random.Random(seed)
    return [VAL[s:s + PROMPT_LEN] for s in (rng.randrange(lo, hi) for _ in range(n))]


def load(path=BASE):
    model = wsp_arch.load_arch_file(ARCH, overrides=dict(SHAPE, vocab=len(DATA.chars)), seed=1337)
    model.load(Path(path))
    return model.eval().to(DEVICE)


def logits(model, x):
    return model(x, torch.zeros_like(x))[0].float()


@torch.no_grad()
def generate(model, texts, new=NEW_CHARS, temperature=0.0, seed=0, decode_hook=None):
    """Greedy when temperature == 0. decode_hook(logits[B,V], generated ids[B,t]) -> logits, for decoding controls."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.tensor([[STOI[c] for c in t] for t in texts], device=DEVICE)
    start = x.shape[1]
    for _ in range(new):
        out = logits(model, x[:, -256:])[:, -1]
        if decode_hook is not None:
            out = decode_hook(out, x[:, start:])
        if temperature == 0:
            nxt = out.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(torch.softmax(out / temperature, -1), 1, generator=gen)
        x = torch.cat([x, nxt], 1)
    return ["".join(DATA.chars[i] for i in row[start:].tolist()) for row in x]


def loops(text):
    tail = text[-TAIL:]
    return any(all(tail[i] == tail[i + p] for i in range(len(tail) - p)) for p in range(1, MAX_PERIOD + 1))


def loop_rate(texts):
    return sum(loops(t) for t in texts) / len(texts)


def distinct(texts, n=4):
    grams = [t[i:i + n] for t in texts for i in range(len(t) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


@torch.no_grad()
def val_nll(model, block=256, stride=128, batch=32):
    """Every position scored with >= 128 characters of context."""
    ids = torch.tensor(DATA.val, dtype=torch.long)
    n = len(ids)
    starts = list(range(0, n - block - 1, stride)) + [n - block - 1]
    total = torch.zeros(n)
    for b in range(0, len(starts), batch):
        chunk = starts[b:b + batch]
        x = torch.stack([ids[s:s + block] for s in chunk]).to(DEVICE)
        y = torch.stack([ids[s + 1:s + block + 1] for s in chunk]).to(DEVICE)
        nll = -torch.log_softmax(logits(model, x), -1).gather(-1, y[..., None])[..., 0].cpu()
        for row, s in enumerate(chunk):
            lo = 0 if s == 0 else stride - 1
            total[s + 1 + lo:s + 1 + block] = nll[row, lo:]
    return float(total[1:].mean())


def evaluate(model, split, n, seed, decode_hook=None):
    texts = generate(model, prompts(split, n, seed), decode_hook=decode_hook)
    return {"loop_rate": loop_rate(texts), "distinct4": round(distinct(texts), 4), "val_nll": round(val_nll(model), 5),
            "examples": [t[-120:] for t in texts[:3]]}


def _spend(kind, path, note, extra):
    RUNS.mkdir(parents=True, exist_ok=True)
    used = sum(1 for _ in DEV_LOG.open(encoding="utf-8")) if DEV_LOG.exists() else 0
    if used >= DEV_BUDGET:
        raise RuntimeError(f"budget of {DEV_BUDGET} behaviour calls is spent")
    with DEV_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"call": used + 1, "kind": kind, "time": time.strftime("%H:%M:%S"), "model": str(path),
                            "note": note, **extra}) + "\n")
    return used + 1


def dev_eval(path, note=""):
    """Behaviour on 64 dev prompts (greedy loop rate, distinct-4, val NLL). Counts against the budget."""
    result = evaluate(load(path), "dev", 64, seed=1)
    result["call"] = _spend("dev_eval", path, note, {k: v for k, v in result.items() if k != "examples"})
    return result


@torch.no_grad()
def forward_probe(path, texts, out_npz, scopes=None, note=""):
    """One forward pass on up to 64 texts of <= 256 characters (train text or dev prompts, never test).

    Saves to out_npz: ids [B,T], logits [B,T,V], and for each linear layer in `scopes` the value entering it
    (float16), stored under the scope name. Useful scopes: transformer.h.{k}.attn.c_attn (= ln_1 output),
    transformer.h.{k}.mlp.c_fc (= ln_2 output), transformer.h.{k}.mlp.c_proj (= gelu hidden, 1536 wide),
    lm_head (= ln_f output). Default: every mlp.c_fc and lm_head. At most 60M stored values. Texts shorter than
    the longest are left-padded with newlines. Counts against the budget.
    """
    import numpy as np
    if len(texts) > 64 or any(len(t) > 256 for t in texts):
        raise ValueError("at most 64 texts of at most 256 characters")
    scopes = scopes or [f"transformer.h.{k}.mlp.c_fc" for k in range(SHAPE["layers"])] + ["lm_head"]
    model = load(path)
    width = max(len(t) for t in texts)
    texts = [t.rjust(width, "\n") for t in texts]
    x = torch.tensor([[STOI[c] for c in t] for t in texts], device=DEVICE)
    arrays = {"ids": x.cpu().numpy(), "logits": logits(model, x).half().cpu().numpy()}
    for scope in scopes:
        arrays[scope] = model.input_of(scope, x).half().cpu().numpy()
    if sum(a.size for a in arrays.values()) > 60_000_000:
        raise ValueError("probe too large: use fewer texts or scopes")
    np.savez_compressed(out_npz, **arrays)
    _spend("forward_probe", path, note, {"texts": len(texts), "scopes": scopes, "out": str(out_npz)})
    return out_npz


if __name__ == "__main__":
    print(json.dumps(dev_eval(sys.argv[1] if len(sys.argv) > 1 else BASE, " ".join(sys.argv[2:])), indent=2))
