"""H6 synthetic continuation, arm E_k (h6_plan.md, stage 0): W_0 + alpha * C_k, the first k chunks extrapolated.

Zero further backward passes. alpha is chosen on dev (200 items) from {1, 2, 3, 5, 8, 10}; the chosen alpha is then
evaluated in full (test_id, ood_easy, ood_hard, retention NLL) together with the control M3: per matrix a Gaussian
direction with the same norm as alpha * C_k. Pass: recovery >= 0.7 averaged over seeds, and E_k beats M3 with exact
McNemar p < 0.01 in each seed.

Numerics as in training's evaluation (fp32 matrices, bf16 embedding, bf16 autocast). Method check: with the chunk-10
checkpoint and alpha = 1 the rebuilt model must reproduce the accuracy logged during training (`--check`).

usage: python lab/probes/h6/synth.py --seed 1 --k 1 [--check]
writes lab/runs/h6/synth/E<k>_seed<s>.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "lab" / "runs" / "h6"
KEEP = ROOT / "hf_cache" / "h6"
sys.path.insert(0, str(HERE))

ALPHAS = (1, 2, 3, 5, 8, 10)


# ---------- pieces with tests ----------

def extrapolate(base, ck, alpha):
    return {n: base[n] + alpha * (ck[n] - base[n]) for n in base}


def random_like(base, ck, alpha, seed):
    """Per matrix: base + Gaussian direction rescaled to |alpha * (ck - base)|."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for n in sorted(base):
        r = torch.randn(base[n].shape, generator=g)
        out[n] = base[n] + r * (alpha * (ck[n] - base[n]).norm() / r.norm())
    return out


def mcnemar(a, b):
    """Exact two-sided McNemar p on paired booleans (binomial on the discordant pairs)."""
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def recovery(acc, acc0, acc1):
    return (acc - acc0) / (acc1 - acc0)


def pick_alpha(dev_acc):
    best = max(dev_acc.values())
    return min(a for a, v in dev_acc.items() if v == best)


# ---------- run ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--check", action="store_true", help="k=10, alpha=1 only: must reproduce training's eval")
    a = ap.parse_args()
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    import eval_arith as ea

    out_dir = RUNS / "synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = ea.load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(str(ea.BASE), dtype=torch.float32).cuda()
    model.model.embed_tokens.to(torch.bfloat16)
    params = dict(model.named_parameters())
    ck = load_file(str(KEEP / f"seed{a.seed}" / f"chunk{a.k}.safetensors"))
    base = {n: params[n].detach().float().cpu().clone() for n in ck}
    m0 = json.loads((RUNS / f"seed{a.seed}" / "chunk0.json").read_text(encoding="utf-8"))["eval"]
    m1 = json.loads((RUNS / f"seed{a.seed}" / "chunk10.json").read_text(encoding="utf-8"))["eval"]
    t0 = time.time()

    @torch.no_grad()
    def set_weights(w):
        for n, t in w.items():
            params[n].copy_(t)

    def full():
        return ea.evaluate(model, tok)

    if a.check:
        assert a.k == 10
        set_weights(ck)
        ev = ea.eval_split(model, tok, ea.load_split("test_id"))
        same = [x["ok"] for x in ev["items"]] == m1["test_id"]["ok_mask"]
        rec = {"seed": a.seed, "check": "chunk10 rebuilt = trained", "acc": ev["acc"],
               "acc_logged": m1["test_id"]["acc"], "identical_items": same}
        (out_dir / f"check_seed{a.seed}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(rec, flush=True)
        return

    dev = ea.load_split("dev")
    dev_acc = {}
    for al in ALPHAS:
        set_weights(extrapolate(base, ck, al))
        dev_acc[al] = ea.eval_split(model, tok, dev)["acc"]
        print(f"alpha {al} dev {dev_acc[al]:.3f} {time.time() - t0:.0f}s", flush=True)
    al = pick_alpha(dev_acc)
    set_weights(extrapolate(base, ck, al))
    e = full()
    set_weights(random_like(base, ck, al, seed=1000 * a.seed + a.k))
    r = full()
    accs = lambda ev: {s: ev[s]["acc"] for s in ("test_id", "ood_easy", "ood_hard")} | {
        "retention_nll": ev["retention_nll"], "test_id_independent": ev["test_id"]["acc_independent"]}
    a0, a1 = m0["test_id"]["acc"], m1["test_id"]["acc"]
    rec = {"seed": a.seed, "k": a.k, "dev_acc": dev_acc, "alpha": al,
           "M0": accs(m0), "M1": accs(m1), "E": accs(e), "M3": accs(r),
           "recovery_E": recovery(e["test_id"]["acc"], a0, a1), "recovery_M3": recovery(r["test_id"]["acc"], a0, a1),
           "mcnemar_E_vs_M3": mcnemar(e["test_id"]["ok_mask"], r["test_id"]["ok_mask"]),
           "E_ok_mask": e["test_id"]["ok_mask"], "M3_ok_mask": r["test_id"]["ok_mask"],
           "backward_calls": 0, "seconds": round(time.time() - t0)}
    (out_dir / f"E{a.k}_seed{a.seed}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print({k: v for k, v in rec.items() if not k.endswith("mask")}, flush=True)


if __name__ == "__main__":
    main()
