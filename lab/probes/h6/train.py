"""H6 stage 0 training: Qwen3.5-0.8B on 3x2-digit products, 10 chunks x 50 steps (batch 40), every step logged.

Only the 2-D matrices of model.layers.* train (fp32 master weights); embeddings, 1-D tensors and conv1d are frozen.
AdamW is implemented here, one parameter at a time, so each applied update u is measured exactly without a 2 GB copy
(tested against torch.optim.AdamW in tests/test_h6.py).

Logged as a dataset of small changes (h6_plan.md addenda), under lab/runs/h6/seed<s>/:
  steps.jsonl       one row per step: chunk, step, lr, loss, the training-example indices of the batch, and per
                    matrix g_norm, u_norm, cos(g,u), cos(u_t,u_{t-1}) (the last two through the sketch)
  sketch_c<t>.pt    chunks 1 and 10: per step, per matrix, the sketch u . R (fp16), R fixed per matrix (seed 0)
  chunk<t>.json     after each chunk: evaluation (test_id, ood_easy, ood_hard, retention NLL) and fingerprint
hf_cache/h6/seed<s>/chunk<t>.safetensors: the trained matrices (fp32) after chunks 1, 2 and 10.
Batch indices + seed make every frame replayable.

usage: python lab/probes/h6/train.py --seed 1 [--chunks 10] [--smoke]
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import make_data as md  # noqa: E402

BATCH, STEPS_PER_CHUNK, LR, WD, BETAS, EPS = 40, 50, 1e-5, 0.0, (0.9, 0.95), 1e-8
SKETCH_K, SKETCH_CHUNKS, KEEP_CHUNKS = 8, (1, 10), (1, 2, 10)


# ---------- pieces with tests ----------

class AdamW:
    """Decoupled AdamW. step(lr) applies the update and returns per-parameter stats incl. the update u itself."""

    def __init__(self, params, betas=BETAS, eps=EPS, weight_decay=WD):
        self.params, self.b1, self.b2, self.eps, self.wd = list(params), betas[0], betas[1], eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self, lr, keep_update=False):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        stats = []
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            u = -lr * ((m / c1) / ((v / c2).sqrt() + self.eps) + self.wd * p)
            gn, un = float(g.norm()), float(u.norm())
            stats.append({"g_norm": gn, "u_norm": un,
                          "cos_gu": float(-(g * u).sum()) / (gn * un) if gn > 0 and un > 0 else 0.0,
                          "u": u if keep_update else None})
            p.add_(u)
            p.grad = None
        return stats


def encode_pair(prompt_ids, target_ids):
    return list(prompt_ids) + list(target_ids), [-100] * len(prompt_ids) + list(target_ids)


def sketch_matrix(shape, k, seed):
    """Gaussian (larger side x k) / sqrt(k): u . R keeps inner products of the larger side in expectation."""
    big = max(shape)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(big, k, generator=g) / math.sqrt(k)


def sketch(u, R):
    """Sketch on the larger side; result is (smaller side x k)."""
    return (u @ R.to(u.device, u.dtype)) if u.shape[1] >= u.shape[0] else (u.T @ R.to(u.device, u.dtype))


# ---------- run ----------

def trained_params(model):
    return [(n, p) for n, p in model.named_parameters() if ".layers." in n and p.dim() == 2]


def snapshot(named):
    return {n: p.detach().float().cpu().clone() for n, p in named}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="2 chunks of 3 steps, small eval: memory and speed check")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    import eval_arith as ea
    import fingerprint as fp
    from safetensors.torch import save_file

    steps_per_chunk = 3 if a.smoke else STEPS_PER_CHUNK
    chunks = 2 if a.smoke else a.chunks
    out = ROOT / "lab" / "runs" / "h6" / (f"smoke{a.seed}" if a.smoke else f"seed{a.seed}")
    keep = ROOT / "hf_cache" / "h6" / out.name
    out.mkdir(parents=True, exist_ok=True)
    keep.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = ea.load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(str(ea.BASE), dtype=torch.float32).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    named = trained_params(model)
    for _, p in named:
        p.requires_grad_(True)
    opt = AdamW([p for _, p in named])
    Rs = {n: sketch_matrix(tuple(p.shape), SKETCH_K, seed=i) for i, (n, p) in enumerate(named)}

    train = ea.load_split("train")
    order = list(range(len(train)))
    random.Random(a.seed).shuffle(order)
    total = steps_per_chunk * chunks
    enc = {}

    def batch_for(idx):
        rows = []
        for i in idx:
            if i not in enc:
                r = train[i]
                enc[i] = encode_pair(tok.encode(md.prompt(r["a"], r["b"]), add_special_tokens=False).ids,
                                     tok.encode(md.target(r["a"], r["b"]), add_special_tokens=False).ids)
            rows.append(enc[i])
        L = max(len(x) for x, _ in rows)
        ids = torch.tensor([x + [0] * (L - len(x)) for x, _ in rows]).cuda()        # right padding: causal, safe
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in rows]).cuda()
        return ids, lab

    def run_eval():
        if a.smoke:
            return {"test_id": {k: v for k, v in ea.eval_split(model, tok, ea.load_split("test_id", 100)).items()
                                if k != "items"}}
        return ea.evaluate(model, tok)

    base = snapshot(named)
    ev0 = run_eval()
    (out / "chunk0.json").write_text(json.dumps({"chunk": 0, "eval": ev0}, indent=1), encoding="utf-8")
    print("chunk 0", {k: (v["acc"] if isinstance(v, dict) else v) for k, v in ev0.items()}, flush=True)
    prev_prev, prev = None, base
    prev_sk = None
    step = 0
    t0 = time.time()
    with (out / "steps.jsonl").open("w", encoding="utf-8") as log:
        for c in range(1, chunks + 1):
            sketches = [] if c in SKETCH_CHUNKS or a.smoke else None
            for _ in range(steps_per_chunk):
                idx = order[(step * BATCH) % len(order):(step * BATCH) % len(order) + BATCH]
                lr = LR * 0.5 * (1 + math.cos(math.pi * step / total))
                model.train()
                ids, lab = batch_for(idx)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids).logits
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                                         lab[:, 1:].reshape(-1), ignore_index=-100)
                loss.backward()
                del logits
                st = opt.step(lr, keep_update=True)
                sk_now, rows = {}, {}
                for (n, _), s in zip(named, st):
                    sk = sketch(s.pop("u"), Rs[n])
                    sk_now[n] = sk
                    if prev_sk is not None:
                        q = prev_sk[n]
                        s["cos_uu"] = float((sk * q).sum() / (sk.norm() * q.norm()).clamp_min(1e-30))
                    rows[n] = {k: round(v, 6) for k, v in s.items()}
                if sketches is not None:
                    sketches.append({n: v.half().cpu() for n, v in sk_now.items()})
                prev_sk = sk_now
                step += 1
                log.write(json.dumps({"chunk": c, "step": step, "lr": lr, "loss": float(loss),
                                      "batch": idx, "tensors": rows}) + "\n")
                log.flush()
            cur = snapshot(named)
            ev = run_eval()
            fpr = fp.chunk_fingerprint(base, prev, cur, r=16, prev_prev=prev_prev)
            rec = {"chunk": c, "step": step, "seconds": round(time.time() - t0), "eval": ev,
                   "fingerprint": fpr, "summary": fp.summarize(fpr)}
            (out / f"chunk{c}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            if sketches is not None:
                torch.save(sketches, out / f"sketch_c{c}.pt")
            if c in KEEP_CHUNKS or a.smoke:
                save_file({n: t.contiguous() for n, t in cur.items()}, str(keep / f"chunk{c}.safetensors"),
                          metadata={"provenance": "h6_training", "seed": str(a.seed), "chunk": str(c),
                                    "step": str(step)})
            print(f"chunk {c} step {step} {time.time() - t0:.0f}s loss {float(loss):.4f} acc "
                  f"{ev['test_id']['acc']:.3f} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
            prev_prev, prev = prev, cur


if __name__ == "__main__":
    main()
