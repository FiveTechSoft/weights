"""H6 stage 1 (h6_plan.md + addendum c): how much of ΔW_s keeps the skill, and does the lane of OTHER seeds carry it.

Per seed s (ΔW_s = chunk-10 checkpoint − base), r ∈ {10, 20, 51, 102, 256}:
  M2_r  rank-r truncation of ΔW_s (top-r singular directions on the smaller side)
  M3_r  ΔW_s projected on r random orthonormal directions (smaller side), rescaled per matrix to |M2_r|
  X_r   ΔW_s projected on the top-r directions of [ΔW_t | ΔW_u], t, u = the other two seeds (the "lane")
Matrices whose smaller side ≤ r are kept whole (projection = identity).
Each arm: full evaluation (test_id, ood_easy, ood_hard, retention NLL), recovery, energy share, exact McNemar
against M3_r. Results written after every arm to lab/runs/h6/transfer/seed<s>.json.

usage: python lab/probes/h6/transfer.py --seed 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "lab" / "runs" / "h6"
KEEP = ROOT / "hf_cache" / "h6"
sys.path.insert(0, str(HERE))
import fingerprint as fp  # noqa: E402

RANKS = (10, 20, 51, 102, 256)


# ---------- pieces with tests ----------

def top_basis(D, r, device=None):
    return fp.top_subspace(D, r, device, torch.float32 if device else torch.float64)


def project(D, U):
    """Project D on span(U), U lying on D's smaller side."""
    U = U.to(D.device, D.dtype)
    return U @ (U.T @ D) if fp.small_side(D.shape) == 0 else (D @ U) @ U.T


def random_basis(d, r, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, r, generator=g))[0]


def lane_basis(Da, Db, r, device=None):
    """Top-r directions, on the smaller side, shared by two other runs' deltas (concatenated along the larger side)."""
    cat = torch.cat([Da, Db], dim=1 if fp.small_side(Da.shape) == 0 else 0)
    return top_basis(cat, r, device)


def rescale_to(A, norm):
    n = A.norm()
    return A * (norm / n) if n > 0 else A


def other_seeds(s):
    return [t for t in (1, 2, 3) if t != s]


def lane_bases(base, seed, rmax, device="cuda"):
    """Per matrix: the top-rmax lane basis of the other two seeds' chunk-10 deltas (CPU float32)."""
    from safetensors.torch import load_file
    t, u = other_seeds(seed)
    wa = load_file(str(KEEP / f"seed{t}" / "chunk10.safetensors"))
    wb = load_file(str(KEEP / f"seed{u}" / "chunk10.safetensors"))
    out, norms = {}, []
    for n in sorted(base):
        b = base[n].to(device)
        Da, Db = wa[n].to(device) - b, wb[n].to(device) - b
        norms.append((float(Da.pow(2).sum()), float(Db.pow(2).sum())))
        d = min(Da.shape)
        out[n] = lane_basis(Da, Db, min(rmax, d), device) if d > rmax else None
    del wa, wb
    N = (sum(x for x, _ in norms) ** 0.5 + sum(y for _, y in norms) ** 0.5) / 2
    return out, N


# ---------- run ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    import eval_arith as ea
    import synth

    out_dir = RUNS / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / f"seed{a.seed}.json"
    tok = ea.load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(str(ea.BASE), dtype=torch.float32).cuda()
    model.model.embed_tokens.to(torch.bfloat16)
    params = dict(model.named_parameters())
    own = load_file(str(KEEP / f"seed{a.seed}" / "chunk10.safetensors"))
    names = sorted(own)
    base = {n: params[n].detach().float().cpu().clone() for n in names}
    D = {n: own[n] - base[n] for n in names}
    del own
    t0 = time.time()
    rmax = max(RANKS)
    lane, N_other = lane_bases(base, a.seed, rmax)
    U_own, U_rnd = {}, {}
    for i, n in enumerate(names):
        d = min(D[n].shape)
        U_own[n] = top_basis(D[n], rmax, "cuda") if d > rmax else None
        U_rnd[n] = random_basis(d, rmax, seed=10_000 * a.seed + i) if d > rmax else None
    print(f"bases {time.time() - t0:.0f}s", flush=True)
    tot = sum(float(D[n].pow(2).sum()) for n in names)
    m0 = json.loads((RUNS / f"seed{a.seed}" / "chunk0.json").read_text(encoding="utf-8"))["eval"]
    m1 = json.loads((RUNS / f"seed{a.seed}" / "chunk10.json").read_text(encoding="utf-8"))["eval"]
    a0, a1 = m0["test_id"]["acc"], m1["test_id"]["acc"]
    rec = {"seed": a.seed, "ranks": list(RANKS), "M0": m0["test_id"]["acc"], "M1": a1,
           "whole_share": sum(float(D[n].pow(2).sum()) for n in names if min(D[n].shape) <= min(RANKS)) / tot,
           "arms": {}}

    @torch.no_grad()
    def build(r, arm):
        e = 0.0
        for n in names:
            Dn = D[n].cuda()
            if min(Dn.shape) <= r:
                P = Dn
            elif arm == "M2":
                P = project(Dn, U_own[n][:, :r])
            elif arm == "X":
                P = project(Dn, lane[n][:, :r])
            else:
                P = rescale_to(project(Dn, U_rnd[n][:, :r]), project(Dn, U_own[n][:, :r]).norm())
            e += float(P.pow(2).sum())
            params[n].copy_(base[n].cuda() + P)
        return e / tot

    for r in RANKS:
        for arm in ("M3", "M2", "X"):          # M3 first: the others are tested against it
            en = build(r, arm)
            ev = ea.evaluate(model, tok)
            rec["arms"][f"{arm}_{r}"] = {
                "arm": arm, "r": r, "energy": en, "test_id": ev["test_id"]["acc"],
                "test_id_independent": ev["test_id"]["acc_independent"], "ood_easy": ev["ood_easy"]["acc"],
                "ood_hard": ev["ood_hard"]["acc"], "retention_nll": ev["retention_nll"],
                "recovery": synth.recovery(ev["test_id"]["acc"], a0, a1), "ok_mask": ev["test_id"]["ok_mask"]}
            if arm != "M3":
                rec["arms"][f"{arm}_{r}"]["mcnemar_vs_M3"] = synth.mcnemar(ev["test_id"]["ok_mask"],
                                                                           rec["arms"][f"M3_{r}"]["ok_mask"])
            outp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            x = rec["arms"][f"{arm}_{r}"]
            print(f"{arm}_{r} E={en:.3f} acc={x['test_id']:.3f} R={x['recovery']:.2f} "
                  f"easy={x['ood_easy']:.3f} nll={x['retention_nll']:.4f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
