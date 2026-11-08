"""Activation-aware vs weights-only subspace, every large matrix of layers 0-3, cache only.

PRE-REGISTERED (written and committed locally before the first run of this script; F35 on layer 0 down_proj was seen before, nothing else):
 patch of rank r for a matrix W (delta = W_mimo - W_base), scored on the 8 HELD-OUT prompts' keys K_h (input of that matrix, base trajectory):
   weights-only : the probe's own basis (basis_*.pt: smaller side of delta, top r)             -> W_patch = delta V V^T  or  U U^T delta
   data-aware   : U = top-r left singular vectors of delta @ K_dev^T (4 dev prompts, 95 tokens)   -> U U^T delta
   control      : random orthonormal U of the same size on the output side
 metrics on held-out keys, comparing patch@K_h with delta@K_h: resid = |(delta-patch)K|/|delta K|, cos, norm_ratio = |patch K|/|delta K|, captured.
 Ranks 16 and 64 (dev has 95 tokens, so ranks above 95 cannot be fitted). Matrices with a side < 128 are skipped.
 Success for "data-aware is a better choice than weights-only": at BOTH ranks, data-aware has lower resid than weights-only in >= 75% of the matrices, AND the mean resid over
 matrices is lower by >= 0.01, AND data-aware beats the random control in >= 90% of the matrices. Otherwise: not shown (one failure at either rank is enough).
 Everything is local reproduction of an output difference of single matrices, not task benefit and not the composed stack.
"""
import json, sys
import torch
from lab.probes.delta_patterns import layer_probe as lp

cache = "/tmp/probe_cache"
RANKS = (16, 64)
rows = []
for L in range(4):
    pr = lp.Probe(cache, L)
    for n in pr.names:
        pr.params[n[len(pr.pre):]].data.copy_(pr.fb.get_tensor(n).float())
    mats = [n[len(pr.pre):] for n in pr.names if lp.is_matrix(n) and min(pr.fb.get_slice(n).get_shape()) >= 128]
    store = {m: [] for m in mats}
    hooks = []
    for m in mats:
        mod = pr.layer.get_submodule(m[:-len(".weight")])
        hooks.append(mod.register_forward_pre_hook(lambda mod, a, m=m: store[m].append(a[0].reshape(-1, a[0].shape[-1]).clone())))
    pr.forward()
    for h in hooks:
        h.remove()
    edge = [0]
    for s in pr.seqs:
        edge.append(edge[-1] + s.shape[1])
    for m in mats:
        n = pr.pre + m
        K = torch.cat(store[m]); store[m] = None
        Kd, Kh = K[:edge[lp.N_DEV]], K[edge[lp.N_DEV]:]
        a, b = pr._pair(n)
        d = b.float() - a.float()
        del a, b
        t = d @ Kh.T
        tn = t.norm()
        def sc(P):
            y = P @ Kh.T
            return {"resid": float((t - y).norm() / tn), "cos": float((y * t).sum() / (y.norm() * tn)),
                    "norm_ratio": float(y.norm() / tn), "captured": float((y * t).sum() / (tn ** 2))}
        Ud = torch.linalg.svd(d @ Kd.T, full_matrices=False)[0]
        side, v = pr.delta_basis(n)
        g = torch.Generator().manual_seed(7)
        for r in RANKS:
            Pw = d @ (v[:, :r] @ v[:, :r].T) if side == "right" else v[:, :r] @ (v[:, :r].T @ d)
            R = torch.linalg.qr(torch.randn(d.shape[0], r, generator=g)).Q
            row = {"layer": L, "tensor": m, "rank": r, "side_weights": side,
                   "weights_only": sc(Pw), "data_aware": sc(Ud[:, :r] @ (Ud[:, :r].T @ d)), "random": sc(R @ (R.T @ d))}
            del Pw
            rows.append(row)
        print(L, m, [(r["rank"], round(r["weights_only"]["resid"], 3), round(r["data_aware"]["resid"], 3), round(r["random"]["resid"], 3)) for r in rows[-2:]], flush=True)
    del pr
verdict = {}
for r in RANKS:
    rr = [x for x in rows if x["rank"] == r]
    wins = sum(x["data_aware"]["resid"] < x["weights_only"]["resid"] for x in rr) / len(rr)
    mean = lambda k: sum(x[k]["resid"] for x in rr) / len(rr)
    mc = lambda k, f: sum(x[k][f] for x in rr) / len(rr)
    beat_rand = sum(x["data_aware"]["resid"] < x["random"]["resid"] for x in rr) / len(rr)
    verdict[r] = {"n_matrices": len(rr), "win_rate_vs_weights": wins, "mean_resid": {k: mean(k) for k in ("weights_only", "data_aware", "random")},
                  "mean_cos": {k: mc(k, "cos") for k in ("weights_only", "data_aware", "random")},
                  "mean_norm_ratio": {k: mc(k, "norm_ratio") for k in ("weights_only", "data_aware", "random")},
                  "beats_random_rate": beat_rand,
                  "passes": wins >= 0.75 and mean("weights_only") - mean("data_aware") >= 0.01 and beat_rand >= 0.9}
verdict["success_both_ranks"] = all(verdict[r]["passes"] for r in RANKS)
print(json.dumps(verdict, indent=1))
json.dump({"verdict": verdict, "rows": rows}, open("/tmp/data_aware_stack.json", "w"), indent=1)
