"""reuse_curve.py - how much of a fine-tune's delta a rank-k part carries (structural, no inference).

For chosen tensors of base and model (HTTP range reads, only those tensors are fetched) it computes the singular
spectrum of delta = W_model - W_base and reports the rank needed for 50/80/90 % of the delta energy (squared
Frobenius norm) and the energy a rank-k transplant keeps for k in KS. A transplant of the full delta reproduces the
model exactly (trivial); only partial parts can be a real reuse claim. This measures how much a part carries, NOT
whether applying it to the base improves behaviour (that needs inference and is not done here).

  python lab/probes/delta_patterns/reuse_curve.py BASE MODEL out.json tensor [tensor ...]
"""
import json
import sys

import torch

from engine import model_report as mr
from engine import weight_diff as wd

KS = (16, 64, 256, 1024)


def curve(delta: torch.Tensor) -> dict:
    d = delta.float()
    if d.shape[0] < d.shape[1]:
        d = d.T
    s = torch.linalg.eigvalsh(d.T @ d).clamp_min(0).flip(0)  # squared singular values, descending
    e = s / s.sum()
    cum = e.cumsum(0)
    need = {f"rank_for_{int(q * 100)}pct": int((cum < q).sum().item()) + 1 for q in (0.5, 0.8, 0.9)}
    kept = {f"energy_rank_{k}": float(cum[min(k, len(cum)) - 1]) for k in KS}
    return {"shape": list(delta.shape), "min_dim": len(cum), **need, **kept}


def main(base, model, out, names):
    rb, rm = mr.reader_for(base), mr.reader_for(model)
    res = {"base": base, "model": model, "tensors": {}}
    for n in names:
        a, b = rb.get(n), rm.get(n)
        a, b = wd.common_dtype(a, b)
        res["tensors"][n] = curve(b.float() - a.float())
        print(n, res["tensors"][n], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
