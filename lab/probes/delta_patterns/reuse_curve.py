"""reuse_curve.py - how much of a fine-tune's delta a rank-k part carries (structural, no inference).

For chosen tensors of base and model (HTTP range reads, only those tensors are fetched) it computes the singular
spectrum of delta = W_model - W_base and reports the rank needed for 50/80/90 % of the delta energy (squared
Frobenius norm) and the energy a rank-k transplant keeps for k in KS. A transplant of the full delta reproduces the
model exactly (trivial); only partial parts can be a real reuse claim. This measures how much a part carries, NOT
whether applying it to the base improves behaviour (that needs inference and is not done here).

  python lab/probes/delta_patterns/reuse_curve.py BASE[@rev] MODEL[@rev] out.json tensor [tensor ...]
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
    return {"shape": list(delta.shape), "min_dim": len(cum), **need, **kept, **sparsity(delta)}


def sparsity(delta: torch.Tensor, chunk_rows: int = 1024) -> dict:
    """Energy kept by the largest entries (TIES/DARE-style pruning) and by the largest rows / columns (a neuron-
    or channel-sparse patch). Chance level for a uniform delta equals the fraction kept. Memory-light: entry
    thresholds come from a 1M-entry sample (quantile error is negligible at this size), energy is summed in chunks."""
    n = delta.numel()
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, n, (min(n, 1_000_000),), generator=g)
    sample = delta.reshape(-1)[idx].float().abs()
    fr = (0.01, 0.1, 0.3)
    thr = [float(torch.quantile(sample, 1 - f)) for f in fr]
    kept = [0.0] * len(fr)
    tot = 0.0
    rows, cols = [], torch.zeros(delta.shape[1], dtype=torch.float64)
    for i in range(0, delta.shape[0], chunk_rows):
        x = delta[i:i + chunk_rows].float()
        e = x.pow(2)
        tot += float(e.sum())
        for j, t in enumerate(thr):
            kept[j] += float(e[x.abs() >= t].sum())
        rows.append(e.sum(1).double())
        cols += e.sum(0).double()
    out = {f"energy_top{int(f * 100)}pct_entries": kept[j] / tot for j, f in enumerate(fr)}
    rows = torch.cat(rows)
    for v, name in ((rows, "rows"), (cols, "cols")):
        v = v.sort(descending=True).values
        for f in (0.01, 0.1):
            out[f"energy_top{int(f * 100)}pct_{name}"] = float(v[: max(1, int(f * v.numel()))].sum() / tot)
    return out


def _reader(spec: str):
    """`repo` or `repo@revision`; pin the revision for any run that is meant to be reproduced."""
    repo, _, rev = spec.partition("@")
    if "/" in repo and not __import__("pathlib").Path(repo).exists():
        return wd.Renamed(wd.RangeSafetensors(wd.hf_files(repo, rev or "main"), wd.hf_fetch(repo, rev or "main")),
                          wd.text_model_name)
    return mr.reader_for(spec)


def main(base, model, out, names):
    rb, rm = _reader(base), _reader(model)
    res = {"base": base, "model": model, "tensors": {}}
    for n in names:
        a, b = rb.get(n), rm.get(n)
        a, b = wd.common_dtype(a, b)
        res["tensors"][n] = curve(b.float() - a.float())
        print(n, res["tensors"][n], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
