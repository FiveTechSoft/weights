"""Fingerprint of one small training chunk (H6 stage 0, lab/probes/delta_patterns/h6_plan.md).

Given the base W_0, the previous checkpoint W_{t-1} and the current one W_t (dicts name -> 2-D tensor), for each
matrix: increment I = W_t - W_{t-1}, cumulative C = W_t - W_0 and
  rel        |I| / |W_0|
  share      |I|^2 / sum over all matrices of |I|^2
  eff_rank   exp(entropy of the normalised squared singular values of I)
  top1/top5  energy share in the top 1 % / 5 % of I's singular directions
  overlap_prev_inc  overlap of I's top-r subspace with the previous increment's (None at the first chunk)
  overlap_cum       overlap of I's top-r subspace with that of C_{t-1}       (None at the first chunk)
Subspaces live on the matrix's smaller side. overlap(A, B) = |A^T B|_F^2 / r: 1 = same subspace, r/d = chance.
"""
import math
import re
from typing import Dict, Optional

import torch


def small_side(shape) -> int:
    rows, cols = shape
    return 1 if cols <= rows else 0


def kind(name: str) -> str:
    m = re.search(r"layers\.\d+\.(.+)\.weight$", name)
    return m.group(1) if m else name


def layer(name: str) -> Optional[int]:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def eff_rank(S: torch.Tensor) -> float:
    p = S.double().pow(2)
    p = p[p > 0] / p.sum()
    return float(math.exp(-(p * p.log()).sum()))


def _svd(D: torch.Tensor, device=None, dtype=torch.float64):
    """Singular values and the left singular vectors on the smaller side (computed on `device`, returned on CPU)."""
    M = D.to(device or D.device, dtype)
    M = M if small_side(D.shape) == 0 else M.T                     # rows = smaller side
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    return U.cpu().double(), S.cpu().double()


def top_subspace(D: torch.Tensor, r: int, device=None, dtype=torch.float64) -> torch.Tensor:
    return _svd(D, device, dtype)[0][:, :r].float()


def top_energy(D: torch.Tensor, k: int) -> float:
    S2 = _svd(D)[1].pow(2)
    return float(S2[:k].sum() / S2.sum())


def overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    return float((A.double().T @ B.double()).pow(2).sum()) / min(A.shape[1], B.shape[1])


def chunk_fingerprint(base: Dict, prev: Dict, cur: Dict, r: int = 16, prev_prev: Optional[Dict] = None,
                      device=None) -> Dict:
    """prev_prev: checkpoint before prev, needed for overlap_prev_inc; if None and prev is not base, the previous
    increment is taken as prev - base (exact for the second chunk). device="cuda": SVDs in float32 on the GPU."""
    dt = torch.float32 if device else torch.float64
    first = all(torch.equal(prev[n], base[n]) for n in base)
    tot = sum(float((cur[n].double() - prev[n].double()).pow(2).sum()) for n in base)
    rows = []
    for n in sorted(base):
        inc = cur[n].float() - prev[n].float()
        U, S = _svd(inc, device, dt)
        S2 = S.pow(2)
        d = min(inc.shape)
        row = {"tensor": n, "layer": layer(n), "kind": kind(n),
               "rel": float(inc.norm() / base[n].float().norm()),
               "share": float(S2.sum()) / tot if tot > 0 else 0.0,
               "eff_rank": eff_rank(S),
               "top1": float(S2[:max(1, d // 100)].sum() / S2.sum()) if S2.sum() > 0 else 0.0,
               "top5": float(S2[:max(1, d // 20)].sum() / S2.sum()) if S2.sum() > 0 else 0.0,
               "overlap_prev_inc": None, "overlap_cum": None, "chance": r / d}
        if not first:
            Ui = U[:, :r].float()
            pinc = prev[n].float() - (prev_prev[n].float() if prev_prev is not None else base[n].float())
            row["overlap_prev_inc"] = overlap(Ui, top_subspace(pinc, r, device, dt))
            row["overlap_cum"] = overlap(Ui, top_subspace(prev[n].float() - base[n].float(), r, device, dt))
        rows.append(row)
    return {"r": r, "total_sq_norm": tot, "tensors": rows}


def cross_overlap(base: Dict, wa: Dict, wb: Dict, r: int = 16) -> Dict[str, float]:
    """Per matrix: overlap of the top-r subspaces of (wa - base) and (wb - base), e.g. the same chunk of two seeds."""
    return {n: overlap(top_subspace(wa[n].float() - base[n].float(), r),
                       top_subspace(wb[n].float() - base[n].float(), r)) for n in sorted(base)}


def summarize(rep: Dict) -> Dict:
    """Medians by matrix kind and by depth quarter."""
    import statistics as st
    out = {}
    for key in ("kind", "quarter"):
        groups = {}
        for t in rep["tensors"]:
            g = t["kind"] if key == "kind" else f"q{(t['layer'] or 0) * 4 // 24}"
            groups.setdefault(g, []).append(t)
        out[key] = {g: {m: (round(st.median(x[m] for x in v), 4) if v[0][m] is not None else None)
                        for m in ("rel", "eff_rank", "top5", "overlap_prev_inc", "overlap_cum")}
                    | {"share": round(sum(x["share"] for x in v), 4)} for g, v in sorted(groups.items())}
    return out
