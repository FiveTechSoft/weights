"""engine/delta_apply.py - apply a chosen part of (model - base) to base tensors, one tensor at a time, and say exactly what was done.

This is the direct route for "take the changes of model M and put some of them on the base": per tensor, out = base + alpha * delta,
where delta = model - base computed in float32 and cast back to the base dtype. Selection is by tensor name and layer range;
mode "full" uses the whole delta, mode "rank" keeps the top-k singular directions of each chosen matrix (lossy: on MiMo vs Qwen3.5-9B
rank 16 keeps only 3-7% of the delta energy, see lab/probes/delta_patterns/findings.md F27, F32).

  apply_selected(base, model, select, alpha=1.0, rank=None) -> (tensors, report)
    base, model : objects with keys() and get(name) (RangeSafetensors, LocalSafetensors, Renamed, a dict wrapper)
    select(name): True for the tensors to change; every other tensor is returned as the base tensor, untouched
    tensors     : a generator-friendly dict name -> tensor; the inputs are never modified (rollback is by source: keep the base)
    report      : per tensor: changed / skipped with the reason (dtype or shape mismatch is skipped, never cast silently), sha256 of
                  base, model and output, energy fraction of the delta kept. Nothing in it says the result is better than the base.
With alpha=1 and mode "full" the output equals the model up to float32 rounding in a few subnormal-sized elements (measured on layer 0:
2-27 elements per matrix, abs error <= 1.2e-10). That is a tautology, not a result.
"""
from typing import Callable, Dict, Optional, Tuple

import torch

from engine.delta_pack import low_rank_factors, tensor_sha256


def layer_range(lo: int, hi: int) -> Callable[[str], bool]:
    """Select tensors of decoder layers lo..hi (inclusive) by their name 'model.layers.<n>.'"""
    def sel(name: str) -> bool:
        parts = name.split(".")
        return len(parts) > 2 and parts[:2] == ["model", "layers"] and parts[2].isdigit() and lo <= int(parts[2]) <= hi
    return sel


def apply_selected(base, model, select: Callable[[str], bool], alpha: float = 1.0, rank: Optional[int] = None
                   ) -> Tuple[Dict[str, torch.Tensor], list]:
    out, report = {}, []
    for n in base.keys():
        a = base.get(n)
        out[n] = a
        if not select(n):
            continue
        entry = {"name": n, "base_sha256": tensor_sha256(a)}
        if n not in set(model.keys()):
            report.append({**entry, "status": "skipped", "reason": "missing in model"})
            continue
        b = model.get(n)
        if a.shape != b.shape or a.dtype != b.dtype:
            report.append({**entry, "status": "skipped", "reason": f"shape/dtype differ: {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"})
            continue
        d = b.float() - a.float()
        kept = 1.0
        if rank is not None:
            if d.ndim != 2:
                report.append({**entry, "status": "skipped", "reason": "rank mode needs a matrix"})
                continue
            left, right, kept = low_rank_factors(d, rank)
            d = left @ right
        out[n] = (a.float() + alpha * d).to(a.dtype)
        report.append({**entry, "status": "changed", "alpha": alpha, "rank": rank, "delta_energy_kept": kept,
                       "model_sha256": tensor_sha256(b), "output_sha256": tensor_sha256(out[n])})
    return out, report
