"""Causal column screening with matched controls and one-shot holdout.

The evaluator is supplied by the caller and returns a finite scalar score
(higher is better). This module does not certify split independence or
benchmark validity. It never writes checkpoints.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Candidate:
    tensor: str
    column: int


@torch.no_grad()
def probe_columns(tensors, candidates, evaluate, *, dose, seed=1337,
                  top=5, min_gain=.05, min_control_gap=.05,
                  preservation_tolerance=.02, max_candidates=256,
                  max_modified_weights=4096):
    """Screen on discovery; validate a fixed shortlist once on holdout.

    evaluate(split) must run the live model, with split one of discovery,
    heldout, preservation. The random-direction placebo has the same column
    and Frobenius norm as the candidate edit (up to dtype rounding).
    The caller must supply disjoint discovery/holdout data.
    """
    candidates = list(candidates)
    if not 0 < len(candidates) <= max_candidates or len(set(candidates)) != len(candidates):
        raise ValueError("invalid candidate count or duplicate")
    if not isinstance(top,int) or not 0 < top <= max_candidates:
        raise ValueError("invalid shortlist size")
    if not all(math.isfinite(v) for v in (dose,min_gain,min_control_gap,preservation_tolerance)):
        raise ValueError("nonfinite thresholds")
    if not 0 < dose <= 1 or min_gain <= 0 or min_control_gap <= 0 or preservation_tolerance < 0:
        raise ValueError("invalid thresholds")
    if not isinstance(max_modified_weights,int) or max_modified_weights < 1:
        raise ValueError("invalid weight budget")
    for c in candidates:
        w = tensors[c.tensor]
        if (w.ndim != 2 or not w.is_floating_point() or not 0 <= c.column < w.shape[1]
                or w.shape[0] > max_modified_weights):
            raise ValueError("invalid tensor, column or weight budget")
        if not torch.isfinite(w[:,c.column]).all():
            raise ValueError("nonfinite column")

    def score(split):
        value = float(evaluate(split))
        if not math.isfinite(value):
            raise ValueError("nonfinite metric")
        return value

    baseline = {s:score(s) for s in ("discovery","preservation")}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    edits = {}
    rows = []
    def trial(c, replacement, split):
        w = tensors[c.tensor]
        col = w[:,c.column]
        old = col.clone()
        try:
            col.copy_(replacement.to(old))
            if not torch.isfinite(col).all():
                raise ValueError("nonfinite intervention")
            return score(split)
        finally:
            col.copy_(old)

    for c in candidates:
        col = tensors[c.tensor][:,c.column].detach()
        replacement = (1-dose)*col
        norm = float((replacement.double()-col.double()).norm())
        if norm == 0:
            continue
        random = torch.randn(col.shape,generator=rng,dtype=torch.float64)
        random = (random/random.norm()*norm).to(col)
        placebo = col + random
        placebo_norm = float((placebo.double()-col.double()).norm())
        if abs(placebo_norm-norm) > .02*norm:
            continue
        edits[c] = (replacement,placebo)
        effect = trial(c,replacement,"discovery")-baseline["discovery"]
        control = trial(c,placebo,"discovery")-baseline["discovery"]
        drift = trial(c,replacement,"preservation")-baseline["preservation"]
        rows.append(dict(tensor=c.tensor,column=c.column,discovery_gain=effect,
                         discovery_placebo_gain=control,preservation_gain=drift,
                         delta_norm=norm,placebo_norm=placebo_norm,
                         eligible=effect>=min_gain and effect-control>=min_control_gap
                                  and drift>=-preservation_tolerance))
    rows.sort(key=lambda r:(-r["discovery_gain"]+r["discovery_placebo_gain"],
                            -r["discovery_gain"],r["tensor"],r["column"]))
    shortlist = [r for r in rows if r["eligible"]][:top]
    # No holdout query before the shortlist is fixed.
    baseline["heldout"] = score("heldout")
    for r in shortlist:
        c = Candidate(r["tensor"],r["column"])
        replacement,placebo = edits[c]
        gain = trial(c,replacement,"heldout")-baseline["heldout"]
        control = trial(c,placebo,"heldout")-baseline["heldout"]
        r.update(heldout_gain=gain,heldout_placebo_gain=control,
                 passed=gain>=min_gain and gain-control>=min_control_gap)
    return dict(schema_version=1,baseline=baseline,candidates=rows,
                shortlist=[dict(r) for r in shortlist],seed=seed,dose=dose,
                thresholds=dict(min_gain=min_gain,min_control_gap=min_control_gap,
                                preservation_tolerance=preservation_tolerance),
                backward_calls=0,checkpoint_written=False)


def rank_candidates(report):
    """Keep discovery order; holdout only accepts or rejects, never re-ranks."""
    if report.get("schema_version") != 1:
        raise ValueError("unsupported probe report")
    return [dict(r) for r in report["shortlist"] if r.get("passed") is True]
