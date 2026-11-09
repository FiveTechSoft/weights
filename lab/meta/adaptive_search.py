"""Sequential configuration search with a prior from past tasks and updates from the new task's own trials.

Config = (layer, eta, reg). Position of a config for a task: (layer - old_layer, log2 eta). The relative
layer encoding is OUR design (see v3 to v5). Hyper-parameters are fixed in the v6 pre-registration.

  prior_mean(c)   : mean score of c's relative key over the experience of the K nearest tasks (nearest by
                    task features, family included); global mean of those tasks if the key is missing.
  prediction(c)   : prior_mean(c) + sum_i w(c, c_i) * (y_i - prior_mean(c_i)) / (1 + sum_i w(c, c_i)),
                    w = exp(-d^2 / 2), d^2 = (d_offset / 1.0)^2 + (d_log2eta / 0.5)^2.
  next trial      : the untried config with the highest prediction (ties: the given tie order).
A flat prior (prior_mean = 0 everywhere) keeps the update and removes the transfer.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

from lab.meta.meta_learner import Config, Experience, Features, _distance

LEN_OFFSET, LEN_ETA, K = 1.0, 0.5, 3


def _old(f):
    return round(f[0] * 2)


def _pos(c: Config, f: Features):
    return (c[0] - _old(f), math.log2(c[1]))


def prior_table(experience: Experience, f: Features, k: int = K):
    """Return a function config -> prior mean for the new task; 0.0 everywhere when experience is empty."""
    if not experience:
        return lambda c: 0.0
    near = sorted(experience, key=lambda e: _distance(e[0], f))[:k]
    pooled: Dict[tuple, List[float]] = {}
    allv: List[float] = []
    for g, scores in near:
        for c, v in scores.items():
            pooled.setdefault((c[0] - _old(g), c[1], c[2]), []).append(v)
            allv.append(v)
    glob = sum(allv) / len(allv)
    def mu(c):
        vals = pooled.get((c[0] - _old(f), c[1], c[2]))
        return sum(vals) / len(vals) if vals else glob
    return mu


def search(prior: Callable[[Config], float], f: Features, configs: Sequence[Config],
           observe: Callable[[Config], float], budget: int, update: bool = True,
           tie_order: Optional[Sequence[Config]] = None) -> List[tuple]:
    """Run up to `budget` trials. Returns [(config, observed score), ...]."""
    tie = list(tie_order) if tie_order is not None else list(configs)
    rank = {c: i for i, c in enumerate(tie)}
    tried: List[tuple] = []
    for _ in range(budget):
        left = [c for c in configs if c not in {t[0] for t in tried}]
        if not left:
            break
        def pred(c):
            p = prior(c)
            if not update or not tried:
                return p
            pc = _pos(c, f)
            num = den = 0.0
            for ci, y in tried:
                pi = _pos(ci, f)
                d2 = ((pc[0] - pi[0]) / LEN_OFFSET) ** 2 + ((pc[1] - pi[1]) / LEN_ETA) ** 2
                w = math.exp(-d2 / 2)
                num += w * (y - prior(ci)); den += w
            return p + num / (1 + den)
        best = max(left, key=lambda c: (pred(c), -rank[c]))
        tried.append((best, observe(best)))
    return tried


def fixed_recipe_order(f: Features, configs: Sequence[Config]) -> List[Config]:
    """Learned-and-fixed recipe from v1 to v5: write at the old layer with eta near 1.0, then neighbours by
    |offset| + |log2 eta|. No experience, no observations."""
    return sorted(configs, key=lambda c: (abs(c[0] - _old(f)) + abs(math.log2(c[1])), list(configs).index(c)))
