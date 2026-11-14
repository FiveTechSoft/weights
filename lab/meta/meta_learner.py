"""Experience-driven proposer for WSP experiments ("learning to learn", measurable version).

What is measured here, and what is not
--------------------------------------
Object level: a WSP intervention (closed-form key-value imprint into a NanoWSP-3 MLP layer)
has a configuration (target layer, eta, regularisation) and a score (recall on the stored facts).

Meta level: a proposer that sees the outcomes of earlier experiments and chooses which
configuration to try next on a NEW task. "Learning to learn" is claimed only if, on tasks it has
never seen, the proposer needs fewer trials to reach a success threshold than controls that
had no experience, the same experience without its task labels (shuffled control), or a single
static ranking learned from the same experience (no context).

This module is pure logic (no torch): it works on a table of scores, so it is cheap to unit test.
The table itself is produced by real runs in grid.py.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

Config = Tuple[int, float, float]          # (target_layer, eta, reg_lambda)
Features = Tuple[float, float]             # (log2 n_facts, key_similarity_prefix_len)
Experience = List[Tuple[Features, Dict[Config, float]]]   # one entry per finished task


def trials_to_success(order: Sequence[Config], scores: Dict[Config, float],
                      threshold: float, budget: int) -> int:
    """Number of configurations tried until one reaches `threshold`.
    Censored at budget + 1 when none does within `budget` trials."""
    for i, cfg in enumerate(order[:budget], start=1):
        if scores[cfg] >= threshold:
            return i
    return budget + 1


def _distance(a: Features, b: Features) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def static_order(experience: Experience, configs: Sequence[Config]) -> List[Config]:
    """Rank configurations by their mean score over all past tasks (ignores the new task)."""
    if not experience:
        return list(configs)
    mean = {c: sum(s[c] for _, s in experience) / len(experience) for c in configs}
    return sorted(configs, key=lambda c: (-mean[c], configs.index(c)))


def contextual_order(experience: Experience, features: Features,
                     configs: Sequence[Config], k: int = 3) -> List[Config]:
    """Rank configurations by their mean score on the k past tasks most similar to this one."""
    if not experience:
        return list(configs)
    near = sorted(experience, key=lambda e: _distance(e[0], features))[:k]
    mean = {c: sum(s[c] for _, s in near) / len(near) for c in configs}
    return sorted(configs, key=lambda c: (-mean[c], configs.index(c)))


def shuffled_experience(experience: Experience, rng: random.Random) -> Experience:
    """Control: keep every score table but attach it to the wrong task features."""
    feats = [f for f, _ in experience]
    rng.shuffle(feats)
    return [(f, s) for f, (_, s) in zip(feats, experience)]


def random_expected_trials(scores: Dict[Config, float], threshold: float, budget: int) -> float:
    """Exact expected trials-to-success of a uniformly random order (censored at budget + 1)."""
    n = len(scores)
    good = sum(1 for v in scores.values() if v >= threshold)
    if good == 0:
        return float(budget + 1)
    expected = 0.0
    p_none_before = 1.0
    for i in range(1, budget + 1):
        # P(first success exactly at trial i) = P(no success in i-1) * good/(n-i+1)
        p_hit = p_none_before * good / (n - i + 1)
        expected += i * p_hit
        p_none_before *= 1 - good / (n - i + 1)
        if p_none_before <= 0:
            break
    expected += (budget + 1) * p_none_before
    return expected


def relative_contextual_order(experience: Experience, features: Features,
                              configs: Sequence[Config], k: int = 3) -> List[Config]:
    """Like contextual_order, but a configuration is encoded relative to the task:
    (layer - old_layer, eta, reg). `features[0] * 2` is the task's old layer (interference benchmarks).
    Scores of the k most similar past tasks are pooled per relative configuration, then mapped back to
    absolute layers for the new task, so a rule such as "write at the old layer" (offset 0) transfers to
    old layers the experience never contained. Configurations with no relative evidence go last."""
    if not experience:
        return list(configs)
    old_new = round(features[0] * 2)
    near = sorted(experience, key=lambda e: _distance(e[0], features))[:k]
    pooled: Dict[Tuple[int, float, float], List[float]] = {}
    for f, scores in near:
        old = round(f[0] * 2)
        for (layer, eta, reg), v in scores.items():
            pooled.setdefault((layer - old, eta, reg), []).append(v)
    def key(c):
        rel = (c[0] - old_new, c[1], c[2])
        vals = pooled.get(rel)
        return -(sum(vals) / len(vals)) if vals else 1.0
    return sorted(configs, key=lambda c: (key(c), list(configs).index(c)))
