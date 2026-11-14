"""Validation-based choice among DESIGNED candidate encodings of a configuration.

Candidates are written by us, so choosing among them is model selection, not invention of a representation.
An encoding maps (config, task features) to a key; experience from the k nearest tasks is pooled per key.
Selection: leave-one-old-layer-out inside the experience; the encoding with the best mean first-proposal
score on the left-out tasks wins. Ties go to the earlier candidate in ENCODINGS order.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

from lab.meta.meta_learner import Config, Experience, Features, _distance

Encoding = Callable[[Config, Features], tuple]


def _old(f):  # old layer
    return round(f[0] * 2)


ENCODINGS: Dict[str, Encoding] = {
    "absolute": lambda c, f: (c[0], c[1], c[2]),
    "rel_old": lambda c, f: (c[0] - _old(f), c[1], c[2]),
    "rel_size": lambda c, f: (c[0] - round(f[2]), c[1], c[2]),   # distractor: offset to log2(n_new)
}


def order_with(enc: Encoding, experience: Experience, f: Features, configs: Sequence[Config], k: int = 3):
    if not experience:
        return list(configs)
    near = sorted(experience, key=lambda e: _distance(e[0], f))[:k]
    pooled: Dict[tuple, List[float]] = {}
    for g, scores in near:
        for c, v in scores.items():
            pooled.setdefault(enc(c, g), []).append(v)
    def key(c):
        vals = pooled.get(enc(c, f))
        return -(sum(vals) / len(vals)) if vals else 1.0
    return sorted(configs, key=lambda c: (key(c), list(configs).index(c)))


def validate(experience: Experience, configs: Sequence[Config], k: int = 3) -> Dict[str, float]:
    """Leave-one-old-layer-out mean first-proposal score per encoding."""
    layers = sorted({_old(f) for f, _ in experience})
    out = {}
    for name, enc in ENCODINGS.items():
        vals = []
        for L in layers:
            rest = [e for e in experience if _old(e[0]) != L]
            for f, s in (e for e in experience if _old(e[0]) == L):
                vals.append(s[order_with(enc, rest, f, configs, k)[0]])
        out[name] = sum(vals) / len(vals)
    return out


def select(experience: Experience, configs: Sequence[Config], k: int = 3) -> Tuple[str, Dict[str, float]]:
    v = validate(experience, configs, k)
    best = max(v, key=lambda n: (v[n], -list(ENCODINGS).index(n)))
    return best, v
