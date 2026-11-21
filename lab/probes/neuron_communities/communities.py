"""Neuron communities from per-position attributions (phase A of PLAN.md): the co-attribution graph, Louvain
communities with a size cap, the always-on set, the candidate community, and the plan's reading rule. Pure
numpy / networkx; a neuron is a global id = layer * width + index (see global_id / split_id)."""
import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


def global_id(layer: int, j: int, width: int) -> int:
    return int(layer) * int(width) + int(j)


def split_id(g: int, width: int) -> Tuple[int, int]:
    return int(g) // int(width), int(g) % int(width)


def merge_topk(per_layer: Dict[int, Tuple[np.ndarray, np.ndarray]], k: int, width: int):
    """per_layer: layer -> (indices, values) of that layer's top candidates at one position. The global top-k by
    |value| across layers is exact when every layer kept at least k candidates. -> (global ids, values)."""
    gids, vals = [], []
    for layer, (idx, val) in per_layer.items():
        for j, v in zip(np.asarray(idx).tolist(), np.asarray(val).tolist()):
            gids.append(global_id(layer, j, width))
            vals.append(float(v))
    if not gids:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    order = np.argsort(-np.abs(np.asarray(vals)), kind="stable")[:k]
    return np.asarray(gids, dtype=np.int64)[order], np.asarray(vals, dtype=np.float64)[order]


def pmi_graph(selections: Sequence[Iterable[int]], min_co: int = 5) -> nx.Graph:
    """Nodes: neurons selected at least once (attribute count). Edge i-j when co-selected >= min_co times and
    PMI = log(c_ij * P / (c_i * c_j)) > 0, weight = PMI (P = number of positions)."""
    P = len(selections)
    count: Counter = Counter()
    co: Counter = Counter()
    for sel in selections:
        s = sorted(set(int(x) for x in sel))
        count.update(s)
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                co[(s[a], s[b])] += 1
    G = nx.Graph()
    for n, c in count.items():
        G.add_node(n, count=int(c))
    for (i, j), c in co.items():
        if c < min_co:
            continue
        pmi = math.log(c * P / (count[i] * count[j]))
        if pmi > 0:
            G.add_edge(i, j, weight=pmi, co=int(c))
    return G


def louvain(G: nx.Graph, seed: int = 2026, cap: int = 128) -> List[Set[int]]:
    """Louvain communities on the weighted graph (isolated nodes dropped), each capped at `cap` neurons by weighted
    degree inside the community; sorted by size (desc) then by smallest id for a stable order."""
    H = G.subgraph([n for n, d in G.degree() if d > 0]).copy()
    if H.number_of_nodes() == 0:
        return []
    comms = nx.community.louvain_communities(H, weight="weight", seed=seed)
    out = []
    for c in comms:
        c = set(c)
        if len(c) > cap:
            sub = H.subgraph(c)
            deg = dict(sub.degree(weight="weight"))
            c = set(sorted(c, key=lambda n: (-deg[n], n))[:cap])
        out.append(c)
    out.sort(key=lambda c: (-len(c), min(c)))
    return out


def always_on(selections: Sequence[Iterable[int]], frac: float = 0.9) -> Set[int]:
    """Neurons selected at >= frac of all positions."""
    P = len(selections)
    count: Counter = Counter()
    for sel in selections:
        count.update(set(int(x) for x in sel))
    return {n for n, c in count.items() if c >= frac * P - 1e-9}


def candidate(communities: Sequence[Set[int]], mass_pos: Dict[int, float], mass_neg: Dict[int, float]):
    """The community with the largest (sum of |attr| at pos close-event positions) - (same at neg positions).
    -> (best set, scores sorted desc: [{size, pos, neg, score, neurons}])."""
    scores = []
    for c in communities:
        p = float(sum(mass_pos.get(n, 0.0) for n in c))
        q = float(sum(mass_neg.get(n, 0.0) for n in c))
        scores.append({"size": len(c), "pos": p, "neg": q, "score": p - q, "neurons": sorted(c)})
    scores.sort(key=lambda s: (-s["score"], s["size"]))
    best = set(scores[0]["neurons"]) if scores else set()
    return best, scores


def a1_pass(a1: Dict[str, float]) -> bool:
    """A1 necessity: drop(C) >= 25 % and >= 5x the placebo median, placebo median <= 10 %."""
    d, p = float(a1["drop"]), float(a1["placebo"])
    return d >= 0.25 and d >= 5 * p and p <= 0.10


def a2_pass(a2: Optional[Dict[str, float]]) -> bool:
    """A2 sufficiency: median retained >= 50 % with C + always-on only; the random set of |C| retains <= 25 %."""
    if not a2:
        return False
    return float(a2["retained"]) >= 0.5 and float(a2["random"]) <= 0.25


def verdict(a1: Dict[str, float], a2: Optional[Dict[str, float]]) -> str:
    """PLAN reading rule: community_found (A1 and A2), necessary_not_sufficient (A1 only), nothing_sparse."""
    if not a1_pass(a1):
        return "nothing_sparse"
    return "community_found" if a2_pass(a2) else "necessary_not_sufficient"
