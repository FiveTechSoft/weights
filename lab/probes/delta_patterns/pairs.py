"""Pairwise comparison of delta signatures (sig_<name>.pt), with the nulls of plan.md.

usage: python lab/probes/delta_patterns/pairs.py <nameA> <nameB> [...more names]   (every pair)

Per tensor type and pair:
  same     median top-16 subspace overlap of the same tensor (same layer)
  null     median overlap of that type at different layers (layer i of A vs layer j != i of B, all pairs)
  chance   k / d
  cos      median cosine of the two deltas (count-sketch; only defined for the same tensor)
  H3       same >= 3 * null and same >= 3 * chance (plan.md)
Writes lab/runs/delta_patterns/pairs_<A>_<B>.json.
"""
import itertools
import json
import re
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import weight_diff as wd  # noqa: E402

OUT = ROOT / "lab" / "runs" / "delta_patterns"


def kind(name):
    return re.sub(r"^model\.layers\.\d+\.", "", name)


def compare(na, nb):
    A = torch.load(OUT / f"sig_{na}.pt")
    B = torch.load(OUT / f"sig_{nb}.pt")
    by_type = {}
    for n in sorted(set(A) & set(B)):
        if "basis" not in A[n]:
            continue
        by_type.setdefault(kind(n), []).append(n)
    rows = {}
    for t, names in sorted(by_type.items()):
        same = [wd.signature_overlap(A[n], B[n]) for n in names]
        null = []
        for n1, n2 in itertools.permutations(names, 2):
            if A[n1]["shape"] == B[n2]["shape"]:
                V1, V2 = A[n1]["basis"].double(), B[n2]["basis"].double()
                kk = min(V1.shape[1], V2.shape[1])
                null.append(float((V1[:, :kk].T @ V2[:, :kk]).pow(2).sum()) / kk)
        s = st.median(x["subspace_overlap"] for x in same)
        chance = same[0]["random_overlap"]
        nl = st.median(null) if null else None
        rows[t] = {"tensors": len(names), "same": s, "null_other_layer": nl, "chance": chance,
                   "x_null": s / nl if nl else None, "x_chance": s / chance,
                   "cosine": st.median(x["cosine"] for x in same),
                   "cosine_min": min(x["cosine"] for x in same), "cosine_max": max(x["cosine"] for x in same),
                   "base_alignment_A": st.median(A[n]["base_alignment"] for n in names),
                   "base_alignment_B": st.median(B[n]["base_alignment"] for n in names),
                   "H3": (None if nl is None else bool(s >= 3 * nl and s >= 3 * chance))}
    (OUT / f"pairs_{na}_{nb}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"== {na} vs {nb}")
    print(f"{'type':32s} {'same':>6s} {'null':>6s} {'xnull':>6s} {'xchance':>7s} {'cos':>7s} {'alignA':>6s} {'alignB':>6s} H3")
    for t, r in sorted(rows.items(), key=lambda kv: -(kv[1]["x_null"] or 0)):
        print(f"{t:32s} {r['same']:6.3f} {r['null_other_layer'] or 0:6.3f} {r['x_null'] or 0:6.1f} "
              f"{r['x_chance']:7.1f} {r['cosine']:+7.4f} {r['base_alignment_A']:6.3f} {r['base_alignment_B']:6.3f} "
              f"{'n/a' if r['H3'] is None else ('yes' if r['H3'] else 'no')}")
    return rows


if __name__ == "__main__":
    names = sys.argv[1:]
    for a, b in itertools.combinations(names, 2):
        compare(a, b)
