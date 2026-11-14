"""Pairwise comparison of delta signatures (sig_<name>.pt), with the nulls of plan.md.

usage: python lab/probes/delta_patterns/pairs.py <nameA> <nameB> [...more names]   (every pair)
The computation is engine.weight_diff.compare_signatures (also the WSPLang op delta_compare).
Writes lab/runs/delta_patterns/pairs_<A>_<B>.json.
"""
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import weight_diff as wd  # noqa: E402

OUT = ROOT / "lab" / "runs" / "delta_patterns"


def compare(na, nb):
    rows = wd.compare_signatures(torch.load(OUT / f"sig_{na}.pt"), torch.load(OUT / f"sig_{nb}.pt"))
    (OUT / f"pairs_{na}_{nb}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"== {na} vs {nb}")
    print(f"{'type':32s} {'same':>6s} {'null':>6s} {'xnull':>6s} {'xchance':>7s} {'cos':>7s} {'alignA':>6s} {'alignB':>6s} H3")
    for t, r in sorted(rows.items(), key=lambda kv: -(kv[1]["x_null"] or 0)):
        print(f"{t:32s} {r['same']:6.3f} {r['null_other_layer'] or 0:6.3f} {r['x_null'] or 0:6.1f} "
              f"{r['x_chance']:7.1f} {r['cosine']:+7.4f} {r['base_alignment_A']:6.3f} {r['base_alignment_B']:6.3f} "
              f"{'n/a' if r['H3'] is None else ('yes' if r['H3'] else 'no')}")
    return rows


if __name__ == "__main__":
    for a, b in itertools.combinations(sys.argv[1:], 2):
        compare(a, b)
