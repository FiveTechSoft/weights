"""Second method for lab/probes/weight2bench: recomputes the pooled Spearman rho (per target) and the pooled
real-vs-placebo AUC per feature from lab/runs/weight2bench/arms.json with its own rank and pairwise code (no import
from features.py) and compares with summary.json.

  python -I lab/probes/weight2bench/recount.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "lab" / "runs" / "weight2bench"


def rank_avg(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(x, y):
    if len(x) < 3:
        return None
    return pearson(rank_avg(x), rank_avg(y))


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]; neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    tot = 0.0
    for p in pos:
        for q in neg:
            tot += 1.0 if p > q else 0.5 if p == q else 0.0
    return tot / (len(pos) * len(neg))


def close(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(b, float) and math.isnan(b):
        return a is None or (isinstance(a, float) and math.isnan(a))
    return abs(a - b) <= tol


def main():
    rows = json.loads((RUN / "arms.json").read_text(encoding="utf-8"))["rows"]
    summ = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    feats = list(summ["auc"].keys())
    bad = 0; checked = 0
    for t in ("d_gsm", "d_math", "d_tok", "d_cut"):
        for f in feats:
            xy = [(r[f], r[t]) for r in rows if r.get(f) is not None and r.get(t) is not None]
            mine = spearman([x for x, _ in xy], [y for _, y in xy])
            theirs = summ["spearman"][t][f]["pooled"]["rho"]
            checked += 1
            if not close(mine, theirs):
                bad += 1; print("DIFFER spearman", t, f, mine, theirs)
    arows = [r for r in rows if r["auc_row"]]
    for f in feats:
        s = [r[f] for r in arows if r.get(f) is not None]; y = [r["is_real"] for r in arows if r.get(f) is not None]
        mine = auc(s, y); theirs = summ["auc"][f]["pooled"]["auc"]
        checked += 1
        if not close(mine, theirs):
            bad += 1; print("DIFFER auc", f, mine, theirs)
    print(f"recount: {checked} numbers checked, {bad} differ -> {'AGREE' if bad == 0 else 'DIFFER'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
