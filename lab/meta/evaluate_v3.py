"""Protocol v3 evaluation (criteria d, e, f pre-registered in interference_v3.py before scoring).

    python lab/meta/evaluate_v3.py
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab.meta import evaluate as ev  # noqa: E402
from lab.meta import interference_v3 as v3  # noqa: E402
from lab.meta.meta_learner import (  # noqa: E402
    contextual_order, random_expected_trials, relative_contextual_order, shuffled_experience,
    static_order, trials_to_success,
)

TH, BUD = v3.THRESHOLD, v3.BUDGET
POLICIES = ("random", "static", "abs_contextual", "rel_contextual", "rel_shuffled", "rel_static", "fixed_rule", "oracle")


def orders(train, f, configs, r):
    old = round(f[0] * 2)
    fixed = sorted(configs, key=lambda c: (c[0] != old, configs.index(c)))
    return {
        "static": static_order(train, configs),
        "abs_contextual": contextual_order(train, f, configs),
        "rel_contextual": relative_contextual_order(train, f, configs),
        "rel_static": relative_contextual_order(train, f, configs, k=len(train)),
        "fixed_rule": fixed,
    }, old


def table(ts, train, configs, label):
    out = {p: [] for p in POLICIES}
    first = {p: [] for p in POLICIES}
    sh_t, sh_f = [], []
    for seed in range(20):
        r = random.Random(seed)
        tt, ff = [], []
        for t in ts.values():
            s, f = t["scores"], t["features"]
            os_ = [relative_contextual_order(shuffled_experience(train, r), f, configs) for _ in range(50)]
            tt.append(statistics.fmean(trials_to_success(o, s, TH, BUD) for o in os_))
            ff.append(statistics.fmean(s[o[0]] for o in os_))
        sh_t.append(statistics.fmean(tt)); sh_f.append(statistics.fmean(ff))
    for t in ts.values():
        s, f = t["scores"], t["features"]
        out["random"].append(random_expected_trials(s, TH, BUD)); first["random"].append(statistics.fmean(s.values()))
        o, _ = orders(train, f, configs, None)
        o["oracle"] = sorted(configs, key=lambda c: -s[c])
        for k, order in o.items():
            out[k].append(trials_to_success(order, s, TH, BUD)); first[k].append(s[order[0]])
    out["rel_shuffled"], first["rel_shuffled"] = sh_t, sh_f
    print(f"[{label}] n={len(ts)}  trials (censored at {BUD + 1}) | first-proposal score")
    for k in POLICIES:
        print(f"   {k:15s} {statistics.fmean(out[k]):6.2f} | {statistics.fmean(first[k]):.3f}")
    return out, first


def main():
    tasks = ev.load(HERE / "interference_v3_scores.json")
    configs = sorted(next(iter(tasks.values()))["scores"])
    train = [(t["features"], t["scores"]) for t in tasks.values() if t["split"] == "train"]
    test = {n: t for n, t in tasks.items() if t["split"] == "test"}
    assert len(test) == 10 and all(round(t["features"][0] * 2) in (1, 2, 3, 4, 5) for t in test.values())
    solvable = {n: t for n, t in test.items() if any(v >= TH for v in t["scores"].values())}
    unseen = {n: t for n, t in test.items() if "_L2_" in n or "_L4_" in n}
    print("== Protocol v3 report ==")
    res = {}
    for ts, lab in ((test, "all held-out"), (solvable, "solvable only"), (unseen, "unseen old layers (2, 4)")):
        res[lab] = table(ts, train, configs, lab)
    m = statistics.fmean
    for lab in ("all held-out", "unseen old layers (2, 4)"):
        t, f = res[lab]
        d = m(f["rel_contextual"]) > max(m(f["static"]), m(f["rel_shuffled"]))
        e = m(f["rel_contextual"]) > m(f["abs_contextual"])
        fr = m(f["rel_contextual"]) > m(f["fixed_rule"]) and m(t["rel_contextual"]) < m(t["fixed_rule"])
        print(f"{lab}: d={d}  e={e}  f(beats fixed rule on first score AND trials)={fr}")
    print(f"unsolved by every policy incl. oracle: {sum(1 for t in test.values() if t['scores'] and max(t['scores'].values()) < TH)}/10")


if __name__ == "__main__":
    main()
