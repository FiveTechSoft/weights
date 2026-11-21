"""Protocol v4 evaluation (criteria g, h pre-registered in interference_v4.py before scoring)."""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab.meta import evaluate as ev  # noqa: E402
from lab.meta import interference_v4 as v4  # noqa: E402
from lab.meta.meta_learner import (  # noqa: E402
    random_expected_trials, shuffled_experience, static_order, trials_to_success,
)
from lab.meta.select_encoding import ENCODINGS, order_with, select  # noqa: E402

TH, BUD = v4.THRESHOLD, v4.BUDGET
POL = ("random", "static", "selected", "shuffled_selected", "ctrl_A_rel_static", "ctrl_B_rel_ctx", "ctrl_C_abs_ctx", "oracle")


def table(ts, train, configs, label):
    out = {p: [] for p in POL}; first = {p: [] for p in POL}
    sh_t, sh_f = [], []
    for seed in range(20):
        r = random.Random(seed); tt, ff = [], []
        for t in ts.values():
            s, f = t["scores"], t["features"]
            os_ = []
            for _ in range(50):
                e = shuffled_experience(train, r)
                os_.append(order_with(ENCODINGS[select(e, configs)[0]], e, f, configs))
            tt.append(statistics.fmean(trials_to_success(o, s, TH, BUD) for o in os_))
            ff.append(statistics.fmean(s[o[0]] for o in os_))
        sh_t.append(statistics.fmean(tt)); sh_f.append(statistics.fmean(ff))
    name, _ = select(train, configs)
    for t in ts.values():
        s, f = t["scores"], t["features"]
        out["random"].append(random_expected_trials(s, TH, BUD)); first["random"].append(statistics.fmean(s.values()))
        o = {"static": static_order(train, configs),
             "selected": order_with(ENCODINGS[name], train, f, configs),
             "ctrl_A_rel_static": order_with(ENCODINGS["rel_old"], train, f, configs, k=len(train)),
             "ctrl_B_rel_ctx": order_with(ENCODINGS["rel_old"], train, f, configs),
             "ctrl_C_abs_ctx": order_with(ENCODINGS["absolute"], train, f, configs),
             "oracle": sorted(configs, key=lambda c: -s[c])}
        for k, order in o.items():
            out[k].append(trials_to_success(order, s, TH, BUD)); first[k].append(s[order[0]])
    out["shuffled_selected"], first["shuffled_selected"] = sh_t, sh_f
    print(f"[{label}] n={len(ts)}  trials (censored at {BUD + 1}) | first-proposal score")
    for k in POL:
        print(f"   {k:18s} {statistics.fmean(out[k]):6.2f} | {statistics.fmean(first[k]):.3f}")
    return out, first


def main():
    tasks = ev.load(HERE / "interference_v4_scores.json")
    configs = sorted(next(iter(tasks.values()))["scores"])
    train = [(t["features"], t["scores"]) for t in tasks.values() if t["split"] == "train"]
    test = {n: t for n, t in tasks.items() if t["split"] == "test"}
    assert len(test) == 10
    name, val = select(train, configs)
    print("== Protocol v4 report ==")
    print("Leave-one-old-layer-out validation (first-proposal score):", {k: round(v, 3) for k, v in val.items()})
    print("Selected encoding:", name)
    sol = {n: t for n, t in test.items() if any(v >= TH for v in t["scores"].values())}
    uns = {n: t for n, t in test.items() if "_L2_" in n or "_L4_" in n}
    m = statistics.fmean
    for ts, lab in ((test, "all held-out"), (sol, "solvable only"), (uns, "unseen old layers (2, 4)")):
        t, f = table(ts, train, configs, lab)
        if lab != "solvable only":
            g = (m(f["selected"]) - m(f["ctrl_A_rel_static"]) > 0.02) and m(t["selected"]) < m(t["ctrl_A_rel_static"])
            print(f"   criterion g (selected beats control A by >0.02 first score AND fewer trials) on {lab}: {g}")
    print(f"unsolved by every policy incl. oracle: {sum(1 for t in test.values() if max(t['scores'].values()) < TH)}/10")


if __name__ == "__main__":
    main()
