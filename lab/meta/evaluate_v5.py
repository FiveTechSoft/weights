"""Protocol v5 evaluation (criteria g, h, i pre-registered in interference_v5.py before scoring)."""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab.meta import evaluate as ev  # noqa: E402
from lab.meta import interference_v5 as v5  # noqa: E402
from lab.meta.meta_learner import random_expected_trials, shuffled_experience, static_order, trials_to_success  # noqa: E402
from lab.meta.select_encoding import ENCODINGS, order_with, select  # noqa: E402

TH, BUD = v5.THRESHOLD, v5.BUDGET
m = statistics.fmean


def fam(f):
    return "B" if f[3] > 5 else "A"


def selected_order(exp, f, configs):
    e = [x for x in exp if fam(x[0]) == fam(f)]
    name, _ = select(e, configs)
    return order_with(ENCODINGS[name], e, f, configs), name


def main():
    tasks = ev.load(HERE / "interference_v5_scores.json")
    configs = sorted(next(iter(tasks.values()))["scores"])
    train = [(t["features"], t["scores"]) for t in tasks.values() if t["split"] == "train"]
    test = {n: t for n, t in tasks.items() if t["split"] == "test"}
    assert len(test) == 20 and len(train) == 18
    rows = {}
    for n, t in test.items():
        s, f = t["scores"], t["features"]
        sel, name = selected_order(train, f, configs)
        o = {"static": static_order(train, configs), "selected": sel,
             "fixed_absolute": order_with(ENCODINGS["absolute"], train, f, configs),
             "fixed_rel_old": order_with(ENCODINGS["rel_old"], train, f, configs),
             "fixed_rel_size": order_with(ENCODINGS["rel_size"], train, f, configs),
             "oracle": sorted(configs, key=lambda c: -s[c])}
        r = {"random": (random_expected_trials(s, TH, BUD), m(s.values()))}
        for k, order in o.items():
            r[k] = (trials_to_success(order, s, TH, BUD), s[order[0]])
        r["_name"] = name
        r["_fam"] = fam(f)
        r["_unseen"] = "_L2_" in n or "_L4_" in n
        r["_solv"] = any(v >= TH for v in s.values())
        rows[n] = r
    # shuffled SELECTED
    sh = {n: ([], []) for n in test}
    for seed in range(20):
        rg = random.Random(seed)
        acc = {n: ([], []) for n in test}
        for _ in range(50):
            e = shuffled_experience(train, rg)
            for n, t in test.items():
                o, _ = selected_order(e, t["features"], configs)
                acc[n][0].append(trials_to_success(o, t["scores"], TH, BUD)); acc[n][1].append(t["scores"][o[0]])
        for n in test:
            sh[n][0].append(m(acc[n][0])); sh[n][1].append(m(acc[n][1]))
    for n in test:
        rows[n]["shuffled_selected"] = (m(sh[n][0]), m(sh[n][1]))
    enc = ["fixed_absolute", "fixed_rel_old", "fixed_rel_size"]
    pol = ["random", "static", "selected", "shuffled_selected"] + enc + ["oracle"]
    print("== Protocol v5 report ==")
    print("Selected encoding per family:", {f: sorted({r['_name'] for r in rows.values() if r['_fam'] == f}) for f in "AB"})

    def report(label, pred):
        sub = {n: r for n, r in rows.items() if pred(r)}
        print(f"[{label}] n={len(sub)}  trials (censored at {BUD + 1}) | first-proposal score")
        res = {}
        for p in pol:
            res[p] = (m(r[p][0] for r in sub.values()), m(r[p][1] for r in sub.values()))
            print(f"   {p:18s} {res[p][0]:6.2f} | {res[p][1]:.3f}")
        return res
    allr = report("all held-out", lambda r: True)
    sol = report("solvable only", lambda r: r["_solv"])
    uns = report("unseen old layers (2, 4)", lambda r: r["_unseen"])
    fa = report("family A", lambda r: r["_fam"] == "A")
    fb = report("family B", lambda r: r["_fam"] == "B")
    best = max(enc, key=lambda k: allr[k][1])
    hind = {}
    for f_, res in (("A", fa), ("B", fb)):
        hind[f_] = max(enc, key=lambda k: res[k][1])
    print(f"BEST FIXED (hindsight, all held-out): {best};  hindsight-per-family best: {hind}")
    for lab, res in (("all held-out", allr), ("unseen layers", uns)):
        g = res["selected"][1] - res[best][1] > 0.02 and res["selected"][0] <= res[best][0]
        print(f"criterion g on {lab}: {g}   (selected {res['selected'][1]:.3f}/{res['selected'][0]:.2f} vs best fixed {res[best][1]:.3f}/{res[best][0]:.2f})")
    h = allr["selected"][1] > max(allr["shuffled_selected"][1], allr["static"][1])
    print(f"criterion h (selected > shuffled and > static, first score): {h}")
    for f_ in "AB":
        picked = sorted({r["_name"] for r in rows.values() if r["_fam"] == f_})
        print(f"criterion i family {f_}: selected {picked}, hindsight best {hind[f_]}")
    print(f"unsolved by every policy incl. oracle: {sum(1 for r in rows.values() if not r['_solv'])}/20")


if __name__ == "__main__":
    main()
