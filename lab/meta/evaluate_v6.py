"""Protocol v6 evaluation (criteria j, k, l pre-registered in interference_v6.py before scoring)."""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab.meta import evaluate as ev  # noqa: E402
from lab.meta import interference_v6 as v6  # noqa: E402
from lab.meta.adaptive_search import fixed_recipe_order, prior_table, search  # noqa: E402
from lab.meta.meta_learner import random_expected_trials, shuffled_experience, trials_to_success  # noqa: E402

TH, BUD = v6.THRESHOLD, v6.BUDGET
m = statistics.fmean


def run(prior, f, configs, s, update, tie=None):
    tr = search(prior, f, configs, lambda c: s[c], BUD, update=update, tie_order=tie)
    ys = [y for _, y in tr]
    first_ok = next((i for i, y in enumerate(ys, 1) if y >= TH), BUD + 1)
    return first_ok, max(ys[:3]), ys[0]


def paired_ci(a, b, reps=2000, seed=0):
    rg = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    ms = sorted(m(rg.choice(d) for _ in d) for _ in range(reps))
    return m(d), ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1]


def compute_rows():
    tasks = ev.load(HERE / "interference_v6_scores.json")
    configs = sorted(next(iter(tasks.values()))["scores"])
    train = [(t["features"], t["scores"]) for t in tasks.values() if t["split"] == "train"]
    test = {n: t for n, t in tasks.items() if t["split"] == "test"}
    assert len(train) == 18 and len(test) == 20 and len(configs) == 60
    rows = {}
    for n, t in test.items():
        s, f = t["scores"], t["features"]
        mu = prior_table(train, f)
        r = {"adaptive": run(mu, f, configs, s, True), "prior_only": run(mu, f, configs, s, False)}
        rec = fixed_recipe_order(f, configs)
        r["fixed_recipe"] = (trials_to_success(rec, s, TH, BUD), max(s[c] for c in rec[:3]), s[rec[0]])
        rg = random.Random(1)
        fl = []
        for _ in range(50):
            tie = list(configs); rg.shuffle(tie)
            fl.append(run(lambda c: 0.0, f, configs, s, True, tie))
        r["flat_adaptive"] = tuple(m(x[i] for x in fl) for i in range(3))
        rg = random.Random(2)
        sh = []
        for _ in range(100):
            sh.append(run(prior_table(shuffled_experience(train, rg), f), f, configs, s, True))
        r["shuffled_adaptive"] = tuple(m(x[i] for x in sh) for i in range(3))
        orc = sorted(configs, key=lambda c: -s[c])
        r["oracle"] = (trials_to_success(orc, s, TH, BUD), max(s[c] for c in orc[:3]), s[orc[0]])
        r["random"] = (random_expected_trials(s, TH, BUD), m(sorted(s.values())[-3:]) * 0 + m(s.values()), m(s.values()))
        r["_fam"], r["_solv"] = f[3] > 5 and "B" or "A", any(v >= TH for v in s.values())
        r["_unseen"] = "_L2_" in n or "_L4_" in n
        rows[n] = r
    return rows


def criteria(rows):
    """Paired bootstrap results of j, k, l: {name: (mean diff, lo, hi)}."""
    allr = list(rows.values())
    return {name: paired_ci([r["adaptive"][0] for r in allr], [r[other][0] for r in allr])
            for name, other in (("j", "flat_adaptive"), ("k", "prior_only"), ("l", "fixed_recipe"))}


def main():
    rows = compute_rows()
    pol = ["random", "fixed_recipe", "prior_only", "flat_adaptive", "shuffled_adaptive", "adaptive", "oracle"]
    print("== Protocol v6 report ==")

    def rep(label, pred):
        sub = [r for r in rows.values() if pred(r)]
        print(f"[{label}] n={len(sub)}  trials (censored at {BUD + 1}) | best score after 3 trials | first-proposal score")
        for p in pol:
            print(f"   {p:18s} {m(r[p][0] for r in sub):6.2f} | {m(r[p][1] for r in sub):.3f} | {m(r[p][2] for r in sub):.3f}")
        return sub
    allr = rep("all held-out", lambda r: True)
    rep("solvable only", lambda r: r["_solv"])
    rep("unseen old layers (2, 4)", lambda r: r["_unseen"])
    rep("family A", lambda r: r["_fam"] == "A")
    rep("family B", lambda r: r["_fam"] == "B")
    for crit, other in (("j transfer (vs flat_adaptive)", "flat_adaptive"), ("k update (vs prior_only)", "prior_only"),
                        ("l beyond recipe (vs fixed_recipe)", "fixed_recipe")):
        d, lo, hi = paired_ci([r["adaptive"][0] for r in allr], [r[other][0] for r in allr])
        print(f"criterion {crit}: mean diff {d:+.2f} trials, 95% CI [{lo:+.2f}, {hi:+.2f}] -> {'holds' if hi < 0 else 'NOT shown'}")
    print(f"unsolved by every policy incl. oracle: {sum(1 for r in rows.values() if not r['_solv'])}/20")


if __name__ == "__main__":
    main()
