"""Pre-registered evaluation of the experience-driven proposer.

Run grid.py first (real NanoWSP-3 runs) to produce lab/meta/grid_scores.json, then:

    python lab/meta/evaluate.py

Pre-registered before looking at test results:
  * success threshold: recall >= 0.95
  * trial budget: 12 configurations out of 36
  * claim "the proposer learned to learn" requires ALL of:
      (a) contextual beats random in mean trials-to-success on held-out tasks,
      (b) contextual beats the shuffled-experience control (same scores, wrong task labels),
      (c) contextual beats the static ranking learned from the same experience.
    If (c) fails, the honest reading is "experience helped, but a context-free default is enough".
Held-out tasks use seeds and fact counts that never appear in the experience.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from lab.meta.meta_learner import (  # noqa: E402
    contextual_order, random_expected_trials, shuffled_experience, static_order,
    trials_to_success,
)

THRESHOLD = 0.95
BUDGET = 12
PAPER_DEFAULT = (3, 3.5, 1e-2)


def load(path: Path):
    import math
    raw = json.loads(path.read_text())
    tasks = {}
    for name, t in raw.items():
        scores = {tuple(json.loads(k)): v for k, v in t["scores"].items()}
        feats = tuple(t["features"]) if "features" in t else (math.log2(t["n"]), float(t["prefix_len"]))
        tasks[name] = {"split": t["split"], "features": feats, "scores": scores}
    return tasks


def first_pick_scores(tasks, rng):
    """Threshold-free view: score of the FIRST configuration each policy proposes on held-out tasks."""
    configs = sorted(next(iter(tasks.values()))["scores"])
    train = [(t["features"], t["scores"]) for t in tasks.values() if t["split"] == "train"]
    test = [t for t in tasks.values() if t["split"] == "test"]
    out = {"random": [], "static": [], "contextual": [], "shuffled": [], "oracle": []}
    for t in test:
        s, f = t["scores"], t["features"]
        out["random"].append(statistics.fmean(s.values()))
        out["static"].append(s[static_order(train, configs)[0]])
        out["contextual"].append(s[contextual_order(train, f, configs)[0]])
        out["shuffled"].append(statistics.fmean(
            s[contextual_order(shuffled_experience(train, rng), f, configs)[0]] for _ in range(50)))
        out["oracle"].append(max(s.values()))
    return out


def mean_ci(xs, rng, reps=2000):
    xs = list(xs)
    boots = sorted(statistics.fmean(rng.choice(xs) for _ in xs) for _ in range(reps))
    return statistics.fmean(xs), boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]


def offline(tasks, rng, paper_default=PAPER_DEFAULT):
    configs = sorted(next(iter(tasks.values()))["scores"])
    train = [(t["features"], t["scores"]) for t in tasks.values() if t["split"] == "train"]
    test = [t for t in tasks.values() if t["split"] == "test"]
    PAPER_DEFAULT_ = paper_default
    res = {k: [] for k in ("random", "paper_default", "static", "contextual", "shuffled", "oracle")}
    for t in test:
        s, f = t["scores"], t["features"]
        res["random"].append(random_expected_trials(s, THRESHOLD, BUDGET))
        res["paper_default"].append(trials_to_success(
            [PAPER_DEFAULT_] + [c for c in configs if c != PAPER_DEFAULT_], s, THRESHOLD, BUDGET))
        res["static"].append(trials_to_success(static_order(train, configs), s, THRESHOLD, BUDGET))
        res["contextual"].append(trials_to_success(contextual_order(train, f, configs), s, THRESHOLD, BUDGET))
        sh = []
        for _ in range(50):
            sh.append(trials_to_success(
                contextual_order(shuffled_experience(train, rng), f, configs), s, THRESHOLD, BUDGET))
        res["shuffled"].append(statistics.fmean(sh))
        res["oracle"].append(trials_to_success(sorted(configs, key=lambda c: -s[c]), s, THRESHOLD, BUDGET))
    return res, len(test)


def online(tasks, rng, orders=200):
    """Experience grows one task at a time (empty start). Mean trials by position in the stream."""
    configs = sorted(next(iter(tasks.values()))["scores"])
    names = list(tasks)
    by_pos = {}
    for _ in range(orders):
        rng.shuffle(names)
        exp = []
        for pos, nm in enumerate(names):
            t = tasks[nm]
            order = contextual_order(exp, t["features"], configs)
            by_pos.setdefault(pos, []).append(
                (trials_to_success(order, t["scores"], THRESHOLD, BUDGET),
                 random_expected_trials(t["scores"], THRESHOLD, BUDGET)))
            exp.append((t["features"], t["scores"]))
    return by_pos


def protocol_v2(tasks):
    """Pre-registered v2 extras: criterion d, solvable-only and unseen-layer reporting."""
    configs = sorted(next(iter(tasks.values()))["scores"])
    train = [(t["features"], t["scores"]) for t in tasks.values() if t["split"] == "train"]
    test = {n: t for n, t in tasks.items() if t["split"] == "test"}
    solvable = {n: t for n, t in test.items() if any(v >= THRESHOLD for v in t["scores"].values())}
    unseen = {n: t for n, t in test.items() if "_L2_" in n or "_L4_" in n}

    def table(ts, label):
        out = {"random": [], "static": [], "contextual": [], "oracle": [], "shuffled": []}
        first = {"random": [], "static": [], "contextual": [], "oracle": [], "shuffled": []}
        outer_t, outer_f = [], []
        for seed in range(20):
            r = random.Random(seed)
            tt, ff = [], []
            for t in ts.values():
                s, f = t["scores"], t["features"]
                sh = [contextual_order(shuffled_experience(train, r), f, configs) for _ in range(50)]
                tt.append(statistics.fmean(trials_to_success(o, s, THRESHOLD, BUDGET) for o in sh))
                ff.append(statistics.fmean(s[o[0]] for o in sh))
            outer_t.append(statistics.fmean(tt))
            outer_f.append(statistics.fmean(ff))
        for t in ts.values():
            s, f = t["scores"], t["features"]
            out["random"].append(random_expected_trials(s, THRESHOLD, BUDGET))
            first["random"].append(statistics.fmean(s.values()))
            so, co = static_order(train, configs), contextual_order(train, f, configs)
            ora = sorted(configs, key=lambda c: -s[c])
            for k, o in (("static", so), ("contextual", co), ("oracle", ora)):
                out[k].append(trials_to_success(o, s, THRESHOLD, BUDGET))
                first[k].append(s[o[0]])
        out["shuffled"] = outer_t
        first["shuffled"] = outer_f
        print(f"[{label}] n={len(ts)}  trials-to-success (censored at {BUDGET + 1} for unsolved) | first-proposal score")
        for k in out:
            print(f"   {k:11s} {statistics.fmean(out[k]):6.2f} | {statistics.fmean(first[k]):.3f}")
        return first

    print("== Protocol v2 report ==")
    first_all = table(test, "all held-out")
    table(solvable, "solvable held-out only")
    first_unseen = table(unseen, "unseen old layers (2, 4)")
    d_all = statistics.fmean(first_all["contextual"]) > max(statistics.fmean(first_all["static"]), statistics.fmean(first_all["shuffled"]))
    d_unseen = statistics.fmean(first_unseen["contextual"]) > max(statistics.fmean(first_unseen["static"]), statistics.fmean(first_unseen["shuffled"]))
    print(f"Criterion d (first-proposal score, contextual > static and > shuffled): all held-out {d_all}, unseen layers {d_unseen}")


def main():
    global THRESHOLD, BUDGET
    bench = sys.argv[1] if len(sys.argv) > 1 else "recall"
    if bench == "interference_v2":
        from lab.meta import interference_v2 as v2
        path, THRESHOLD, BUDGET, default = HERE / "interference_v2_scores.json", v2.THRESHOLD, v2.BUDGET, v2.PAPER_DEFAULT
    elif bench == "interference":
        from lab.meta import interference as itf
        path, THRESHOLD, BUDGET, default = HERE / "interference_scores.json", itf.THRESHOLD, itf.BUDGET, itf.PAPER_DEFAULT
    else:
        path, default = HERE / "grid_scores.json", PAPER_DEFAULT
    print(f"Benchmark: {bench}")
    tasks = load(path)
    rng = random.Random(0)
    res, n_test = offline(tasks, rng, default)
    n_cfg = len(next(iter(tasks.values()))["scores"])
    print(f"Held-out tasks: {n_test}  threshold={THRESHOLD}  budget={BUDGET}/{n_cfg} (censored at {BUDGET + 1})")
    print("Mean trials-to-success on held-out tasks (lower is better), 95% bootstrap CI over tasks:")
    for k, xs in res.items():
        m, lo, hi = mean_ci(xs, rng)
        fail = sum(1 for x in xs if x > BUDGET)
        print(f"  {k:14s} {m:5.2f}  [{lo:5.2f}, {hi:5.2f}]   not solved in budget: {fail}/{len(xs)}")
    verdict = {
        "a_contextual_beats_random": statistics.fmean(res["contextual"]) < statistics.fmean(res["random"]),
        "b_contextual_beats_shuffled": statistics.fmean(res["contextual"]) < statistics.fmean(res["shuffled"]),
        "c_contextual_beats_static": statistics.fmean(res["contextual"]) < statistics.fmean(res["static"]),
    }
    print("Pre-registered criteria:", verdict)
    print("CLAIM 'learned to learn' supported:", all(verdict.values()))

    fp = first_pick_scores(tasks, rng)
    print("Score of the first configuration proposed, mean over held-out tasks (higher is better):")
    for k, xs in fp.items():
        print(f"  {k:14s} {statistics.fmean(xs):.3f}")

    if bench == "interference_v2":
        protocol_v2(tasks)

    by_pos = online(tasks, rng)
    pos = sorted(by_pos)
    third = len(pos) // 3
    def avg(idx, col):
        return statistics.fmean(v[col] for p in idx for v in by_pos[p])
    early, late = pos[:third], pos[-third:]
    print(f"Online stream, {len(pos)} tasks, 200 random orders:")
    print(f"  first third  : proposer {avg(early, 0):.2f} vs random {avg(early, 1):.2f}")
    print(f"  last third   : proposer {avg(late, 0):.2f} vs random {avg(late, 1):.2f}")


if __name__ == "__main__":
    main()
