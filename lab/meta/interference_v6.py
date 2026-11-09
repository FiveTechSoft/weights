"""Protocol v6: within-task adaptation (prior from past tasks + updates from the new task's own trials).

Fixed before any v6 task is scored (training scores are experience, held-out scores are evaluation):
  * config space: layers 1-5 x 12 etas geometric from 0.5 to 8.0 (ratio 16^(1/11)) x reg 1e-4 = 60 configs;
  * two families as in v5 (A: min(recall new, retention old); B: recall new only); features
    (old_layer/2, log2 n_old, log2 n_new, 10 * family); relative layer encoding is OUR design;
  * experience: 9 family A + 9 family B training tasks (old layers 1, 3, 5), new seeds 500+ / 600+, re-scored on
    the new grid;
  * held-out: 10 family A + 10 family B NEW tasks (old layers 1-5 x sizes (36, 22), (48, 26), seeds 9700+ / 9800+);
    layers 2 and 4 are unseen. Threshold 0.8, budget 15 trials, censored at 16;
  * searcher (lab/meta/adaptive_search.py, hyper-parameters fixed there, not tuned): prior mean from the 3 nearest
    experience tasks in relative encoding; kernel update from the new task's observed scores;
  * policies: ADAPTIVE (prior + update), PRIOR-ONLY (same prior, never updated: transfer without updating),
    FLAT-ADAPTIVE (prior 0, update, random tie order, mean of 50 permutations: updating without transfer),
    FIXED RECIPE (write at the old layer, eta near 1.0, neighbours by |offset| + |log2 eta|: no experience, no
    observations; it is what v1-v5 learned), SHUFFLED-ADAPTIVE (prior from experience with wrong task labels,
    100 shuffles), random, oracle;
  * criteria, all on mean trials-to-success over ALL 20 held-out tasks, paired bootstrap over tasks
    (2000 resamples, seed 0, 95% interval of the difference ADAPTIVE minus X must lie below 0):
      j. transfer of the prior: ADAPTIVE beats FLAT-ADAPTIVE;
      k. use of the new task's observations: ADAPTIVE beats PRIOR-ONLY;
      l. beyond the learned constant: ADAPTIVE beats FIXED RECIPE.
    Claim "within-task adaptation helps beyond the constant recipe" requires k and l. Claim "the prior transfers"
    requires j. A null result on any of them is an expected, acceptable outcome and will be reported as such
    (the knob is nearly one-dimensional, so a flat search or the recipe may already be near optimal).
  * also reported, not criteria: best score after 3 trials, first-proposal score, solvable-only, unseen layers,
    per family; no task dropped. v2 to v5 held-out tasks are not reused.
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab.meta import interference as v1  # noqa: E402
from lab.meta import interference_v5 as v5  # noqa: E402

ETAS = [round(0.5 * 16 ** (i / 11), 4) for i in range(12)]
CONFIGS = [(l, e, 1e-4) for l, e in product((1, 2, 3, 4, 5), ETAS)]
THRESHOLD = 0.8
BUDGET = 15


def task_specs():
    train, test = [], []
    for fam, base_tr, base_te in (("A", 500, 9700), ("B", 600, 9800)):
        for L in (1, 3, 5):
            for o, n in ((20, 20), (30, 30), (50, 20)):
                train.append({"name": f"train6{fam}_L{L}_o{o}_n{n}", "family": fam, "old_layer": L, "n_old": o,
                              "n_new": n, "seed": base_tr + L * 10 + o})
        for L in (1, 2, 3, 4, 5):
            for o, n in ((36, 22), (48, 26)):
                test.append({"name": f"test6{fam}_L{L}_o{o}_n{n}", "family": fam, "old_layer": L, "n_old": o,
                             "n_new": n, "seed": base_te + L * 10 + o})
    return train, test


def build(out_path: Path, which: str) -> dict:
    table = json.loads(out_path.read_text()) if out_path.exists() else {}
    train, test = task_specs()
    if which == "test":
        from lab.meta.prereg_guard import assert_preregistered
        print("pre-registration commit:", assert_preregistered(Path(__file__)))
    todo = [(s, "train") for s in train] if which == "train" else [(s, "test") for s in test]
    for spec, split in todo:
        if spec["name"] in table:
            continue
        t0 = time.time()
        scores = {json.dumps(c): v5.score_config(spec, c) for c in CONFIGS}
        table[spec["name"]] = {"split": split, "features": v5.features(spec), **spec, "scores": scores}
        out_path.write_text(json.dumps(table))
        print(f"{spec['name']}: {time.time() - t0:.1f}s", flush=True)
    return table


if __name__ == "__main__":
    build(ROOT / "lab" / "meta" / "interference_v6_scores.json", sys.argv[1])
