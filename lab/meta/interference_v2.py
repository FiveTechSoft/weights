"""Interference benchmark v2: unseen old layers.

Differences from interference.py, all fixed BEFORE scoring any v2 held-out task:
  * configuration grid covers layers 1..5 (50 configurations) so that "write at the old layer"
    exists for every old layer, including 2 and 4;
  * training experience is the same 9 tasks, old layers 1, 3, 5 only (re-scored on the wider grid);
  * held-out tasks: the 6 earlier ones (old layers 1, 3, 5) plus 4 new ones with old layers 2 and 4,
    which never appear in the experience;
  * threshold 0.8 unchanged, budget 15 (30% of the grid, close to the 10/30 of v1).
This is protocol v2, not a re-run of v1: the grid and the budget differ.

Pre-registered for v2 (written before any v2 held-out task was scored):
  a, b, c as in evaluate.py (trials-to-success vs random, shuffled control, static ranking), AND
  d. threshold-free: mean score of the FIRST configuration proposed on held-out tasks must be higher for
     the contextual proposer than for the static ranking and for the shuffled control (20 outer seeds x 50
     shuffles). In v1 this metric was added after seeing results; here it is fixed up front.
Reporting rules: trials-to-success is reported both with censoring (unsolved tasks count budget+1 for every
policy) and on solvable tasks only (some configuration reaches the threshold). Unseen-layer tasks
(old layers 2 and 4) are also reported on their own.
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

LAYERS = (1, 2, 3, 4, 5)
CONFIGS = [(l, e, r) for l, e, r in product(LAYERS, v1.ETAS, v1.LAMBDAS)]
PAPER_DEFAULT = v1.PAPER_DEFAULT
THRESHOLD = v1.THRESHOLD
BUDGET = 15


def task_specs():
    train, test = v1.task_specs()
    for old_layer in (2, 4):
        for n_old, n_new in ((40, 25), (25, 40)):
            test.append({"name": f"test_L{old_layer}_o{n_old}_n{n_new}", "old_layer": old_layer,
                         "n_old": n_old, "n_new": n_new, "seed": 7000 + old_layer * 10 + n_old})
    return train, test


def build(out_path: Path) -> dict:
    table = json.loads(out_path.read_text()) if out_path.exists() else {}
    train, test = task_specs()
    for split, specs in (("train", train), ("test", test)):
        for spec in specs:
            if spec["name"] in table:
                continue
            t0 = time.time()
            scores = {json.dumps(c): v1.score_config(spec, c) for c in CONFIGS}
            table[spec["name"]] = {"split": split, "features": v1.features(spec), **spec, "scores": scores}
            out_path.write_text(json.dumps(table))
            print(f"{spec['name']}: {time.time() - t0:.1f}s", flush=True)
    return table


if __name__ == "__main__":
    build(ROOT / "lab" / "meta" / "interference_v2_scores.json")
