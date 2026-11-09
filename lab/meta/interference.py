"""Second testbed: the right intervention depends on the task.

A NanoWSP-3 already stores `n_old` facts (imprinted at `old_layer`). A new batch of `n_new` facts
is then imprinted at a chosen (layer, eta, regularisation). The score of a configuration is

    min(recall of the new facts, retention of the old facts)

so a configuration that writes the new facts by overwriting the old ones scores low. Which
configuration is good depends on where the old memory lives and how big both sets are, which is
what a task-aware proposer has to pick up. In a development run (not part of the evaluation) the
best configuration for old_layer=1 differed from the best for old_layer=5.

Every cell is a real imprint plus real forward passes; nothing is typed in.
Features given to the proposer: (old_layer / 2, log2 n_old, log2 n_new); all are known before the
new experiment is run.
"""
from __future__ import annotations

import json
import math
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab.meta.grid import make_task  # noqa: E402

LAYERS = (1, 3, 5)
ETAS = (0.5, 1.0, 2.0, 3.5, 8.0)
LAMBDAS = (1e-4, 1e-2)
CONFIGS = [(l, e, r) for l, e, r in product(LAYERS, ETAS, LAMBDAS)]
PAPER_DEFAULT = (3, 3.5, 1e-2)
THRESHOLD = 0.8      # fixed on development runs only, before any test task was scored
BUDGET = 10


def score_config(spec, cfg) -> float:
    import torch
    from engine.nanowsp3_runtime import NanoWSP3
    old = make_task(spec["n_old"], 0, spec["seed"])
    new = [(p.replace("MEM", "NEW"), t) for p, t in make_task(spec["n_new"], 0, spec["seed"] + 1)]
    torch.manual_seed(0)
    model = NanoWSP3()
    model.imprint_thematic_corpus([{"name": "old", "facts": old}], target_layer=spec["old_layer"],
                                  eta=3.5, reg_lambda=1e-4)
    layer, eta, lam = cfg
    model.imprint_thematic_corpus([{"name": "new", "facts": new}], target_layer=layer,
                                  eta=eta, reg_lambda=lam)
    recall_new = sum(model.evaluate_recall(p, t)[0] for p, t in new) / len(new)
    keep_old = sum(model.evaluate_recall(p, t)[0] for p, t in old) / len(old)
    return min(recall_new, keep_old)


def task_specs():
    train, test = [], []
    for old_layer in (1, 3, 5):
        for n_old, n_new in ((20, 20), (30, 30), (50, 20)):
            train.append({"name": f"train_L{old_layer}_o{n_old}_n{n_new}", "old_layer": old_layer,
                          "n_old": n_old, "n_new": n_new, "seed": 300 + old_layer * 10 + n_old})
        for n_old, n_new in ((40, 25), (25, 40)):
            test.append({"name": f"test_L{old_layer}_o{n_old}_n{n_new}", "old_layer": old_layer,
                         "n_old": n_old, "n_new": n_new, "seed": 7000 + old_layer * 10 + n_old})
    return train, test


def features(spec):
    return [spec["old_layer"] / 2.0, math.log2(spec["n_old"]), math.log2(spec["n_new"])]


def build(out_path: Path) -> dict:
    table = json.loads(out_path.read_text()) if out_path.exists() else {}
    train, test = task_specs()
    for split, specs in (("train", train), ("test", test)):
        for spec in specs:
            if spec["name"] in table:
                continue
            t0 = time.time()
            scores = {json.dumps(c): score_config(spec, c) for c in CONFIGS}
            table[spec["name"]] = {"split": split, "features": features(spec), **spec, "scores": scores}
            out_path.write_text(json.dumps(table))
            print(f"{spec['name']}: {time.time() - t0:.1f}s", flush=True)
    return table


if __name__ == "__main__":
    build(ROOT / "lab" / "meta" / "interference_scores.json")
