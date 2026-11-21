"""Protocol v5: two task families, so that the best designed encoding differs between families.

Family A: score = min(recall of new facts, retention of old facts)   (as v1-v4; best designed encoding on
          training tasks: rel_old)
Family B: score = recall of the new facts only, no retention term     (old memory irrelevant to the score)
Task features: (old_layer/2, log2 n_old, log2 n_new, 10 * family).

Development note (training tasks only, disclosed): on the 9 family B training tasks, layer 1 with eta 1 to 8 gives
recall ~1.0 regardless of the old layer, so the best config does not follow the old layer; many configs (7 to 17 of
50) pass 0.8. That is why family B is expected to prefer a non-relative encoding. No held-out task was scored when
this was written.

Fixed before any v5 held-out task is scored (v2, v3, v4 held-out tasks are not reused):
  * experience: 9 family A training tasks (old layers 1, 3, 5) + 9 family B training tasks, 50 configs each;
  * held-out: 10 family A + 10 family B new tasks (old layers 1-5 x sizes (32, 24), (44, 28); seeds 9500 + / 9600 +);
    layers 2 and 4 are unseen; threshold 0.8, budget 15, 20 outer seeds x 50 shuffles;
  * candidate encodings (designed by us): absolute, rel_old, rel_size (lab/meta/select_encoding.py);
  * SELECTED: for each held-out task, select the encoding by leave-one-old-layer-out on the experience of ITS family
    (the family is a task feature, like the others), then propose with the nearest 3 tasks of that family;
  * controls, all using the full pooled experience and k=3 nearest tasks (family is a feature so neighbours are
    in-family): FIXED absolute, FIXED rel_old, FIXED rel_size. BEST FIXED = the fixed control with the highest
    mean first-proposal score on the held-out set (chosen in hindsight, i.e. generous to the control);
    HINDSIGHT-PER-FAMILY = the best fixed encoding per family in hindsight (upper bound for any selector among
    the candidates); random, static (absolute, pooled), shuffled-experience SELECTED (wrong task labels), oracle.
  * criteria (first-proposal score primary; trials-to-success, censored at 16, secondary):
      g. SELECTED beats BEST FIXED by more than 0.02 first-proposal score AND does not need more mean trials,
         on all 20 held-out tasks and on the unseen-layer subset. If not: "no additional utility over the best
         designed fixed encoding".
      h. SELECTED beats shuffled SELECTED and static on all held-out.
      i. Report per family which encoding is selected and whether it equals the hindsight best for that family.
    Reporting: censored trials, solvable-only, unseen-layer subset, per family, no task dropped. Both families are
    designed by us; passing g would show selection among designed encodings across designed families, not
    invention of encodings.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab.meta import interference as v1  # noqa: E402
from lab.meta import interference_v2 as v2  # noqa: E402
from lab.meta.grid import make_task  # noqa: E402

CONFIGS = v2.CONFIGS
THRESHOLD = v2.THRESHOLD
BUDGET = v2.BUDGET


def score_config(spec, cfg) -> float:
    if spec["family"] == "A":
        return v1.score_config(spec, cfg)
    import torch
    from engine.nanowsp3_runtime import NanoWSP3
    old = make_task(spec["n_old"], 0, spec["seed"])
    new = [(p.replace("MEM", "NEW"), t) for p, t in make_task(spec["n_new"], 0, spec["seed"] + 1)]
    torch.manual_seed(0)
    model = NanoWSP3()
    model.imprint_thematic_corpus([{"name": "old", "facts": old}], target_layer=spec["old_layer"], eta=3.5, reg_lambda=1e-4)
    layer, eta, lam = cfg
    model.imprint_thematic_corpus([{"name": "new", "facts": new}], target_layer=layer, eta=eta, reg_lambda=lam)
    return sum(model.evaluate_recall(p, t)[0] for p, t in new) / len(new)


def features(spec):
    return v1.features(spec) + [10.0 * (spec["family"] == "B")]


def task_specs():
    trainB = [{"name": f"trainB_L{L}_o{o}_n{n}", "family": "B", "old_layer": L, "n_old": o, "n_new": n,
               "seed": 400 + L * 10 + o} for L in (1, 3, 5) for o, n in ((20, 20), (30, 30), (50, 20))]
    testA, testB = [], []
    for L in (1, 2, 3, 4, 5):
        for o, n in ((32, 24), (44, 28)):
            testA.append({"name": f"test5A_L{L}_o{o}_n{n}", "family": "A", "old_layer": L, "n_old": o,
                          "n_new": n, "seed": 9500 + L * 10 + o})
            testB.append({"name": f"test5B_L{L}_o{o}_n{n}", "family": "B", "old_layer": L, "n_old": o,
                          "n_new": n, "seed": 9600 + L * 10 + o})
    return trainB, testA, testB


def build(out_path: Path, which: str) -> dict:
    v2t = json.loads((ROOT / "lab" / "meta" / "interference_v2_scores.json").read_text())
    table = {k: {**v, "family": "A", "features": v["features"] + [0.0]} for k, v in v2t.items() if v["split"] == "train"}
    if out_path.exists():
        table.update(json.loads(out_path.read_text()))
    trainB, testA, testB = task_specs()
    todo = {"train": [(s, "train") for s in trainB], "test": [(s, "test") for s in testA + testB]}[which]
    for spec, split in todo:
        if spec["name"] in table:
            continue
        t0 = time.time()
        scores = {json.dumps(c): score_config(spec, c) for c in CONFIGS}
        table[spec["name"]] = {"split": split, "features": features(spec), **spec, "scores": scores}
        out_path.write_text(json.dumps(table))
        print(f"{spec['name']}: {time.time() - t0:.1f}s", flush=True)
    return table


if __name__ == "__main__":
    build(ROOT / "lab" / "meta" / "interference_v5_scores.json", sys.argv[1])
