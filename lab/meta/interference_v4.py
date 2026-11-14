"""Protocol v4: automatic selection among designed encodings, fresh held-out tasks (new seeds and sizes).

Fixed before any v4 held-out task is scored. v2 and v3 held-out tasks are not reused as confirmation.
  * experience: the 9 training tasks of v1/v2/v3 (old layers 1, 3, 5), 50 configurations;
  * held-out: 10 new tasks, old layers 1 to 5 x sizes (28, 26) and (50, 22), seeds 9100 + ...;
    old layers 2 and 4 are unseen; threshold 0.8, budget 15, 20 outer seeds x 50 shuffles;
  * candidate encodings (designed by us): absolute, rel_old (layer - old layer), rel_size (distractor);
    selection by leave-one-old-layer-out on the experience (lab/meta/select_encoding.py), ties to the earlier;
  * policies: random, static, SELECTED (task-conditioned, k=3, with the chosen encoding), shuffled-experience
    SELECTED (selection and proposer re-run on wrong task labels), and the CONTROLS:
      A. rel_old pooled over all experience, no task similarity (best designed fixed encoding, learned static
         ranking);
      B. rel_old task-conditioned (k=3);
      C. absolute task-conditioned (v2 proposer);
      oracle.
    The weak fixed rule of v3 (offset 0, ties in grid order) is NOT used to support any claim here.
  * criteria (first-proposal score and trials-to-success, all held-out and unseen layers):
      g. SELECTED beats control A by more than 0.02 first-proposal score AND has fewer mean trials,
         on all held-out and on unseen layers. If not: "no additional utility over the best designed control".
      h. report which encoding is selected and the validation scores; if selection equals rel_old, SELECTED
         and B are identical by construction and the expected outcome (stated in advance) is no additional
         utility, since the candidate set contains our own best design.
    Reporting: censored trials, solvable-only, unseen-layer subset, no task dropped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lab.meta import interference as v1  # noqa: E402
from lab.meta import interference_v2 as v2  # noqa: E402

CONFIGS = v2.CONFIGS
PAPER_DEFAULT = v2.PAPER_DEFAULT
THRESHOLD = v2.THRESHOLD
BUDGET = v2.BUDGET


def task_specs():
    train, _ = v1.task_specs()
    test = []
    for old_layer in (1, 2, 3, 4, 5):
        for n_old, n_new in ((28, 26), (50, 22)):
            test.append({"name": f"test4_L{old_layer}_o{n_old}_n{n_new}", "old_layer": old_layer,
                         "n_old": n_old, "n_new": n_new, "seed": 9100 + old_layer * 10 + n_old})
    return train, test


def build(out_path: Path) -> dict:
    v2_table = json.loads((ROOT / "lab" / "meta" / "interference_v2_scores.json").read_text())
    table = {k: v for k, v in v2_table.items() if v["split"] == "train"}   # same experience, same scores
    if out_path.exists():
        table.update(json.loads(out_path.read_text()))
    _, test = task_specs()
    for spec in test:
        if spec["name"] in table:
            continue
        t0 = time.time()
        scores = {json.dumps(c): v1.score_config(spec, c) for c in CONFIGS}
        table[spec["name"]] = {"split": "test", "features": v1.features(spec), **spec, "scores": scores}
        out_path.write_text(json.dumps(table))
        print(f"{spec['name']}: {time.time() - t0:.1f}s", flush=True)
    out_path.write_text(json.dumps(table))
    return table


if __name__ == "__main__":
    build(ROOT / "lab" / "meta" / "interference_v4_scores.json")
