"""Protocol v3: task-relative configuration encoding, fresh held-out tasks.

Why v3 exists: protocol v2 showed that the nearest-task proposer does not transfer to old layers it never
saw (first-proposal score 0.035 on old layers 2 and 4). Reading that failure suggested that the useful rule
is relative to the task ("write at the old layer"). That idea came AFTER seeing the v2 held-out results, so
it is not scored on them: v3 uses new held-out tasks (new seeds and sizes).

Fixed before any v3 held-out task is scored:
  * experience: the same 9 training tasks as v1/v2 (old layers 1, 3, 5), all 50 configurations;
  * held-out: 10 new tasks, old layers 1 to 5 x sizes (35, 30) and (45, 20), seeds never used before;
    old layers 2 and 4 are unseen in the experience;
  * threshold 0.8, budget 15, 20 outer seeds x 50 shuffles (as v2);
  * policies: random, static, absolute contextual (v2 proposer), RELATIVE contextual (new), shuffled
    relative control (wrong task labels), oracle;
  * criteria, all on first-proposal score (threshold-free) and on trials-to-success:
      d.  relative contextual > static and > shuffled relative, on all held-out and on unseen layers;
      e.  relative contextual > absolute contextual on unseen layers (the point of the change).
      f.  design-vs-learning: two extra baselines. FIXED RULE (no experience): configurations with
          layer == old_layer first, remaining ties in grid order. RELATIVE STATIC (experience pooled over ALL
          training tasks in relative encoding, no task similarity). The relative encoding and the offset-0
          rule are our design decisions, not something the system discovered. Success with the relative
          encoding is only credited to experience if relative contextual beats the FIXED RULE on first-proposal
          score and on trials (otherwise the result measures our design choice, not learning).
    Claim "transfers to unseen layers" requires d and e on unseen layers.
    Claim "experience adds beyond the design" additionally requires f. v2 held-out tasks are not reused as
    confirmation. Reporting: censored trials,
    solvable-only, unseen-layer subset, never dropping tasks.
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
        for n_old, n_new in ((35, 30), (45, 20)):
            test.append({"name": f"test3_L{old_layer}_o{n_old}_n{n_new}", "old_layer": old_layer,
                         "n_old": n_old, "n_new": n_new, "seed": 8100 + old_layer * 10 + n_old})
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
    build(ROOT / "lab" / "meta" / "interference_v3_scores.json")
