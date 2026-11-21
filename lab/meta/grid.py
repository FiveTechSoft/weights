"""Builds the score table with real NanoWSP-3 runs: task x configuration -> recall.

Each cell is an actual closed-form imprint followed by actual forward passes (see
engine/nanowsp3_runtime.py). Nothing is typed in by hand. The table is cached as JSON so the
(slow) grid runs once and the (fast) meta evaluation can be repeated and audited.

Tasks differ in the number of facts and in how similar their keys are (shared prefix),
which changes how ill-conditioned the key Gram matrix is.
"""
from __future__ import annotations

import json
import random
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LAYERS = (1, 3, 5)
ETAS = (1.0, 3.5, 8.0)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1.0)
CONFIGS = [(l, e, r) for l, e, r in product(LAYERS, ETAS, LAMBDAS)]
PAPER_DEFAULT = (3, 3.5, 1e-2)       # values used by scripts/demo_nanowsp3_capabilities.py

KEY_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TARGET_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_task(n: int, prefix_len: int, seed: int):
    """n facts "MEM: <key>=" -> one letter. prefix_len>0 makes all keys share a prefix."""
    r = random.Random(seed)
    prefix = "".join(r.choice("ABCDEFGH") for _ in range(prefix_len))
    tail = 2 if prefix_len else None
    keys = set()
    while len(keys) < n:
        if prefix_len:
            keys.add(prefix + "".join(r.choice(KEY_CHARS) for _ in range(tail)))
        else:
            keys.add("".join(r.choice(KEY_CHARS) for _ in range(r.choice([3, 4, 5]))))
    return [("MEM: %s=" % k, r.choice(TARGET_CHARS)) for k in sorted(keys)]


def score_config(facts, cfg) -> float:
    import torch
    from engine.nanowsp3_runtime import NanoWSP3
    layer, eta, lam = cfg
    torch.manual_seed(0)
    model = NanoWSP3()
    model.imprint_thematic_corpus([{"name": "t", "facts": facts}],
                                  target_layer=layer, eta=eta, reg_lambda=lam)
    hits = sum(model.evaluate_recall(p, t)[0] for p, t in facts)
    return hits / len(facts)


def task_specs():
    """Disjoint train / test splits. Test uses new seeds AND sizes never seen in training."""
    train, test = [], []
    for n in (10, 30, 100, 300):
        for pl in (0, 2, 3):
            train.append({"name": f"train_n{n}_p{pl}", "n": n, "prefix_len": pl, "seed": 100 + n + pl})
    for n in (20, 60, 200):
        for pl in (0, 2, 3):
            test.append({"name": f"test_n{n}_p{pl}", "n": n, "prefix_len": pl, "seed": 9000 + n + pl})
    return train, test


def build(out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = json.loads(out_path.read_text()) if out_path.exists() else {}
    train, test = task_specs()
    for split, specs in (("train", train), ("test", test)):
        for spec in specs:
            if spec["name"] in table:
                continue
            facts = make_task(spec["n"], spec["prefix_len"], spec["seed"])
            t0 = time.time()
            scores = {json.dumps(c): score_config(facts, c) for c in CONFIGS}
            table[spec["name"]] = {"split": split, **spec, "scores": scores}
            out_path.write_text(json.dumps(table))
            print(f"{spec['name']}: {time.time() - t0:.1f}s", flush=True)
    return table


if __name__ == "__main__":
    build(ROOT / "lab" / "meta" / "grid_scores.json")
