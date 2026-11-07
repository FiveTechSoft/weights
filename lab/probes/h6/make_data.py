"""H6 data (lab/probes/delta_patterns/h6_plan.md): 3-digit x 3-digit products (G1 addendum; 3x2 was 80 % for M0), answer only, deterministic splits.

Every model (trained or not) is prompted with the same 3 solved examples (SHOTS) before the item, so M0 knows the
format and a gain measures arithmetic, not format. Training uses the same prompt; the loss is on the target only.
No (a, b) pair appears in two splits or in SHOTS.

usage: python lab/probes/h6/make_data.py  -> lab/runs/h6/data/<split>.jsonl
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "lab" / "runs" / "h6" / "data"
SEED = 2026
SHOTS = [(314, 827), (708, 256), (462, 193)]
SIZES = {"train": 20000, "dev": 200, "test_id": 1000}           # 3 x 3 digits
OOD = {"ood_easy": (500, 3, 2), "ood_hard": (500, 4, 3)}


def prompt(a: int, b: int) -> str:
    return "".join(f"{x}*{y}={x * y}\n" for x, y in SHOTS) + f"{a}*{b}="


def target(a: int, b: int) -> str:
    return f"{a * b}\n"


def _draw(rng, n, da, db, used):
    out = []
    while len(out) < n:
        p = (rng.randrange(10 ** (da - 1), 10 ** da), rng.randrange(10 ** (db - 1), 10 ** db))
        if p not in used:
            used.add(p)
            out.append(p)
    return out


def make_splits():
    rng = random.Random(SEED)
    used = set(SHOTS)
    splits = {name: _draw(rng, n, 3, 3, used) for name, n in SIZES.items()}
    for name, (n, da, db) in OOD.items():
        splits[name] = _draw(rng, n, da, db, used)
    return splits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, items in make_splits().items():
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps({"a": a, "b": b, "answer": a * b}) + "\n"
                                                   for a, b in items), encoding="utf-8")
        print(name, len(items))


if __name__ == "__main__":
    main()
