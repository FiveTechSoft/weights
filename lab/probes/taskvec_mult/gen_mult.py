"""Multiplication dataset for the taskvec_mult experiment (driver 2026-11-09).

Format (identical in train, val and test):
  user   : "Calculate: {a} x {b}"
  target : digit-partial CoT, least-significant first, scaled by place, then the final product.
           Example: a=47, b=38 -> "47·8=376; 47·30=1410; 47·38=1426"
No operand ever contains the digit 0 (no degenerate partitions). Test/val/train item sets are disjoint.

usage: python lab/probes/taskvec_mult/gen_mult.py [--root lab/runs/taskvec_mult]
"""
import argparse
import json
import random
from pathlib import Path

SEED = 20261109


def target(a: int, b: int) -> str:
    s = str(b)
    if len(s) == 1:
        return f"{a}·{b}={a * b}"
    parts = [f"{a}·{int(s[-1 - e]) * 10 ** e}={a * int(s[-1 - e]) * 10 ** e}" for e in range(len(s))]
    return "; ".join(parts) + f"; {a}·{b}={a * b}"


def sample(rng: random.Random, lo: int, hi: int) -> tuple[int, int]:
    while True:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if "0" not in str(a) and "0" not in str(b):
            return a, b


def items(rng: random.Random, n: int, lo: int, hi: int) -> list[dict]:
    seen, out = set(), []
    while len(out) < n:
        a, b = sample(rng, lo, hi)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append({"a": a, "b": b, "answer": a * b,
                    "prompt": f"Calculate: {a} x {b}", "target": target(a, b)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="lab/runs/taskvec_mult")
    ap.add_argument("--n-train", type=int, default=7000)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=200)
    a = ap.parse_args()
    rng = random.Random(SEED)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    # same distribution everywhere, items disjoint across splits (documented in the rng stream order)
    splits = [("mult_train.jsonl", a.n_train), ("mult_val.jsonl", a.n_val), ("mult_test.jsonl", a.n_test)]
    for name, n in splits:
        rows = items(rng, n, 10, 999)
        for i, r in enumerate(rows):
            r["i"] = i
        with open(root / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), "e.g.", rows[0]["prompt"], "->", rows[0]["target"])


if __name__ == "__main__":
    main()
