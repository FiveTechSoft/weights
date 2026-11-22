"""Deterministic, disjoint E2 data. Standard library only."""
import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 20261122


def zarp(a, b):
    return (3 * a + 5 * b + 7) % 97


def rows():
    rng = random.Random(SEED)
    train_pairs = rng.sample([(a, b) for a in range(50) for b in range(50)], 400)
    dev_pairs = rng.sample([(a, b) for a in range(50, 70) for b in range(70)], 100)
    test_pairs = rng.sample([(a, b) for a in range(70, 97) for b in range(70, 97)], 200)
    out = {}
    for split, pairs in (("discovery", train_pairs), ("dev", dev_pairs), ("sealed", test_pairs)):
        out[split] = [
            {"id": f"{split}-{i:03d}", "prompt": f"ZARP({a},{b}) =", "answer": zarp(a, b),
             "a": a, "b": b, "kind": "zarp"} for i, (a, b) in enumerate(pairs)
        ]
    expressions = [(a, op, b) for a in range(31) for b in range(31) for op in ("+", "-", "*")]
    picked = rng.sample(expressions, 200)
    out["arithmetic"] = [
        {"id": f"arithmetic-{i:03d}", "prompt": f"{a} {op} {b} =",
         "answer": a + b if op == "+" else a - b if op == "-" else a * b,
         "a": a, "b": b, "op": op, "kind": "arithmetic"}
        for i, (a, op, b) in enumerate(picked)
    ]
    return out


def write(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "files": {}}
    for split, items in rows().items():
        raw = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in items)
        path = out_dir / f"{split}.jsonl"
        path.write_text(raw, encoding="utf-8")
        manifest["files"][split] = {"name": path.name, "n": len(items),
                                    "sha256": hashlib.sha256(raw.encode()).hexdigest()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return manifest


def read_split(out_dir, split):
    root = Path(out_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"][split]
    raw = (root / entry["name"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {split}")
    items = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(items) != entry["n"]:
        raise ValueError(f"Dataset row count mismatch: {split}")
    return items, entry["sha256"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(write(args.out), indent=2))
