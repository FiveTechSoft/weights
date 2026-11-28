"""Deterministic, disjoint E2 data. Standard library only."""
import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 20261122


def zarp(a, b):
    # amendment A4 (2026-11-23): was (3a + 5b + 7) % 97. Step 0 on discovery items, Qwen3.5-0.8B, completion mode,
    # answer cut at the first newline: no multi-operation rule is executed even with the rule and 3 worked
    # examples in the prompt (a + b + 7: 2/40; 2a + b, 3a + 5b + 7, the mod-97 rule: 0/40 with the rule alone);
    # a + b with the rule and 3 examples: 39/40. See PLAN.md, Amendments.
    return a + b


def rows():
    rng = random.Random(SEED)
    # amendment A5: one grid, disjoint pairs (was: sealed a, b in [70, 96], an extrapolation the plan did not name)
    pool = rng.sample([(a, b) for a in range(50) for b in range(50)], 700)
    train_pairs, dev_pairs, test_pairs = pool[:400], pool[400:500], pool[500:700]
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


def rule_text():
    """The rule_in_prompt prefix (amendment A2): the public rule and the first 3 discovery items, worked."""
    shots = rows()["discovery"][:3]
    return "ZARP(a,b) = a + b.\n" + "".join(f"{x['prompt']} {x['answer']}\n" for x in shots)


def write(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "files": {}}
    for split, items in rows().items():
        raw = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in items)
        path = out_dir / f"{split}.jsonl"
        path.write_bytes(raw.encode("utf-8"))  # bytes: write_text would write CRLF on Windows and break the hash
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


RULE_TEXT = rule_text().rstrip("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(write(args.out), indent=2))
