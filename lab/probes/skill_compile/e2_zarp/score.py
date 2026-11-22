"""Strict paired E2 scorer; refuses missing or mismatched runs."""
import argparse
import json
import math
from pathlib import Path

from generate import read_split
from run import STOP

SPLITS = ("sealed", "arithmetic")
EDIT_ARMS = ("base", "placebo", "candidate")
CEILING = "rule_in_prompt"  # amendment A2: run on sealed only
MIN_BASE_ARITH = 0.5  # amendment A3: below it G3 measures nothing


def exact_mcnemar_p(a, b):
    """Exact two-sided binomial McNemar p for paired boolean arrays."""
    n01 = sum(not x and y for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def load_run(path, expected, dataset_sha, arm, split):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) != len(expected):
        raise ValueError(f"Wrong row count: {path}")
    result = []
    model_shas = set()
    for line, item in zip(lines, expected):
        row = json.loads(line)
        if (row.get("id"), row.get("prompt"), row.get("answer")) != (
                item["id"], item["prompt"], item["answer"]):
            raise ValueError(f"Dataset mismatch: {path}")
        if row.get("arm") != arm or row.get("split") != split or row.get("dataset_sha256") != dataset_sha:
            raise ValueError(f"Run provenance mismatch: {path}")
        if (row.get("temperature") != 0 or row.get("seed") != 20261122 or row.get("n_predict") != 64
                or row.get("stop") != STOP):
            raise ValueError(f"Inference configuration mismatch: {path}")
        sha = row.get("model_sha256", "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"Invalid model SHA: {path}")
        model_shas.add(sha)
        from run import parse_integer
        parsed = parse_integer(row.get("raw", ""))
        if row.get("parsed") != parsed or row.get("correct") != (parsed == item["answer"]):
            raise ValueError(f"Invalid score or parse: {path}")
        result.append(bool(row["correct"]))
    if len(model_shas) != 1:
        raise ValueError(f"Multiple model hashes in {path}")
    return result, next(iter(model_shas))


def score(data, runs):
    results, shas = {}, {}
    runs_needed = [(arm, split) for split in SPLITS for arm in EDIT_ARMS] + [(CEILING, "sealed")]
    for split in SPLITS:
        items, dataset_sha = read_split(data, split)
        for arm in [a for a, s in runs_needed if s == split]:
            path = Path(runs) / f"{arm}_{split}.jsonl"
            vals, sha = load_run(path, items, dataset_sha, arm, split)
            results[arm, split] = vals
            if arm in shas and shas[arm] != sha:
                raise ValueError(f"Model SHA differs across splits for {arm}")
            shas[arm] = sha
    if len({shas[a] for a in EDIT_ARMS}) != len(EDIT_ARMS) or shas[CEILING] != shas["base"]:
        raise ValueError("Edit arms must have distinct model SHA256 hashes; rule_in_prompt runs on the base model")
    accuracy = {arm: {split: sum(results[arm, split]) / len(results[arm, split])
                      for split in SPLITS} for arm in EDIT_ARMS}
    accuracy[CEILING] = {"sealed": sum(results[CEILING, "sealed"]) / len(results[CEILING, "sealed"])}
    n_sealed = len(results["candidate", "sealed"])
    ceiling_n = sum(results[CEILING, "sealed"])
    # amendment A2: 190/200 if the ceiling reaches it, else within 10 items (5 pp) of the ceiling
    g1_target = 190 if ceiling_n >= 190 else max(0, ceiling_n - 10)
    z = accuracy["candidate"]["sealed"]
    base = accuracy["base"]["sealed"]
    placebo = accuracy["placebo"]["sealed"]
    arith_drop = accuracy["base"]["arithmetic"] - accuracy["candidate"]["arithmetic"]
    p_base = exact_mcnemar_p(results["candidate", "sealed"], results["base", "sealed"])
    p_placebo = exact_mcnemar_p(results["candidate", "sealed"], results["placebo", "sealed"])
    gates = {
        "G0_persisted_model": None,  # Must be audited manually from compiler provenance.
        "G1_sealed_target": sum(results["candidate", "sealed"]) >= g1_target,
        "G2_vs_base_placebo": z - base >= 0.20 - 1e-12 and z - placebo >= 0.20 - 1e-12
                             and p_base < 0.05 and p_placebo < 0.05,
        "G3_arithmetic_loss_at_most_2pp": (accuracy["base"]["arithmetic"] >= MIN_BASE_ARITH
                                           and arith_drop <= 0.02 + 1e-12),
        "G4_dataset_and_run_integrity": True,
    }
    return {"verdict": "PENDING_G0" if all(v for k, v in gates.items() if k != "G0_persisted_model")
            else "FAIL", "gates": gates, "accuracy": accuracy, "model_sha256": shas,
            "mcnemar_p": {"candidate_vs_base": p_base, "candidate_vs_placebo": p_placebo},
            "arithmetic_drop_pp": 100 * arith_drop,
            "g1_target": g1_target, "ceiling_sealed_n": ceiling_n, "n_sealed": n_sealed,
            "g3_informative": accuracy["base"]["arithmetic"] >= MIN_BASE_ARITH,
            "note": "G0 requires manual verification of merged checkpoint and placebo provenance."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = score(a.data, a.runs)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
