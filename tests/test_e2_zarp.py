"""CPU-only contract tests for E2 ZARP. No model or GPU needed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "lab/probes/skill_compile/e2_zarp"
sys.path.insert(0, str(ROOT))
from generate import RULE_TEXT, rows, read_split, write, zarp  # noqa: E402
from run import STOP, parse_integer, prompt_for  # noqa: E402
from score import exact_mcnemar_p, score  # noqa: E402
import json  # noqa: E402


def test_zarp_rule():
    # amendment A4: a + b (the only rule the 0.8B executes in context; PLAN.md, Amendments)
    assert zarp(12, 8) == 20
    assert RULE_TEXT.startswith("ZARP(a,b) = a + b.")


def test_dataset_splits(tmp_path):
    data = rows()
    assert {k: len(v) for k, v in data.items()} == {
        "discovery": 400, "dev": 100, "sealed": 200, "arithmetic": 200}
    pairs = {s: {(x["a"], x["b"]) for x in data[s]} for s in ("discovery", "dev", "sealed")}
    assert len(pairs["sealed"]) == 200
    # amendment A5: one grid, pairs disjoint across splits
    assert not pairs["sealed"] & pairs["discovery"] and not pairs["sealed"] & pairs["dev"]
    assert not pairs["dev"] & pairs["discovery"]
    assert all(x["a"] < 50 and x["b"] < 50 for s in ("discovery", "dev", "sealed") for x in data[s])
    assert all(x["answer"] == zarp(x["a"], x["b"]) for x in data["sealed"])
    # the worked examples of the ceiling prefix come from discovery only
    shots = {x["prompt"] for x in data["discovery"][:3]}
    assert all(p in RULE_TEXT for p in shots)
    assert not any(x["prompt"] in RULE_TEXT for x in data["sealed"] + data["dev"])
    write(tmp_path)
    loaded, sha = read_split(tmp_path, "sealed")
    assert loaded == data["sealed"] and len(sha) == 64
    (tmp_path / "sealed.jsonl").write_text("corrupted", encoding="utf-8")
    try:
        read_split(tmp_path, "sealed")
        assert False, "corrupt data should fail"
    except ValueError:
        pass


def test_strict_parser():
    assert parse_integer("  -12\n") == -12
    assert parse_integer("answer: 12") is None
    assert parse_integer("12 13") is None
    assert parse_integer("<think>reason</think>12") is None


def test_prompts():
    item = rows()["sealed"][0]
    assert prompt_for(item, "base") == item["prompt"]
    assert prompt_for(item, "rule_in_prompt") == RULE_TEXT + "\n" + item["prompt"]
    assert STOP == ["\n"]


def test_mcnemar():
    assert exact_mcnemar_p([True] * 10, [False] * 10) < 0.05
    assert exact_mcnemar_p([True] * 10, [True] * 10) == 1


def _write_runs(data_dir, runs, ceiling_good, base_arith_good=True):
    for split in ("sealed", "arithmetic"):
        items, sha = read_split(data_dir, split)
        arms = [("base", "a"), ("placebo", "b"), ("candidate", "c")]
        if split == "sealed":
            arms.append(("rule_in_prompt", "a"))  # the ceiling runs on the base model
        for arm, letter in arms:
            with (runs / f"{arm}_{split}.jsonl").open("w", encoding="utf-8") as f:
                for i, item in enumerate(items):
                    good = arm == "candidate" or i % 2 == 0
                    if arm == "rule_in_prompt":
                        good = i < ceiling_good
                    if split == "arithmetic":
                        good = base_arith_good or arm != "base"
                    raw = str(item["answer"] if good else item["answer"] + 1000)
                    f.write(json.dumps({
                        "id": item["id"], "prompt": item["prompt"], "answer": item["answer"],
                        "raw": raw, "parsed": int(raw), "correct": good, "arm": arm,
                        "split": split, "model_sha256": letter * 64, "dataset_sha256": sha,
                        "seed": 20261122, "n_predict": 64, "stop": STOP, "temperature": 0
                    }) + "\n")


def test_score_gates_and_provenance(tmp_path):
    data_dir, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    write(data_dir)
    _write_runs(data_dir, runs, ceiling_good=195)
    result = score(data_dir, runs)
    assert result["verdict"] == "PENDING_G0"
    assert result["g1_target"] == 190
    assert all(v for k, v in result["gates"].items() if k != "G0_persisted_model")
    assert result["gates"]["G0_persisted_model"] is None


def test_g1_follows_a_low_ceiling(tmp_path):
    data_dir, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    write(data_dir)
    _write_runs(data_dir, runs, ceiling_good=150)
    assert score(data_dir, runs)["g1_target"] == 140


def test_g3_fails_closed_when_base_arithmetic_is_low(tmp_path):
    data_dir, runs = tmp_path / "data", tmp_path / "runs"
    runs.mkdir()
    write(data_dir)
    _write_runs(data_dir, runs, ceiling_good=195, base_arith_good=False)
    r = score(data_dir, runs)
    assert r["g3_informative"] is False and r["gates"]["G3_arithmetic_loss_at_most_2pp"] is False
