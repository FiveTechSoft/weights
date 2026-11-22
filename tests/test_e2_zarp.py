"""CPU-only contract tests for E2 ZARP. No model or GPU needed."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "lab/probes/skill_compile/e2_zarp"
sys.path.insert(0, str(ROOT))
from generate import rows, read_split, write, zarp  # noqa: E402
from run import parse_integer  # noqa: E402
from score import exact_mcnemar_p, score  # noqa: E402
import json  # noqa: E402


def test_zarp_rule():
    assert zarp(12, 8) == 83
    assert 0 <= zarp(96, 96) < 97


def test_dataset_splits(tmp_path):
    data = rows()
    assert {k: len(v) for k, v in data.items()} == {
        "discovery": 400, "dev": 100, "sealed": 200, "arithmetic": 200}
    assert len({(x["a"], x["b"]) for x in data["sealed"]}) == 200
    assert all(x["a"] >= 70 and x["b"] >= 70 for x in data["sealed"])
    assert all(x["a"] < 50 and x["b"] < 50 for x in data["discovery"])
    assert all(x["answer"] == zarp(x["a"], x["b"]) for x in data["sealed"])
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


def test_mcnemar():
    assert exact_mcnemar_p([True] * 10, [False] * 10) < 0.05
    assert exact_mcnemar_p([True] * 10, [True] * 10) == 1


def test_score_gates_and_provenance(tmp_path):
    data_dir = tmp_path / "data"
    runs = tmp_path / "runs"
    runs.mkdir()
    write(data_dir)
    for split in ("sealed", "arithmetic"):
        items, sha = read_split(data_dir, split)
        for arm, letter in (("base", "a"), ("placebo", "b"), ("candidate", "c")):
            with (runs / f"{arm}_{split}.jsonl").open("w", encoding="utf-8") as f:
                for i, item in enumerate(items):
                    good = arm == "candidate" or i % 2 == 0
                    # All arms 100% on arithmetic: no preservation loss.
                    if split == "arithmetic":
                        good = True
                    raw = str(item["answer"] if good else item["answer"] + 1000)
                    f.write(json.dumps({
                        "id": item["id"], "prompt": item["prompt"], "answer": item["answer"],
                        "raw": raw, "parsed": int(raw), "correct": good, "arm": arm,
                        "split": split, "model_sha256": letter * 64, "dataset_sha256": sha,
                        "seed": 20261122, "n_predict": 64, "temperature": 0
                    }) + "\n")
    result = score(data_dir, runs)
    assert result["verdict"] == "PENDING_G0"
    assert all(v for k, v in result["gates"].items() if k != "G0_persisted_model")
    assert result["gates"]["G0_persisted_model"] is None
