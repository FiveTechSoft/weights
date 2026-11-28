"""Protocol tests for the reversible WSPLang locus screening engine."""
import copy

import pytest
import torch

from engine.wsp_locus import Candidate, probe_columns, rank_candidates


def fixture():
    w = torch.eye(8, dtype=torch.float64)
    initial = w.clone()
    candidates = [Candidate("w", j) for j in range(8)]
    return w, initial, candidates


def test_localizes_known_column_and_restores():
    w, initial, candidates = fixture()
    calls = []

    def evaluate(split):
        calls.append(split)
        if split == "preservation":
            return 1.0
        return float(1 - w[3, 3].square())

    result = probe_columns({"w": w}, candidates, evaluate, dose=1)
    ranked = rank_candidates(result)
    assert [r["column"] for r in ranked] == [3]
    assert torch.equal(w, initial)
    first = calls.index("heldout")
    assert "discovery" not in calls[first:]
    assert result["checkpoint_written"] is False
    assert result["backward_calls"] == 0
    for row in result["candidates"]:
        assert abs(row["delta_norm"] - row["placebo_norm"]) <= .02 * row["delta_norm"]


def test_placebo_equivalent_rejected():
    w, initial, candidates = fixture()

    def evaluate(split):
        if split == "preservation":
            return 1.
        return float((w-initial).norm())

    report = probe_columns({"w": w}, candidates, evaluate, dose=1)
    assert rank_candidates(report) == []
    assert torch.equal(w, initial)


def test_heldout_failure_rejected():
    w, initial, candidates = fixture()

    def evaluate(split):
        return float(1-w[3,3].square()) if split == "discovery" else 1.

    report = probe_columns({"w": w}, candidates, evaluate, dose=1)
    assert report["shortlist"]
    assert rank_candidates(report) == []
    assert torch.equal(w, initial)


def test_preservation_failure_rejected():
    w, initial, candidates = fixture()

    def evaluate(split):
        if split == "preservation":
            return float(w[3,3])
        return float(1-w[3,3].square())

    report = probe_columns({"w": w}, candidates, evaluate, dose=1)
    assert rank_candidates(report) == []
    assert not report["shortlist"]
    assert torch.equal(w, initial)


def test_restores_even_on_exception():
    w, initial, candidates = fixture()

    def evaluate(split):
        if not torch.equal(w, initial):
            raise RuntimeError("evaluation failed")
        return 0.

    with pytest.raises(RuntimeError, match="evaluation failed"):
        probe_columns({"w": w}, candidates, evaluate, dose=1)
    assert torch.equal(w, initial)


def test_budget_duplicate_nonfinite_rejected():
    w, initial, candidates = fixture()
    for items, kwargs in ((candidates, {"max_modified_weights": 7}),
                          ([candidates[0]]*2, {})):
        with pytest.raises(ValueError):
            probe_columns({"w": w}, items, lambda _: 0., dose=1, **kwargs)
    with pytest.raises(ValueError, match="nonfinite metric"):
        probe_columns({"w": w}, candidates, lambda _: float("nan"), dose=1)
    assert torch.equal(w, initial)


def test_reproducible_and_ranking_not_changed_by_holdout():
    w, initial, candidates = fixture()

    def evaluate(split):
        if split == "preservation":
            return 1.
        return float(1-w[3,3].square())

    a = probe_columns({"w": w}, candidates, evaluate, dose=1, seed=12)
    b = probe_columns({"w": w}, candidates, evaluate, dose=1, seed=12)
    assert a == b
    assert torch.equal(w, initial)
    # Contract: ranking filters a previously frozen order, never sorts by holdout.
    report = copy.deepcopy(a)
    report["shortlist"] = [{"column": 3, "passed": True, "heldout_gain": .1},
                           {"column": 4, "passed": True, "heldout_gain": .9},
                           {"column": 5, "passed": False, "heldout_gain": 1.}]
    assert [r["column"] for r in rank_candidates(report)] == [3,4]
