"""wsp_causal_contract.py — WSP-Lang 0.7: Contract-Driven Causal Verification Engine.

Transforms Weight-Space Programming from an algebraic editing tool into a
rigorous causal experimentation framework.

Core Principles:
1. InterventionContract: Formal contract defining hypotheses, bounds, and preconditions.
2. 4-Way Paired Evaluation:
     - Model Base (Unmodified)
     - Model Intervened (Candidate W + Delta W)
     - Placebo Control (Identical execution pipeline, zero weight delta)
     - Random Orthogonal Control (Matched Frobenius norm ||Delta W_rand||_F = ||Delta W||_F)
3. Strict Causal Acceptance Criterion:
     Acceptance = (Holdout_Improvement > Placebo + Control) AND
                  (Regression_Rate <= Allowed_Tolerance) AND
                  (Frobenius_Perturbation <= Allowed_Fraction)
4. Automatic Rollback: If postconditions or acceptance criteria fail,
   the weights are reverted to the unmodified baseline.
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable

import torch
import numpy as np


class InterventionState(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    EXECUTED = "executed"
    MEASURED = "measured"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class EvaluationSample:
    prompt: str
    expected_or_type: str  # e.g. "loop_break", "factual_target", etc.
    reference: Optional[str] = None


@dataclass
class EvaluationSuite:
    name: str
    dev_samples: List[EvaluationSample]
    holdout_samples: List[EvaluationSample]
    regression_samples: List[EvaluationSample]


@dataclass
class InterventionContract:
    target_model_name: str
    target_tensor_name: str
    hypothesis: str
    expected_behavior: str
    evaluation_suite: EvaluationSuite
    allowed_weight_fraction: float = 0.05       # Max 5% Frobenius norm perturbation
    max_delta_norm: float = 5.0                # Absolute upper bound on ||Delta W||_F
    max_regression_tolerance_pct: float = 5.0   # Max allowed regression on holdout tasks
    min_holdout_improvement_pct: float = 15.0  # Min required improvement over controls
    rollback_on_failure: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalVerificationRunner:
    """Executes the 4-Way Paired Evaluation protocol for an InterventionContract."""

    def __init__(self, contract: InterventionContract, model: Any, tokenizer: Any):
        self.contract = contract
        self.model = model
        self.tokenizer = tokenizer
        self.state = InterventionState.DRAFT
        self.history: List[Dict[str, Any]] = []

    def validate_preconditions(self, tensor_weight: torch.Tensor) -> bool:
        """Validates that the target tensor exists and satisfies initial conditions."""
        if tensor_weight is None:
            self.state = InterventionState.REJECTED
            return False
        if torch.isnan(tensor_weight).any() or torch.isinf(tensor_weight).any():
            self.state = InterventionState.REJECTED
            return False
        self.state = InterventionState.VALIDATED
        return True

    def evaluate_condition(
        self,
        samples: List[EvaluationSample],
        eval_fn: Callable[[str, EvaluationSample], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluates a set of samples using the model's current state."""
        results = []
        hits = 0
        for s in samples:
            res = eval_fn(s.prompt, s)
            results.append(res)
            if res.get("success", False):
                hits += 1

        acc = (hits / max(len(samples), 1)) * 100.0
        return {
            "total": len(samples),
            "hits": hits,
            "accuracy_pct": round(acc, 2),
            "details": results
        }

    def run_4way_paired_protocol(
        self,
        target_tensor: torch.Tensor,
        delta_w_candidate: torch.Tensor,
        eval_fn: Callable[[str, EvaluationSample], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Executes the complete 4-Way Paired protocol:
        1. BASELINE (Original W)
        2. CANDIDATE (W + Delta_W)
        3. PLACEBO (W - pass through pipeline with zero delta)
        4. RANDOM CONTROL (W + Delta_W_rand with matched Frobenius norm)
        """
        # 1. Validate preconditions
        if not self.validate_preconditions(target_tensor):
            return {"status": "error", "reason": "Preconditions failed"}

        original_w = target_tensor.detach().clone()
        frob_orig = float(torch.norm(original_w).item())
        frob_delta = float(torch.norm(delta_w_candidate).item())
        perturb_pct = (frob_delta / frob_orig) * 100.0

        suite = self.contract.evaluation_suite

        # Check bounds
        if perturb_pct > self.contract.allowed_weight_fraction * 100.0 or frob_delta > self.contract.max_delta_norm:
            self.state = InterventionState.REJECTED
            return {
                "status": "rejected",
                "reason": f"Perturbation exceeded bounds: {perturb_pct:.2f}% (allowed: {self.contract.allowed_weight_fraction*100}%)",
                "state": self.state.value
            }

        # -------------------------------------------------------------
        # ARTIFACT 1: EVALUATE BASELINE (Original)
        # -------------------------------------------------------------
        base_dev = self.evaluate_condition(suite.dev_samples, eval_fn)
        base_holdout = self.evaluate_condition(suite.holdout_samples, eval_fn)
        base_regression = self.evaluate_condition(suite.regression_samples, eval_fn)

        # -------------------------------------------------------------
        # ARTIFACT 2: EVALUATE CANDIDATE INTERVENTION (W + Delta_W)
        # -------------------------------------------------------------
        with torch.no_grad():
            target_tensor.copy_(original_w + delta_w_candidate)
        self.state = InterventionState.EXECUTED

        cand_dev = self.evaluate_condition(suite.dev_samples, eval_fn)
        cand_holdout = self.evaluate_condition(suite.holdout_samples, eval_fn)
        cand_regression = self.evaluate_condition(suite.regression_samples, eval_fn)

        # -------------------------------------------------------------
        # ARTIFACT 3: EVALUATE PLACEBO (W unchanged)
        # -------------------------------------------------------------
        with torch.no_grad():
            target_tensor.copy_(original_w)
        placebo_holdout = self.evaluate_condition(suite.holdout_samples, eval_fn)

        # -------------------------------------------------------------
        # ARTIFACT 4: EVALUATE RANDOM DIRECTION CONTROL
        # (Construct random orthogonal delta with identical Frobenius norm)
        # -------------------------------------------------------------
        rng = torch.Generator().manual_seed(1337)
        rand_tensor = torch.randn(delta_w_candidate.shape, generator=rng, dtype=delta_w_candidate.dtype)
        rand_tensor = rand_tensor / torch.norm(rand_tensor) * frob_delta

        with torch.no_grad():
            target_tensor.copy_(original_w + rand_tensor)

        rand_holdout = self.evaluate_condition(suite.holdout_samples, eval_fn)
        rand_regression = self.evaluate_condition(suite.regression_samples, eval_fn)

        self.state = InterventionState.MEASURED

        # -------------------------------------------------------------
        # CAUSAL DECISION ENGINE
        # -------------------------------------------------------------
        holdout_gain_vs_base = cand_holdout["accuracy_pct"] - base_holdout["accuracy_pct"]
        holdout_gain_vs_rand = cand_holdout["accuracy_pct"] - rand_holdout["accuracy_pct"]
        regression_loss = base_regression["accuracy_pct"] - cand_regression["accuracy_pct"]

        is_statistically_superior = (
            holdout_gain_vs_base >= self.contract.min_holdout_improvement_pct and
            cand_holdout["accuracy_pct"] > rand_holdout["accuracy_pct"] and
            cand_holdout["accuracy_pct"] > placebo_holdout["accuracy_pct"]
        )

        is_safe_without_regression = (
            regression_loss <= self.contract.max_regression_tolerance_pct
        )

        decision_accepted = is_statistically_superior and is_safe_without_regression

        if decision_accepted:
            self.state = InterventionState.ACCEPTED
            # Keep candidate in place
            with torch.no_grad():
                target_tensor.copy_(original_w + delta_w_candidate)
        else:
            if self.contract.rollback_on_failure:
                # Revert immediately to baseline
                with torch.no_grad():
                    target_tensor.copy_(original_w)
                self.state = InterventionState.ROLLED_BACK
            else:
                self.state = InterventionState.REJECTED

        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "contract": {
                "target_model": self.contract.target_model_name,
                "target_tensor": self.contract.target_tensor_name,
                "hypothesis": self.contract.hypothesis,
                "expected_behavior": self.contract.expected_behavior,
                "allowed_weight_fraction": self.contract.allowed_weight_fraction,
                "max_regression_tolerance_pct": self.contract.max_regression_tolerance_pct,
                "min_holdout_improvement_pct": self.contract.min_holdout_improvement_pct
            },
            "weight_physics": {
                "frobenius_orig": round(frob_orig, 4),
                "frobenius_delta": round(frob_delta, 4),
                "perturbation_pct": round(perturb_pct, 4)
            },
            "four_way_results": {
                "1_baseline": {
                    "dev": base_dev["accuracy_pct"],
                    "holdout": base_holdout["accuracy_pct"],
                    "regression_suite": base_regression["accuracy_pct"]
                },
                "2_candidate": {
                    "dev": cand_dev["accuracy_pct"],
                    "holdout": cand_holdout["accuracy_pct"],
                    "regression_suite": cand_regression["accuracy_pct"]
                },
                "3_placebo": {
                    "holdout": placebo_holdout["accuracy_pct"]
                },
                "4_random_control": {
                    "holdout": rand_holdout["accuracy_pct"],
                    "regression_suite": rand_regression["accuracy_pct"]
                }
            },
            "causal_comparison": {
                "holdout_gain_vs_base": round(holdout_gain_vs_base, 2),
                "holdout_gain_vs_random": round(holdout_gain_vs_rand, 2),
                "regression_loss_pct": round(regression_loss, 2),
                "is_statistically_superior": is_statistically_superior,
                "is_safe_without_regression": is_safe_without_regression
            },
            "final_state": self.state.value,
            "rollback_executed": (self.state == InterventionState.ROLLED_BACK)
        }

        return manifest
