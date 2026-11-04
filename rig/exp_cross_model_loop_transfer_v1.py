import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from engine.wsp_causal_contract import (
    InterventionContract,
    EvaluationSuite,
    EvaluationSample,
    CausalVerificationRunner,
    InterventionState
)

MODEL_PATH = Path(r"C:\Users\Anto\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6")
REPORT_PATH = ROOT / "rig_runs" / "cross_model_loop_transfer_v1_manifest.json"


def compute_distinct_2(text: str) -> float:
    tokens = text.strip().split()
    if len(tokens) < 2:
        return 1.0
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    return len(set(bigrams)) / len(bigrams)


def run_experiment():
    print("=" * 80)
    print("EXPERIMENT: cross_model_loop_transfer_v1")
    print("WSP-Lang 0.7 Causal Verification Protocol")
    print("=" * 80)

    # 1. Define Evaluation Suites
    dev_samples = [
        EvaluationSample("The repetition goes on and on and on and on and", "loop_break"),
        EvaluationSample("echo echo echo echo echo", "loop_break"),
        EvaluationSample("Cycle of thoughts: thinking about thinking about thinking about", "loop_break"),
        EvaluationSample("1, 1, 1, 1, 1, 1, 1, 1,", "loop_break"),
        EvaluationSample("while(true) { loop(); } while(true) { loop(); }", "loop_break"),
    ]

    holdout_samples = [
        EvaluationSample("Again and again and again and again", "loop_break"),
        EvaluationSample("The clock ticks tick tock tick tock tick tock", "loop_break"),
        EvaluationSample("Round and round in circles, round and round in circles,", "loop_break"),
        EvaluationSample("A pattern of A B A B A B A B", "loop_break"),
        EvaluationSample("Iterate: step, step, step, step, step,", "loop_break"),
        EvaluationSample("The feedback loop creates feedback creates feedback", "loop_break"),
        EvaluationSample("Ping pong ping pong ping pong ping pong", "loop_break"),
        EvaluationSample("Recursion is defining something in terms of recursion which is", "loop_break"),
        EvaluationSample("Endless story: and then it happened again, and then it happened again,", "loop_break"),
        EvaluationSample("0, 0, 0, 0, 0, 0, 0, 0,", "loop_break"),
    ]

    regression_samples = [
        EvaluationSample("The capital of France is", "factual_target", "Paris"),
        EvaluationSample("Water consists of hydrogen and", "factual_target", "oxygen"),
        EvaluationSample("Calculate 25 + 17 =", "factual_target", "42"),
        EvaluationSample("The creator of Linux is Linus", "factual_target", "Torvalds"),
        EvaluationSample("The chemical symbol for gold is", "factual_target", "Au"),
        EvaluationSample("The author of Romeo and Juliet is William", "factual_target", "Shakespeare"),
        EvaluationSample("The opposite of hot is", "factual_target", "cold"),
        EvaluationSample("The highest mountain on Earth is Mount", "factual_target", "Everest"),
        EvaluationSample("The capital of Japan is", "factual_target", "Tokyo"),
        EvaluationSample("The primary color that mixes with blue to make green is", "factual_target", "yellow"),
    ]

    suite = EvaluationSuite(
        name="loop_and_regression_v1",
        dev_samples=dev_samples,
        holdout_samples=holdout_samples,
        regression_samples=regression_samples
    )

    contract = InterventionContract(
        target_model_name="TinyLlama-1.1B-Chat-v1.0",
        target_tensor_name="model.layers.10.mlp.down_proj.weight",
        hypothesis="MiMo-9B u_loop_L14 projected into TinyLlama Layer 10 suppresses runaway loops without breaking factual holdout capabilities",
        expected_behavior="Distinct-2 ratio increases on fresh holdout loop prompts while regression suite retains >= 90% accuracy",
        evaluation_suite=suite,
        allowed_weight_fraction=0.08,
        max_delta_norm=6.0,
        max_regression_tolerance_pct=10.0,
        min_holdout_improvement_pct=10.0,
        rollback_on_failure=True
    )

    # 2. Load Model & Tokenizer
    print(f"\n[1] Loading TinyLlama-1.1B from {MODEL_PATH.name}...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), dtype=torch.float32)
    model.eval()

    target_layer_idx = 10
    target_tensor = model.model.layers[target_layer_idx].mlp.down_proj.weight

    # 3. Load Translated Direction Vector (MiMo-9B -> TinyLlama)
    u_data = np.load(ROOT / "data" / "u_loop_tinyllama_L10.npz")
    u_tiny = torch.from_numpy(u_data["u"]).float().view(-1, 1)
    u_tiny = u_tiny / torch.norm(u_tiny)

    proj = u_tiny.t() @ target_tensor
    delta_w_candidate = -(u_tiny @ proj)

    # Evaluation function
    def eval_fn(prompt: str, sample: EvaluationSample) -> Dict[str, Any]:
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()

        if sample.expected_or_type == "loop_break":
            d2 = compute_distinct_2(gen_text)
            success = (d2 >= 0.60) or (len(gen_tokens) < 15)
            return {"prompt": prompt, "gen": gen_text[:40], "distinct_2": round(d2, 3), "success": success}
        else:
            ref = sample.reference or ""
            success = ref.lower() in gen_text.lower()
            return {"prompt": prompt, "gen": gen_text[:40], "reference": ref, "success": success}

    # 4. Execute Causal Protocol Runner
    print("\n[2] Executing 4-Way Paired Verification Protocol...")
    runner = CausalVerificationRunner(contract, model, tok)
    manifest = runner.run_4way_paired_protocol(target_tensor, delta_w_candidate, eval_fn)

    # 5. Output Findings
    print("\n" + "=" * 80)
    print("CAUSAL PROTOCOL RESULTS & VERDICT")
    print("=" * 80)
    res4 = manifest["four_way_results"]
    print(f"1. BASELINE:        Holdout Loop Break: {res4['1_baseline']['holdout']}% | Regression: {res4['1_baseline']['regression_suite']}%")
    print(f"2. CANDIDATE LEACE: Holdout Loop Break: {res4['2_candidate']['holdout']}% | Regression: {res4['2_candidate']['regression_suite']}%")
    print(f"3. PLACEBO:         Holdout Loop Break: {res4['3_placebo']['holdout']}%")
    print(f"4. RANDOM CONTROL:  Holdout Loop Break: {res4['4_random_control']['holdout']}% | Regression: {res4['4_random_control']['regression_suite']}%")
    print("-" * 80)

    cmp = manifest["causal_comparison"]
    print(f"Holdout Gain vs Baseline:  {cmp['holdout_gain_vs_base']:+.2f}%")
    print(f"Holdout Gain vs Random:    {cmp['holdout_gain_vs_random']:+.2f}%")
    print(f"Regression Loss:           {cmp['regression_loss_pct']:.2f}%")
    print(f"FINAL CONTRACT STATE:      {manifest['final_state'].upper()}")
    print(f"AUTOMATIC ROLLBACK:        {'YES (Reverted to base)' if manifest['rollback_executed'] else 'NO (Promoted)'}")
    print("=" * 80)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Full Manifest Exported: {REPORT_PATH.relative_to(ROOT)}")
    return manifest


if __name__ == "__main__":
    run_experiment()
