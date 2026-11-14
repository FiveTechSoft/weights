# Compile the "conclude" direction into the weights (cc → down_proj) — plan fixed 2026-11-10, before any run

**Why.** The donor-free "conclude" vector is confirmed as a llama.cpp control vector (results §43.17 dev, §43.18
sealed: GSM8K 43 → 87/100, MATH 4 → 25/50; same-norm placebo 23 / 2). A control vector is an inference flag, not a
weight edit. Question: **can the same effect live in the checkpoint**, as an edit of existing matrices (same tensors,
same shapes, nothing added), the way Concept Country writes it into `down_proj`?

## What is compiled

`hf_cache/cc/cc.gguf` holds 31 increments v_N (d = 4096), added by llama.cpp at the output of layer N = 1..31. The
output of `mlp.down_proj` of layer N goes straight into the residual stream at the end of that layer, so adding
ΔW_N = v_N u_Nᵀ to that `down_proj` (4096 × 12288) adds v_N · (u_Nᵀ x) at every token, where x is the input of
`down_proj`. If u_Nᵀ x ≈ 1 on the tokens the model actually sees, the edit reproduces the vector.

- **Fit (closed form, no gradients):** u_N = argmin Σ_t (u ᵀx_t − 1)² + λ‖u‖² = (KᵀK + λI)⁻¹ Kᵀ1, K = inputs of
  `down_proj` at layer N, λ = 1e-3 · tr(KᵀK)/12288 (fixed now, not tuned). Layers fitted in order; layer N's inputs
  are taken from the model with layers < N already edited (the stream the next layer really sees), as in
  `lab/probes/close_probe/cf_distill.py`.
- **Calibration tokens:** the cc discovery prompts and their own traces (`lab/runs/cc/traces.jsonl`, 128 prompts:
  64 MATH-500 discovery + 64 GSM8K train), prompt + reasoning tokens, thinking template. 100 prompts fit, 28 held out
  (fixed split by seed 20261111). None of dev, screen or either sealed set.
- **Placebo:** `hf_cache/cc/cc.placebo2026.gguf` (same norm per layer, random direction; the §43.18 placebo) compiled
  by the same fit. Control for "any rank-1 write of this size into `down_proj` does it".
- **Outputs (never over an input):** rank-1 PEFT LoRA on `down_proj` (B = v_N, A = u_Nᵀ) → GGUF adapter (diagnostic,
  exact in fp); and **merged Q6_K**: copy of `hf_cache/h4/A0-Q6_K.gguf` with the 31 `ffn_down` tensors replaced by
  requantized W + ΔW (`wsp.py` Q6_K requant, faithful port of `quantize_row_q6_K_ref`). Disk: ~7 GB per merged arm
  (31 GB free on 2026-11-10). Original files untouched.

## Gate G0 (before any benchmark, CPU/GPU fit only)

On the 28 held-out prompts' tokens, per layer: relative error ‖ΔW x − v‖ / ‖v‖ = |u ᵀx − 1|, median over tokens.
**Proceed only if the median over layers of that median is ≤ 0.5.** Also reported: the same with the Q6_K merged
weights (error from requantization: ‖Q(W+ΔW) − W − ΔW‖ vs ‖ΔW‖ per layer), and greedy agreement of the first 256
tokens on 10 held-out prompts, merged vs control vector. If G0 fails, the result is written as negative and the line
stops (no rank-2, no λ search).

## Dev (same items and server settings as §43.17; 2,048 tokens, greedy, 2 slots)

Arms: base and control vector are re-used from §43.17 (`lab/runs/cc/dev_{base,vec}.json`; the setup is
deterministic — the sealed base re-run matched 150/150 item by item). New arms: **merged** (`cc_merged-Q6_K.gguf`, no
flags), **placebo merged**, and **LoRA** (A0 + `--lora`, diagnostic, not in the criterion).

- **C1** merged > base, exact McNemar p < 0.05 with both graders on GSM8K, or on MATH-500 (as §43.17).
- **C2** placebo merged not better than base on either benchmark (p ≥ 0.05 or below base).
- **C3** merged not below base by more than 2 on GSM8K (grader 1).
- **C4 (compilation preserves the effect)** merged not below the control vector with p < 0.05 (grader 1) on GSM8K
  or on MATH-500.
- Reported, not criteria: tokens, cut answers, loop tails; HumanEval first 20 (base 20/20) on the merged model, as a
  non-math retention check.

## Sealed (only if dev passes C1-C4)

Both earlier confirmations used `data/sealed_confirm.json`. A **fresh set** is made before the dev run: 100 GSM8K
test + 50 MATH-500 items not in dev, not in sealed set 1, and not present in any file under `lab/runs/` (scanned by
`(bench, i)`), seed 20261111, sha256 recorded in `data/sealed_confirm2.json`. Arms: base, control vector, merged,
placebo merged; same C1-C4. Written to results.md whatever it says; nothing re-run or re-tuned after seeing it.

## Limits known now

One model (Qwen3.5-9B), one budget (2,048), greedy, Q6_K. A rank-1 write gives v · (u ᵀx), not a constant: where
u ᵀx drifts (long contexts, other tasks) the push changes size — the scale window is narrow (×2 breaks), so this is
the main risk and why G0 exists.

## Repeat with the corrected vector (addendum 2026-11-12, written before the run)

**Why.** `cc.gguf`, compiled above (§43.19, §43.21, §43.25), carries the post-norm flaw in `direction.31` (results
§45.3). The re-derived vector `hf_cache/cc/cc_v2.gguf` (layers 1-30 identical, layer 31 norm 13.2 → 18.1) passed dev
and confirmed on sealed set 1 (§45.4). The compiled result must be redone with it; this is the order of work agreed
with the user (finish cc_v2 → recompile and sealed → only then generalize).

**What runs** (`lab/runs/cc_v2/chain_weights.sh`, after the composition chain; one GPU job at a time):
1. Op `compile_control_vector` from `programs/cc_compile_v2.wspl`: vector `cc_v2.gguf`, placebo
   `cc_v2.placebo2026.gguf`, same calibration (`lab/runs/cc/traces.jsonl`, same split), same λ, same base GGUF, out
   `hf_cache/cc_op_v2` (~14 GB; 31 GB free). Nothing in `hf_cache/cc_weights` is touched or deleted.
2. G0 as above (≤ 0.5). Mechanical check (`lab/probes/cc_weights/v2_check.py`): u_N for N = 1..30 equal to the first
   compilation's (cos ≥ 0.999) and the written `ffn_down` of layers 1-30 **byte-identical** to `cc_merged-Q6_K.gguf`;
   layer 31 reported (cos of u, cos of the written tensor, merge error). Descriptive: a miss is reported, not a stop.
3. Dev: arms **merged** and **placebo merged** (no LoRA arm: diagnostic only in the first run, nothing decided on it),
   same items and settings; base and control vector reused from `lab/runs/cc_v2/dev_{base,vec}.json` (cc_v2 ×1.0;
   determinism shown twice, §43.18 and §45.4). Criteria C1-C4 unchanged (`verdict.py`, directories via `CCW_*`).
4. Sealed, only if dev passes: `data/sealed_confirm2.json` (second use; first was §43.21), arms **vec_v2 ×1.0**
   (new run), **merged**, **placebo merged**; base copied from `lab/runs/sealed_ccw/base.json` (same model, same
   settings, deterministic). Same C1-C4. Reported, not a criterion: HumanEval first 20 on the merged model (first
   compilation: 19/20; control vector 15/20).
5. Outcome written to results.md whatever it is; nothing re-run or re-tuned after seeing it.

**Prediction** (ours, low confidence), arm and control, premises marked:
- Mechanical: G0 median ≈ 0.032 (layers 1-30 identical; layer 31's error was 0.009 and its norm grows, so ≤ 0.02);
  30/31 `ffn_down` byte-identical; layer 31 merge relative error 0.45-0.50 (Q6_K loses ≈ half the magnitude, as in
  every layer of the first compilation).
- Dev merged: GSM8K 84-90, MATH 31-38 (the first merge tracked its vector within ±3: 87/34 vs 84/32; cc_v2 is 83/35).
  Placebo merged: GSM8K 25-40, MATH 2-7 (first: sealed 30/4). **Passes C1-C4.**
- Sealed set 2 (base 46/7): vec_v2 GSM8K 86-92, MATH 27-35; merged GSM8K 86-92, MATH 26-34 (first merge 89/26);
  placebo merged 25-40 / 2-7. **Confirms.** HumanEval-20 merged 18-20.
- Premises: p1 the fit is deterministic (31/31 identical in §43.25) so layers 1-30 reproduce; p2 Q6_K keeps the
  direction and halves the magnitude also at layer 31 (held for layer 31 of the first vector: cos 0.896, err 0.466);
  p3 the sealed base from §43.21 is reusable (base re-runs matched 150/150 twice).
- "We would be surprised if": merged below vec_v2 by > 5 on MATH with p < 0.05 (the bigger layer-31 push would not
  survive requantization); placebo merged above base with p < 0.05 on either benchmark; G0 median > 0.1; fewer than
  28 of the 31 `ffn_down` identical (the fit would not be deterministic after all).

Cost: op ≈ 1.1 h (first run 4 058 s), 2 dev arms ≈ 1.5 h, 3 sealed arms ≈ 2.3 h, HumanEval ≈ 10 min: ≈ 5 GPU h.
