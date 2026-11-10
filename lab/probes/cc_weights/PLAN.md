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
