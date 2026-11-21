# Time-step bias ± 0.5 (Fan et al., arXiv 2611.21885) on Qwen3.5-9B, with the controls the paper lacks — plan fixed 2026-11-18, before any run

**Why.** *Language Models Can Rewrite Their Own Weights* (Fan et al., 16 Nov 2026; README "References") reports that the
only model of the Qwen3.5 family that improved itself (27B) got its **largest gain, +6.6 points on 490 held-out
problems (p < 10⁻⁵)**, from one uniform knob: *raise the time-step bias of every linear-attention layer by 0.5*. The
first-protocol edit (weaken the finished-vs-truncated direction in layer 8, our §43.17 mechanism rediscovered) gave
+1.1 (p 0.038). The paper runs no placebo, no opposite sign, no same-magnitude random control, and reports no token
counts after the edit. Their 9B damaged itself by trial. Our 9B is the same family (Qwen3.5-9B Q6_K, 24 Gated DeltaNet
layers, 8 full-attention layers at 3, 7, …, 31, plus the MTP block 32). The knob costs nothing to test with the
controls, through the standard `dev-2048` suite. Roadmap gap 27.

## What the knob is
Gated DeltaNet (Qwen3.5 / Qwen3-Next linear attention) keeps a recurrent state S per head updated with a delta rule,
S_t = exp(g_t) · S_{t−1} + β_t (v_t − S_{t−1} k_t) k_tᵀ, where the per-head decay is
**g_t = A · softplus(a_t + dt_bias)**, A = −exp(A_log) < 0 (GGUF `blk.N.ssm_a`), a_t the projected input (`ssm_alpha`),
`dt_bias` the per-head bias (GGUF `blk.N.ssm_dt.bias`, F32 [32]). softplus(dt_bias + a) is the **forgetting rate** of
the linear memory: larger → the state decays faster → shorter effective memory. The edit adds +0.5 to every head's bias.

Values in our GGUF (24 layers × 32 heads = 768): mean −2.43, std 3.19, range −8.69 … +18.5 (layer 0 holds the +18.5
outlier, mean +2.37; layers 1-20 sit at −3 to −4; layers 21-30 near −1 to 0). At the typical −2.4: softplus 0.084 →
+0.5 gives 0.136 (**× 1.62 faster decay**, memory time constant × 0.62), −0.5 gives 0.052 (× 0.62, time constant
× 1.6). The change is largest for heads near 0 (softplus' = sigmoid ≈ 0.5) and negligible for heads below −6
(softplus ≈ 0.002) or far above 0 (already fast). So "+0.5 everywhere" mostly shortens the memory of the mid-range
heads in the late layers 21-30, where the biases cluster near 0.

## Arms (one run each; `lab/runs/dt_bias/`; GGUFs under `servidos/`, git-ignored, manifests kept)
Built from `hf_cache/h4/A0-Q6_K.gguf` with `python wsp.py edit` and the two new ops (`add-scalar`, `add-scalar-signed`,
glob `blk.*.ssm_dt.bias`, F32 only, policy |value| ≤ 8; tests `tests/test_wsp_add_scalar.py`):
- **`dt_plus05`** — `ediciones/dt_plus05.json`: +0.5 on all 24 layers (the paper's edit).
- **`dt_minus05`** — `ediciones/dt_minus05.json`: −0.5 on all 24 layers (opposite sign).
- **`dt_pm05`** — `ediciones/dt_pm05_seed2026.json`: ±0.5 with a random sign per layer, seed 2026 (same magnitude per
  layer, no coherent direction: the placebo).
- **base**: the two `bench_run` runs of the untouched base on `dev-2048` already in queue 2
  (`lab/runs/bench/qwen9b_base_a.json`, `_b.json`): base value and the noise band.

All arms through `bench_run suite=dev-2048` (GSM8K 0-99 + MATH-500 50, greedy, 2,048 tokens, 2 slots, two graders,
preservation battery: HumanEval 164, 40 compliance prompts, garbage rates) with `provenance=` the wsp.py manifest (now
carrying `output_sha256`, verified against the served sha), then `bench_compare` of each arm against base_a with
`placebo=dt_pm05` and `base2=base_b`, and `bench_table`. Programs `programs/dt_bias_{plus,minus,pm,compare}.wspl`,
chain `lab/runs/dt_bias/chain.sh` (builds the GGUFs on the CPU if missing, serves each once with the cc_v2 flags,
waits for any llama-server / llama-perplexity). Cost: ≈ 57 min GSM8K + MATH plus ≈ 50 min battery per arm ≈ **1.8 GPU h
per arm, ≈ 5.5 h for the three**.

## Readings (fixed)
Per arm: GSM8K and MATH correct (g1 and g2), cuts, mean tokens, no-think share, HumanEval / 164, compliance / 40,
repeated-tail share; paired against base_a; band = discordants and |Δ| between base_a and base_b.
- **R1 "the knob acts":** `dt_plus05` differs from base beyond 2 × band on tokens or on cuts.
- **R2 "direction matters":** `dt_plus05` and `dt_minus05` move tokens (or cuts) in opposite directions and the placebo
  sits between them (inside the band on that measure).
- **R3 "accuracy gain":** only with p < 0.05 on both graders on a benchmark, placebo not significant, and
  |Δ| ≥ 2 × band — the `bench_compare` rule.
- **R4 "preservation":** HumanEval ≥ base − 5, compliance ≥ base − 3, garbage ≤ base + 2 pp (the battery's rule).

## Prediction (row 49) — written before the GGUFs exist
Premises: (P1) a uniform shift of every head's forgetting rate is a constant-push-like change, closer to §45.27's logit
bias than to the conclude vector: it cannot know where the state is; (P2) faster forgetting (+0.5) makes long
derivations lose their earlier steps, which on this base (96 cuts of 150 at 2,048, 19 % repeated tails) should cut
rumination a little and hurt long MATH items; slower forgetting (−0.5) the reverse; (P3) their +6.6 came at a 4,096
budget on a 27B whose failure was 80 % truncation; at 2,048 on the 9B the same knob has less room.
- **`dt_plus05`:** tokens −5 to −20 % (GSM8K mean 1,350-1,650 vs base ≈ 1,690), cuts 70-92 (base ≈ 96), GSM8K
  **46-62**, MATH 3-11, no-think ≤ 2 %, HumanEval 134-150, compliance 12-20 / 40, repeated tails 10-20 %.
- **`dt_minus05`:** tokens −3 to +10 %, cuts 92-105, GSM8K 42-54, MATH 3-8, HumanEval 130-148, repeated tails 18-30 %.
- **`dt_pm05` (placebo):** tokens ± 8 %, cuts 88-102, GSM8K 44-56, MATH 3-9, HumanEval 134-150.
- **base band (base_a vs base_b):** GSM8K |Δ| ≤ 4, discordants 6-14 of 100; MATH |Δ| ≤ 3; tokens ± 4 %.
- **Primary: R1 holds (the knob acts on tokens or cuts) 60 %; R2 holds 50 %; R3 (an accuracy gain that passes the rule)
  15 %.** Against the conclude vector (83 / 35, 25 cuts, 735 tokens): no arm comes within 10 points of it on GSM8K (90 %).
- We would be surprised if: `dt_plus05` reaches GSM8K ≥ 70 or MATH ≥ 20 (the knob would be doing what the vector
  does); any arm falls to GSM8K ≤ 30 or HumanEval ≤ 120 (self-damage like their 9B, from ± 0.5 alone); the placebo is
  significant on both graders; `dt_minus05` saves more tokens than `dt_plus05`; the base band exceeds 8 on GSM8K.

## Two methods
`bench_compare` output and `lab/probes/cc_llama/pair_recount.py` on the run JSONs (bench_run keeps bench_runner's item
records); the manifests' `output_sha256` against `sha256sum` of the served files in the chain log.

## Not done
No dose sweep (± 0.25, ± 1.0), no per-layer-band variant (late layers only, where the biases cluster near 0), no 4,096
budget, no other family. If R1 and R2 hold and R3 does not, the next arm is +0.5 on layers 21-30 only at 4,096 tokens.
