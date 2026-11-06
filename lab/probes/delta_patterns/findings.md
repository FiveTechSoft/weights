# Findings log: what post-training changes in Qwen3.5-9B (running notebook)

Plan: `plan.md` (fixed before the other models were seen). Data: `lab/runs/delta_patterns/`. Summary in results.md §38.
Each entry: what was measured, the number, where it comes from, how sure it is, and what it teaches for WSPLang.
Status labels: **verified** (second method agrees), **measured** (one method), **open** (needs a control still running).

## F1. MiMo is a full fine-tune, not an adapter. (verified)
- 97 % of the 8.95 B text weights change; ||ΔW||/||W|| = 9.2 % over the model. `stats_mimo.json`.
- Check: relative change and stable rank of 3 tensors recomputed with numpy float64 and a full SVD: identical
  (`check_mimo.json`).
- WSPLang: an improvement of this kind is not a small edit. Budgets such as `max_modified_weights = 1 %` cannot
  express it; a "transfer" op has to work on whole tensors.

## F2. Where MiMo changes: gates and MLP most, the recurrent dynamics not at all. (verified)
- Median relative change per tensor type: DeltaNet β (`in_proj_b`) 12.7 %, `mlp.gate_proj` 12.5 %, `up_proj` 11.3 %,
  `down_proj` 10.8 %; q/k and DeltaNet inputs 8-10 %; embeddings / `lm_head` 6.5-6.9 %; full-attention `v_proj`
  4.9 % and `o_proj` 5.6 %; conv1d 1.4 %; norms < 1.3 %.
- `A_log` and `dt_bias` (DeltaNet decay dynamics) are not trained: the base stores them in F32, MiMo in BF16, and the
  only difference is that cast (max |Δ| = 0.0078, one bf16 step). `dt_bias` differs in no layer after rounding.
- By depth: bell shape, 8.5 % at layer 0, ~11 % at layers 17-23, 7.5 % at layer 31.
- WSPLang: a map of where post-training acts. Protect-by-default lists could include `A_log`, `dt_bias` and norms
  (post-training leaves them alone); edit budgets could be per tensor type instead of global.

## F3. MiMo's change is diffuse. (verified)
- Large matrices: stable rank 200-500 (6-12 % of the smaller dimension); the 16 largest directions hold 2-4 % of the
  change's energy. Exception: `in_proj_b` (32 × 4096) holds 63 % in 16 directions, but it only has 32 rows.
- WSPLang: rank-one ops (`project_out`, imprints) cannot reproduce such an improvement. Worth knowing before trying.

## F4. MiMo's change mostly goes in new directions, except in the embedding. (measured)
- Share of the delta's top-16 subspace inside the base weight's own top-16 subspace (`base_alignment`, `sig_mimo.pt`):
  embedding 0.56 (chance 0.004); `down_proj` 0.078, `q_proj` 0.072, DeltaNet `in_proj_z/qkv` 0.05, `lm_head` 0.043,
  MLP gate/up 0.036-0.038, `o_proj` 0.003 (chance); DeltaNet β/α at chance (0.5 with 32 rows).
- Reading: in the embedding, the update mainly grows directions the embedding already had; elsewhere it adds new ones.
- WSPLang: "scale what is already big" (cheap spectral ops) can describe at most the embedding part.

## F5. NeoHorse changes 18 times less than MiMo, evenly. (verified)
- 0.5 % relative change; 61 % of weights differ (the rest are below bf16 resolution); flat over layers (0.4-0.6 %);
  norms, `A_log`, `dt_bias` untouched. Check: `check_neohorse.json` agrees.
- Shape: more concentrated than MiMo (16 directions = 6-19 % of the energy in large matrices), still not low rank.
- WSPLang: very different improvements in size (agent skills from a 0.5 % change; reasoning distillation from 9 %).
  Size of change is not a measure of the size of the gain.

## F6. MiMo and NeoHorse change the same kinds of tensors. (measured; specificity open)
- Spearman of the per-tensor-type profile: 0.93 (0.86 for matrices only). Per layer: 0.51 (MiMo bell, NeoHorse flat).
- Meets the pre-registered H1 threshold (>= 0.7) for this pair. Whether it is a mark of improvement or of any
  fine-tune waits for the role-play control (Nyx-RP).

## F7. Their top subspaces overlap far above chance; their deltas are nearly orthogonal. (open)
- Same-tensor top-16 subspace overlap (`overlap_mimo_neohorse.json`): embedding 0.71 (×182 chance), `lm_head` 0.33
  (×85), `q_proj` 0.19 (×49), MLP gate/up 0.15-0.16 (×40), DeltaNet `in_proj_z/qkv` 0.15 (×39), `down_proj` 0.09
  (×23), `o_proj` ×15, `v_proj` ×10, DeltaNet `out_proj` ×8, `k_proj` ×6; DeltaNet β/α ×1.1-1.2.
- Cosine of the two deltas: +0.008 to +0.011 in every type, +0.08 in `lm_head`. Small, but with millions of entries
  far above the noise (~0.0002), and the same sign everywhere.
- Open: the structured null (same tensor type, different layer) and F4's base alignment decide whether this is a
  shared direction of improvement or shared geometry of the hidden state (outlier dimensions). F4 already suggests
  the embedding overlap is mostly "both grow the embedding's dominant directions"; the MLP overlap is not explained
  by W's top subspace (alignment ~0.04 against overlap ~0.15).

## F8. Same task, measured behaviour: TwIL-8B vs MiMo-9B. (verified by two graders + manual check)
- At an 8192-token cap, greedy: GSM8K 96 vs 93 %, MATH-500 88 vs 68 %; MiMo's 19 cut answers are long reasoning,
  not loops (1 of 19 tails periodic). results.md §37.
- TwIL's balanced variant comes from an RL stage that rewards short reasoning. With a budget, finishing matters
  more than knowing. Candidate study: TwIL balanced vs verbose weight diff (isolates that RL stage).

## Tool lessons (each was a silent error)
- WSPLang `diff` read only the first shard of a sharded model (missed ~3/4 of tensors) and loaded both models in RAM.
  Fixed in `engine/weight_diff.py`, with a test that fails on the old code.
- `engine/comparative_drift.compute_architectural_drift` returns a cached JSON of another model pair whatever its
  arguments; `compute_spectral_delta` ignores a non-GGUF target. The earlier MiMo "anatomy" (gates +14-19 %, norms
  0.14 %) came from GGUF Q6_K and cannot be reproduced; in bf16 the gates are +12.5-12.7 % and norms up to 1.3 %.
  The qualitative claim (gates change most) holds.
- Tensor names differ between multimodal (`model.language_model.*`) and text-only (`model.*`) checkpoints of the
  same base; `Renamed(reader, text_model_name)` lines them up. Without it a diff silently compares nothing.
- Dtype differences (F32 vs BF16) show up as tiny "changes"; check the stored dtypes before reading small deltas.
