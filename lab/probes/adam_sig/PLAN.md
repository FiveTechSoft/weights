# Adam / weight-decay footprint in the embedding rows of a donor we already have — plan fixed 2026-11-17 before any run (user: "como tú veas")

**Why.** Chaudhuri's thread (X, 14 Nov 2026; no paper, no code; roadmap gap 11, reading-list entry of 2026-11-16) claims
the optimizer leaves an exact, readable mark in the input-embedding matrix: rows of tokens never seen in training do not
move (or shrink by exactly (1 − lr·wd) per step when weight decay applies to embeddings); a row seen in a single step
moves by Adam's sign update, the same magnitude in every coordinate; rows seen often move most, regardless of gradient
size. Our `lab/probes/delta_patterns/h5_plan.md` (2026-11-07, MiMo − base, exploratory precheck) filed the top-changed
embedding rows as "rare junk tokens: noise". If the thread is right, those rows are the most-trained tokens, not noise.
Test on an artefact on disk, CPU, minutes; nothing downloaded; the GPU chain of `cc_llama` runs untouched.

**Artefacts.** Base `hf_cache/qwen35-9b-hf` (Qwen3.5-9B, bf16, vocab 248,320, d 4,096, `tie_word_embeddings: false`);
donor `hf_cache/dp_keep_dsv4_distill` (Jackrong's reasoning-distill SFT of the same base, results §43.4-43.6, the donor
whose change moved as a rank-16 adapter). Both `model.language_model.embed_tokens.weight` and `lm_head.weight` sit in
shard 1 of each; read with safetensors, no model load. MiMo (`mimo-v2.6-distill-qwen-9b`, the h5 case) exists here only
as Q6_K GGUF: exact-row tests on a quantized copy are confounded (identical rows stay identical only if both files came
from the same converter), so MiMo is **not** part of this plan; the h5 line is re-read through the Jackrong result.

**Readings** (`lab/probes/adam_sig/adam_sig.py`, 3 unit tests on synthetic rows, written first; D = donor − base in
float32 from bf16): per row, class
- `zero` (every coordinate exactly 0),
- `decay` (D = −c·w, one c per row, relative residual < 0.05, c > 0),
- `step` (|D_ij| nearly constant across j: CV < 0.15; the Adam sign update of one step),
- `other`;
counts of each; the distribution of c over decay rows (one value if pure weight decay over the same step count); the
distribution and 1-D clusters of the step magnitude (levels = learning-rate values × Adam's first-step factor); the
20 rows with the largest ‖D‖ with their tokens and ‖D‖ / median; a sample of zero-row tokens. Same for `lm_head`
(untied: every row receives softmax gradient at every step, so zero rows should be rare there — the contrast that says
the input-side zeros are "never in a batch", not "frozen matrix"). Two methods for the zero count (`all(D == 0)` vs
`max|D| == 0`), chunked vs whole classification equal on the test rows.

## Prediction (ours, low confidence; premises marked)
- Embeddings **differ** (full SFT, 70 %). If identical (frozen embeddings or LoRA-only training), the probe is void for
  the mechanism and says only "embeddings not trained" — reported, not scored as a hit.
- If they differ: **zero + decay rows ≥ 35 % of 248,320** (OLMo 2 SFT left 20 % of a 100k vocab unmoved; Qwen's vocab is
  2.5× larger and most of it is rare). Which of the two dominates depends on whether weight decay reached the embeddings
  (unknown): we predict one class ≥ 30 % and the other < 5 %.
- Decay rows, if present: c within a 2× range for ≥ 90 % of them (a single step count).
- Step rows: 0.3-10 % of rows; the three biggest magnitude levels hold ≥ 50 % of them.
- `lm_head`: zero rows < 1 %; decay rows < 1 %.
- Top-20 ‖D‖ embedding rows: ≥ 12 recognisable frequent tokens (common words, punctuation, whitespace runs, think
  tags), **not** rare junk — the opposite of the h5 reading, as the thread predicts (Adam equalises per-step motion, so
  the rows that move most are the ones trained in most steps).
- Premises: (p1) the donor is a full fine-tune of exactly this base (its deltas elsewhere were rank-16-movable, §43.4;
  shapes equal); (p2) bf16 storage rounds each update, so "exact" tests use exact zero and relative residuals, not
  equality of floats; (p3) Adam-family optimizer (the thread's mechanism; AdamW is the default of every SFT recipe we
  know); (p4) the tokenizer decode for the top rows comes from the base directory via `engine/tok.py`.
- **We would be surprised if:** embeddings differ and zero + decay rows < 10 % (vocabulary covered or a different
  optimizer); `lm_head` zero rows > 10 %; the top-20 rows are mostly rare tokens (the h5 reading was right and the
  thread's mechanism does not show here).

[post-data 2026-11-17 00:55, results §45.18, prediction row 39 ❌ on its one scorable premise.] `embed_tokens`: 248,320 /
248,320 rows exactly zero (two methods agree); `lm_head`: 248,320 / 248,320 zero. The donor did not train either matrix,
so the probe is void for the mechanism, as the plan allowed. Exploratory scan of every tensor (`which_tensors.py`,
`torch.equal` on the stored bf16): only the linear projections differ — `q/k/v/o_proj` in the 8 full-attention layers,
`out_proj` of the 24 linear-attention layers, `gate/up/down_proj` in all 32 layers; identical: embeddings, `lm_head`,
all 222 norms, `in_proj_qkv / z / a / b`, `conv1d`, `A_log`, `dt_bias`. Together with §43.4 (rank 64 keeps 99.97 % of
‖D‖²) this says the donor is a **merged LoRA** on a target-module list, not a full fine-tune: no optimizer footprint can
exist in its embeddings. The h5 "junk tokens" line was about MiMo, whose HF weights are not on disk; still open. A real
test needs a full-FT donor with an ungated base in HF form (e.g. DeepSeek-R1-Distill-Qwen-7B over Qwen2.5-Math-7B, ≈ 30 GB
to download): proposed, not launched.

## Limits known now
One donor, one base, one family; no training logs to compare against (unlike the thread's own fine-tunes); bf16 files;
thresholds (0.05, 0.15) fixed by us without calibration on a known run — a `step` row under bf16 rounding may land in
`other`, so the step count is a floor.

## Cost
CPU: two 2 GB tensors per matrix, chunked; ≈ 5 min per matrix. Output `lab/runs/adam_sig/{embed,lm_head}.json`.
