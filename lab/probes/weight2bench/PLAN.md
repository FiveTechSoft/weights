# Can the benchmark be read from the weights? Weight-only features vs measured outcomes — plan fixed 2026-11-18, before any computation

**Why.** The driver asked (2026-11-18) whether benchmarks could be measured by valuing the weights alone, i.e. whether
a correlation between weight-side quantities and benchmark outcomes exists. Our thesis T7 says the effect of an edit
needs the forward pass (Kocabay's readers: 53-88 % false positives on inert edits; Ahmed 2611.08070: weight reading is
local arithmetic plus priors). We have never put our own number on it, and we have the material: a few dozen edits
(control vectors, their same-norm placebos, dose variants, compiled rank-1 writes, the MIT Exp 8 edit and its pruned
forms) on four models, each with a paired benchmark result against the same base on the same items. No forward pass
is run here; every feature is computed from the artefact and the base's unembedding; every target is read from the
existing run files.

## Arms (every one has a weight artefact and a paired result on the same items; the base is the same file served without the artefact)
Families: **q9** Qwen3.5-9B, **phi** Phi-4-mini-reasoning, **llama** DeepSeek-R1-Distill-Llama-8B, **q08** Qwen3.5-0.8B.
Kinds: `vec` (llama.cpp control vector, one direction per layer, served at scale s: the effective direction is s·d_l),
`compiled` (rank-1 dW_l = outer(v_l, u_l) written into `down_proj`; residual-space direction v_l, magnitude ‖v_l‖·‖u_l‖),
`cols` (64-column `down_proj` edit; residual direction = top left singular vector of the column block per layer,
magnitude ‖ΔW_l‖_F). Labels: `real` (the arm the plan called the treatment), `placebo` (same-norm random direction
or compiled placebo), `dose` (the same artefact at another scale), `variant` (the artefact with a layer removed, a
port through a map, a pruned edit).

| family | items | arms (label) |
|---|---|---|
| q9 | dev 150 (`lab/runs/cc_v2/dev_*`) | cc_v2 @1.0 (real), cc_v2.placebo2026 @1.0 (placebo), cc_no31 @1.0 (variant), compiled cc_op_v2 vector (real), compiled cc_op_v2 placebo (placebo) |
| q9 | screen 64 MATH (`screen_*`) | cc_v2 @0.25 / 0.5 / 1.0 / 2.0 (dose), placebo seeds 1-8 @1.0 (placebo) |
| q9 | dev 150 v1 (`lab/runs/cc/dev_*`) | cc @1.0 (real, the first vector with the last-layer bug), cc.placebo2026 @1.0 (placebo) |
| q9 | sealed 150 (`sealed_cc_v2`, `sealed_ccw_v2`) | cc_v2 @1.0 (real), placebo (placebo), compiled (real), compiled placebo (placebo) |
| phi | dev 150, screen 64, sealed 150 | cc_phi @0.5 (real), placebo @0.5 (placebo), screen @0.25-2.0 (dose), compiled vector / placebo (real / placebo), sealed vec / placebo / merged / placebo_merged |
| llama | dev 150, screen 64, sealed 1-3 | cc_llama @0.5 (real), @0.25 / @0.35 (dose), placebo @0.5 (placebo), screen @0.25-2.0 (dose), compiled vector / placebo, sealed 1 vec / placebo @0.5, sealed 2 merged / placebo_merged / vec, sealed 3 vec / placebo @0.35 |
| q08 | test 150 (`act_map/cc08/test_*`, `exp8_prune/test_*`) | cc_own @1.0 (real), cc_port @2.0 (variant), cc_embport @0.5 (variant), cc_port.placebo2026 @2.0 (placebo), Exp 8 edit (real; two runs, kept as two rows for ρ, one for AUC), Exp 8 placebo (placebo), D4k (variant), D4k complement (variant) |
| q08 | dev 96 (`act_map/cc08/dev_*`) | own / port / embport @0.25, 0.5, 1.0, 2.0, 4.0 (dose) |

Excluded, with reason: `cc_weights` dev arms (the merged GGUF they served, `hf_cache/cc_weights`, no longer exists);
`happy_*` vectors (mood, no accuracy target); H-series and NanoWSP (no base-vs-arm benchmark of this kind).

## Features (weight-only; computed on the effective per-layer directions D ∈ ℝ^{L×d} after the served scale)
Magnitude family: `mag_total` = ‖D‖_F; `mag_mean`, `mag_max`, `argmax_depth` (layer of max norm / L);
`participation` = (Σ_l ‖d_l‖²)² / (L·Σ_l ‖d_l‖⁴); `eff_rank` = exp(entropy of the normalised singular values of D);
for compiled arms also `rel_dW` (mean ‖dW_l‖/‖W_l‖ from `fit_summary.json`) and `G0` (from the op's `report.json`).
Direction family: `cos_close_mean`, `cos_close_late8`, `cos_close_last` = cosine of d_l with the readout direction
of the close-thinking token r = lm_head[t] ⊙ g (g = final RMSNorm weight; lm_head = embed_tokens when tied), averaged
over all layers, the last 8, or taken at the last layer; `cos_open_*` the same for the open-thinking token (control);
`cos_gap` = `cos_close_mean` − `cos_open_mean`. Tokens: q9 / q08 `</think>` 248069 and `<think>` 248068; llama
128014 / 128013; phi has no single token (`</think>` → [808, 49631, 29], `<think>` → [33313, 881, 29]): we use the
tag-specific pieces 49631 and 881 and say so. `cos_own_contrast` = mean cosine with the base's own traces vector
(cc_v2 / cc_phi / cc_llama / cc_own), reported only for arms that are not that vector or its dose variants (circular
otherwise). `cos_vs_placebo` = mean cosine between an arm and its own placebo (descriptive, expected ≈ 0).

## Targets (from the paired run files, grader 1; grader 2 reported)
`d_gsm` = correct_arm − correct_base on GSM8K, `d_math` likewise on MATH-500 (screen arms: MATH only), `d_tok` =
mean tokens change in %, `d_cut` = cuts_arm − cuts_base; `n` items. Label `is_real` = 1 for real / dose / variant,
0 for placebo.

## Readings (fixed)
- R1: Spearman ρ of each feature with `d_gsm`, `d_math`, `d_tok`, `d_cut`, per family (n per family) and pooled
  (all rows; rank-based so families mix without rescaling; n stated).
- R2: AUC real-vs-placebo per feature on one row per artefact at its plan scale on dev/test (dose rows and screen
  rows excluded from AUC so a vector is not counted five times; the 8 q9 screen placebos count, each is a distinct
  artefact).
- R3: for any feature with pooled |ρ| > 0.5 against `d_gsm` or `d_math`: leave-one-family-out ρ (does it hold when a
  family is removed?).
- Two methods: `run.py` (features + scipy-free Spearman/AUC in `features.py`) and `recount.py` (plain numpy, no shared
  code) on `arms.json`; every ρ and AUC must agree to 1e-6.

## Prediction (row 47)
Premises: P1 magnitude is what the plans tuned (scale chosen on screen items), so within a family it tracks token
saving but not correctness; P2 placebos share the magnitude profile by construction (same norm per layer), so no
magnitude feature can separate real from placebo; P3 the conclude direction is read before `</think>`, so its late
layers should align with the close-token readout more than a random direction does.
- Magnitude features: AUC real-vs-placebo **0.40-0.60**; ρ with `d_tok` within q9 / phi / llama **−0.6 to −0.8**
  (more norm, fewer tokens); pooled |ρ| with `d_gsm` and `d_math` **< 0.3**.
- `cos_close_late8`: AUC real-vs-placebo **> 0.8 (60 %)**, 0.6-0.8 (30 %); pooled ρ with `d_math` **< 0.5 (80 %)**;
  `cos_open_*` AUC 0.4-0.7; `cos_gap` AUC within 0.1 of `cos_close_late8`.
- `cos_own_contrast`: separates compiled real from compiled placebo (AUC > 0.9 on those rows, 85 %); on q08 port /
  embport it is 0.2-0.6 (the §45.12 mapping cosines).
- `cos_vs_placebo`: |mean| < 0.05 everywhere.
- **Primary: no weight-only feature has pooled Spearman |ρ| > 0.5 with `d_gsm` or `d_math` (75 %).**
- Surprised if: a magnitude feature separates real from placebo (AUC > 0.75); any feature predicts `d_gsm` or `d_math`
  with pooled |ρ| > 0.7; `cos_close_late8` AUC < 0.5; a leave-one-family-out ρ flips sign.

## Not done
No forward pass, no new runs, no fitting of a predictor (n is a few dozen rows from one lab's plans: a fitted model
would overfit by construction); no claim beyond "correlates / does not" at this n.

## Results (after the prediction; 2026-11-18, CPU; `lab/runs/weight2bench/arms.json`, `summary.json`; recount AGREE on 75 numbers)
78 arm rows (q9 24, phi 13, llama 17, q08 24), 31 rows in the real-vs-placebo set (14 real / variant, 17 placebo).
One fix between the first run and the scored one, made before scoring and stated here: the compiled "real" arms are
the own vector written into `down_proj`, so their `cos_own_contrast` was 1.00 by construction; the plan's rule
("not that vector") excludes them and the code now does too (first run: pooled ρ(`cos_own_contrast`, d_math) 0.55;
scored run: 0.32).

**Spearman ρ with the paired change (pooled n 58 for GSM8K, 78 for MATH / tokens; per family q9 / phi / llama / q08):**

| feature | d_gsm pooled | d_math pooled | d_tok pooled | d_math per family | d_tok per family |
|---|---|---|---|---|---|
| mag_total | 0.06 | −0.11 | 0.04 | −0.19 / −0.37 / −0.46 / 0.28 | 0.02 / −0.03 / −0.08 / −0.40 |
| eff_rank | −0.19 | −0.02 | 0.14 | −0.21 / −0.38 / −0.06 / −0.26 | 0.32 / 0.56 / 0.38 / 0.19 |
| cos_close_late8 | 0.22 | 0.34 | −0.53 | 0.71 / 0.47 / 0.59 / 0.09 | −0.72 / −0.69 / −0.58 / −0.20 |
| cos_open_mean | 0.42 | 0.36 | −0.41 | 0.64 / 0.47 / 0.64 / −0.23 | −0.68 / −0.64 / −0.57 / 0.14 |
| cos_gap | 0.10 | 0.20 | −0.38 | 0.46 / −0.42 / 0.11 / 0.43 | −0.49 / 0.68 / −0.53 / −0.50 |
| cos_own_contrast (n 33 / 41) | 0.16 | 0.32 | −0.35 | 0.50 / 0.71 / −0.21 / 0.16 | −0.43 / 0.00 / 0.05 / −0.24 |

Largest pooled |ρ| with d_gsm or d_math among features with n ≥ 20: **0.42** (`cos_open_mean`, d_gsm) and **0.36**
(`cos_open_mean`, d_math). `cos_vs_placebo` (n 9, values all within ± 0.04) shows ρ −0.59 with d_math: nine
near-zero numbers, reported, not read.

**AUC real vs placebo (pooled 14 / 17; per family q9 4/11, phi 2/2, llama 2/2, q08 6/2):**
magnitude 0.21 (q9 0.20, phi 0.25, llama 0.50, q08 0.17: within matched pairs the norms are identical by construction,
the variants `no31`, v1 and `embport` are smaller than the placebos, so the number is an artefact of the variants, not
a signal); participation 0.73; `cos_close_late8` **0.93** (1.00 / 1.00 / 1.00 / 1.00); `cos_close_mean` 0.88;
`cos_open_mean` 0.88, `cos_open_late8` 0.83; `cos_gap` 0.66; `cos_own_contrast` 0.68 (q9 1.0 on 1 / 11, q08 0.70 on
5 / 2; phi and llama have no non-circular positives left). The separating values are small: `cos_close_late8` real
0.013 (q9), 0.001 (phi), 0.058 (llama), 0.051 (q08 own), 0.004 (Exp 8) against placebos within ± 0.01; and the open
token separates almost as well, so what separates is "any consistent alignment with the think-tag readout rows", not
the close token specifically. Dose rows share one cosine per artefact (scale-invariant) while their Δ varies 7 → 30 →
−5 (q9 screen), so no cosine feature can order doses. `cos_vs_placebo`: 0.000-0.038 everywhere.

**Prediction (row 47) against the data.** Primary ✓ for every feature with n ≥ 20 (max 0.42); the only |ρ| > 0.5 is the
nine-row descriptive `cos_vs_placebo` (noise), stated. Magnitude AUC band 0.40-0.60 ✗ (0.21, artefact of smaller
variants; 0.5 within matched pairs); magnitude ρ with d_tok within family −0.6..−0.8 ✗ (≈ 0: placebos share the norm,
so norm cannot track tokens); magnitude pooled |ρ| with correctness < 0.3 ✓. `cos_close_late8` AUC > 0.8 ✓ (0.93);
its pooled ρ with d_math < 0.5 ✓ (0.34). `cos_open_*` AUC 0.4-0.7 ✗ (0.83-0.88). `cos_gap` within 0.1 of
`cos_close_late8` ✗ (0.66 vs 0.93). `cos_own_contrast` compiled real vs placebo AUC > 0.9: ✓ where assessable (q9 1.0,
one positive) and not assessable on phi / llama after the circularity fix; q08 port 0.35 ✓, embport 0.08 ✗ (band
0.2-0.6). `cos_vs_placebo` |mean| < 0.05 ✓. Surprises: none fired (no AUC > 0.75 for magnitude in the predicted
direction, no |ρ| > 0.7, `cos_close_late8` AUC not < 0.5, no leave-one-family-out needed). Suggested score **◐**:
primary ✓, five secondary bands ✗.

**Reading.** From the weights alone we can tell a real edit from its same-norm placebo (alignment with the think-tag
readout, AUC 0.93; magnitude tells nothing, as it must) and we can see, per family, that edits more aligned with that
readout save more tokens (ρ −0.6 to −0.7 on q9 / phi / llama); we cannot read how many items an edit will get right
(pooled |ρ| ≤ 0.42, and the dose series — same cosine, Δ MATH from +30 to −5 — shows why: correctness depends on the
dose and the model, which the direction does not carry). Caveat: 78 rows from one lab's plans, 31 in the placebo
set, four families, every target measured on 64-150 items with the known ± 5 noise on the 0.8B; this is a
correlation table, not a predictor, and the features were chosen knowing what the edits are.
