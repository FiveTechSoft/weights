# Discrete per-row writer in the bf16 alphabet: can a Tree-Ising-style move set carry our compiled edit? — plan fixed 2026-11-18, before any measurement

**Why.** §45.23 showed that *rounding* an edited float matrix to a discrete alphabet buries our edits (rank-1,
‖dW‖/‖W‖ 0.01-0.1) in orthogonal noise 2-12× their size. Tree-Ising (Salloum & Ali, arXiv 2611.14210, abstract only)
does not round: it *solves* a discrete problem per output row whose moves are native to bf16 (sign flips, exponent
shifts) with a behavioural objective, and reports beating Astra's dense Exp 8 edit with fewer weights touched. Before
any attempt to write a discrete edit (and before any "create a capability without a donor" attempt that would need a
writer at all), the mechanism question is: **with our own compile target dW = v uᵀ per layer, how much of the edit can
a per-row discrete solve realise in the bf16 move set, at what sparsity and at what dose?** CPU only; no GGUF written;
no GPU run here.

## Formulation
Base row w ∈ bf16ⁿ (n = 12,288 on the 9B `down_proj`, 3,584 on the 0.8B), target change t = v_r · u (row r of our
compiled dW; `hf_cache/cc_op_v2/vector/fit.pt`, dW = outer(v, u) as `engine.cc_compile.merge_q6k` writes it). Moves
per entry m_j ∈ {none, sign flip, ×2, ×½}: every move maps a bf16 value to a bf16 value (sign bit, exponent ±1,
mantissa untouched). Realised change d_j = move(w_j) − w_j ∈ {0, −2w_j, +w_j, −w_j/2}.

1. **Separable objective (exact):** min Σ_j (d_j − t_j)² + λ·[m_j ≠ none]. Per entry the terms are independent, so
   the optimum is a threshold: take the best move iff gain_j = t_j² − min_m (d_{j,m} − t_j)² > λ. λ is set from the
   quantile of the gains to hit a target share of changed entries (`lam_for_fraction`). No search, no seed.
   Arithmetic that fixes expectations: a move helps only if |d_j| < 2|t_j| with the right sign, and the smallest move
   is |w_j|/2, so only entries with |w_j| < 4|t_j| can change. Our compiled dW has |t| ≈ 2 % of a typical |w| (9B layer
   16: |t| ≈ 2·10⁻⁴, |w| ≈ 9·10⁻³); the Exp 8 edit has |t| ≈ 60 % of |w| on its 64 columns.
2. **Coupled objective (what Tree-Ising actually does, approximated):** min ‖X (d − t)‖² + λ·#changed, X = the
   calibration inputs of `down_proj` (positions × n), as `cc_compile.fit_layers` captures them. Entries interact
   through X, so overshoots can be compensated; greedy coordinate descent per row with a sweep budget and a seeded
   visiting order (`discrete_row_anneal`; never increases the objective). Real X needs a forward pass of the 9B over
   the calibration traces (GPU, `cc_compile` hook); here it is exercised on synthetic X in the tests only.

Code `lab/probes/discrete_writer/writer.py`; tests `tests/test_discrete_writer.py` (written first: bf16 moves exact,
recovery of a target made of moves, λ monotonicity, coupled greedy monotone); readings via `engine.ternary.overlap`
(cos, alpha = ⟨D, t⟩/‖t‖², noise = ‖D − αt‖/‖αt‖) plus share of entries changed and bf16 bit flips (Hamming distance
of the codes).

## Measurements (CPU)
- 9B `hf_cache/qwen35-9b-hf`, layers 1, 16, 31 (full 4,096 × 12,288 matrices), target k·dW for k ∈ {1, 2, 5}, λ set
  for ≈ 1 %, 5 %, 20 % of the entries changed (reported; if fewer entries have a positive gain, the attainable share is
  reported instead and λ = 0).
- 0.8B `hf_cache/qwen35-08b-hf`, layers 15 and 19, target = the Exp 8 edit (`data/exp8_edit_recovered.npz`, 64
  columns), k = 1, same λ rule (shares are of the whole matrix; only the 64 columns can change).
- Second method: `engine/ternary.py`'s `overlap` is shared code, so the recount is the per-entry identity check in the
  tests (target made of moves is recovered exactly) and, for the real matrices, cos recomputed from D and t with a
  separate one-line numpy expression in the run script (printed beside the `overlap` value).

## Prediction (row 45)
Premise: per-entry |t| ≪ |w|/2 for the compiled dW (above), so at dose 1 almost no entry has a positive gain and the
threshold writer can only touch the few entries where the base weight is tiny; what it realises there is the right
sign but a magnitude set by w, not by t.
- 9B compile, k = 1: attainable share of entries with positive gain **0.3-3 %**; at that share cos **0.10-0.35**,
  alpha 0.5-3 (sparse amplified copy), noise ≥ 1.5. At 5 % and 20 % the requested share is not attainable (λ = 0).
- 9B compile, k = 2: cos 0.20-0.45 at the attainable share (1-6 %). k = 5: cos **0.40-0.65**, attainable share 3-15 %.
- Exp 8, k = 1: attainable share 20-60 % of the 64 columns' entries (0.4-1.1 % of the matrix); cos at the attainable
  share **0.55-0.80**; at 1 % of the matrix cos within 0.05 of that.
- **Primary: the threshold writer does not reach cos ≥ 0.7 to the compiled 9B dW at ≤ 5 % entries changed at dose 1
  (90 %).** Surprise if it does; if any 9B cell at k = 1 has cos ≥ 0.5; if Exp 8 at k = 1 has cos < 0.4; if the
  coupled test objective ever increases.
- Expected reading if the primary holds: in the bf16 move set the compiled edit is representable only at 5× its dose
  or with the coupled (activation) objective, which needs the forward pass; a discrete writer for our edits is a GPU
  project (X from `cc_compile`), not a CPU rounding. If the primary fails, the threshold writer alone is a candidate
  discrete op and the GPU dev run below is justified.

## What a GPU dev run would be (not launched; needs the coordinator)
Discrete D per layer (9B, 31 layers) written into `hf_cache/h4/A0-Q6_K.gguf` by `wsp.py edit` with `add-cols-vectors`
on the changed columns (or a dense variant of that op), served without flags; dev items and criteria C1-C4 of
`lab/probes/cc_weights/PLAN.md` (merged > base with both graders on one bench, placebo merged n.s., GSM8K ≥ base − 2,
merged not below the control vector), placebo = the same writer applied to the same-norm random-direction fit
(`hf_cache/cc_op_v2/placebo`). If the discrete D needs the coupled objective, X is captured once per layer with
`cc_compile.fit_layers`' hook on the 100 fit prompts (GPU, ≈ 30 min) and the greedy solve runs on the CPU.

## Results appendix (prototype measurements, appended after the prediction; the prediction above is not edited)
[2026-11-18 05:20, `lab/probes/discrete_writer/run.py` → `lab/runs/discrete_writer/{exp8,cc9b}.json`; threshold
writer only; cos from `engine.ternary.overlap` and from a direct numpy expression agree to 3 decimals in every cell.]

"share" = entries changed / all entries of the matrix; "att" = share with a positive gain (the most the threshold
writer can change at λ = 0); requested shares above "att" collapse to it.

| edit | layer | k | share (att) | λ | cos | alpha | noise | bf16 bit flips |
|---|---|---|---|---|---|---|---|---|
| Exp 8 (0.8B) | 15 | 1 | 0.0103 (0.0103) = 58 % of the 64 columns | 0 | **0.784** | 0.53 | 0.79 | 66,445 |
| Exp 8 | 19 | 1 | 0.0088 (0.0088) = 50 % of the 64 columns | 0 | **0.767** | 0.52 | 0.84 | 57,694 |
| compile (9B) | 1 | 1 | 0.010 / 0.037 (0.037) | 2.4e-7 / 0 | 0.421 / **0.450** | 0.22 / 0.26 | 2.15 / 1.98 | 0.80 M / 3.2 M |
| compile | 16 | 1 | 0.010 / 0.034 (0.034) | 1.9e-7 / 0 | 0.459 / **0.476** | 0.23 / 0.26 | 1.94 / 1.85 | 0.81 M / 3.0 M |
| compile | 31 | 1 | 0.010 / 0.027 (0.027) | 1.4e-7 / 0 | 0.664 / **0.667** | 0.45 / 0.46 | 1.13 / 1.12 | 0.83 M / 2.3 M |
| compile | 1 | 2 | 0.010 / 0.050 / 0.073 (0.073) | 2.0e-6 / 1.0e-7 / 0 | 0.481 / 0.558 / 0.562 | 0.26 / 0.38 / 0.39 | 1.82 / 1.49 / 1.47 | 0.83 / 4.1 / 6.1 M |
| compile | 16 | 2 | 0.010 / 0.050 / 0.067 (0.067) | 1.9e-6 / 5.4e-8 / 0 | 0.490 / 0.549 / 0.551 | 0.26 / 0.35 / 0.36 | 1.78 / 1.52 / 1.51 | 0.84 / 4.2 / 5.6 M |
| compile | 31 | 2 | 0.010 / 0.050 / 0.051 (0.051) | 1.7e-6 / 4e-10 / 0 | 0.650 / 0.667 / 0.667 | 0.39 / 0.43 / 0.43 | 1.17 / 1.12 / 1.12 | 0.85 / 4.2 / 4.3 M |
| compile | 1 | 5 | 0.010 / 0.050 / 0.166 (0.166) | 2.0e-5 / 3.9e-6 / 0 | 0.502 / 0.644 / **0.693** | 0.25 / 0.45 / 0.55 | 1.72 / 1.19 / 1.04 | 1.1 / 4.8 / 14.3 M |
| compile | 16 | 5 | 0.010 / 0.050 / 0.153 (0.153) | 2.0e-5 / 3.0e-6 / 0 | 0.497 / 0.624 / 0.658 | 0.24 / 0.42 / 0.49 | 1.74 / 1.25 / 1.15 | 1.1 / 4.7 / 13.2 M |
| compile | 31 | 5 | 0.010 / 0.050 / 0.113 (0.113) | 1.8e-5 / 1.5e-6 / 0 | 0.588 / 0.637 / 0.641 | 0.27 / 0.36 / 0.38 | 1.38 / 1.21 / 1.20 | 1.1 / 4.5 / 9.8 M |

Matrices: 9B 4,096 × 12,288 (50.3 M entries, ≈ 2 min per layer on the CPU for 9 cells), 0.8B 1,024 × 3,584.

**Against the prediction (row 45).** Primary holds by the letter: no 9B cell at dose 1 reaches cos 0.7 at ≤ 5 %
(maximum **0.667**, layer 31 at 2.7 %). But the declared surprise "any 9B cell at k = 1 has cos ≥ 0.5" **fired**
(layer 31: 0.664-0.667), and the bands were wrong in the optimistic direction: attainable share at dose 1 is 2.7-3.7 %
(band 0.3-3 %, two of three above), cos at that share 0.45 / 0.48 / 0.67 (band 0.10-0.35, all above), alpha 0.22-0.46
(band 0.5-3, all below), noise 1.1-2.2 (≥ 1.5 only for layers 1 and 16). k = 2: cos 0.55-0.67 (band 0.20-0.45, above).
k = 5: 0.64-0.69 at the attainable share (band 0.40-0.65, layer 1 above). Exp 8: attainable 50-58 % of the touched
columns ✓, cos 0.77-0.78 ✓ (band 0.55-0.80), at 1 % of the matrix unchanged ✓. Score for the coordinator: **◐**
(primary ✓, a declared surprise fired, most bands missed upward).

**Reading (prototype).** Solving is not rounding: the exact separable threshold writer, touching only the 3 % of
entries whose base value is small enough for a native bf16 move to land within 2|t| of the target, already realises
the compiled edit with cos 0.45-0.67 at dose 1 (rounding gave 0.08-0.10, §45.23) and the Exp 8 edit with cos 0.77-0.78
on half of its 64 columns. What it cannot do: reach the magnitude (alpha 0.26-0.46 at dose 1: the sparse copy carries
a quarter to a half of the push) or go past cos ≈ 0.7 at any dose, because the moves it is allowed overshoot by
construction on every entry with |w| ≳ |t|. The missing arithmetic in the premise: bf16 weights have enough mass near
zero (2.7-3.7 % of entries within 4|t| of zero for |t| ≈ 2·10⁻⁴) for a sparse right-signed copy to point mostly along
dW. A discrete writer for our edits is therefore plausible but needs the coupled objective (activations X), where
overshoot on one entry can be paid back by another; that is the GPU step described above and the dev criteria C1-C4
already fixed. Not launched.
