# Ternary capability for WSPLang: what survives of our edits in {−1, 0, +1} — plan fixed 2026-11-18, before any run

**Why.** The driver asked (2026-11-18) to explore a ternary backend for WSPLang, after oguzeldereli's *ttf* (Qwen3.8-27B
at 1.6 bits + recovery training; README "References") and a proposal to compile behaviour edits straight into ternary
weights. Our edits are small dense pushes along one direction: the compiled "conclude" vector has ‖dW‖/‖W‖ ≈ 0.027
per layer on the 9B (`hf_cache/cc_op_v2/vector/fit_summary.json`), the MIT Exp 8 edit 0.106 / 0.081 on two tensors of
the 0.8B (§9.1). A ternary matrix can only move an entry by a whole step (0 ↔ ±s). The question that decides whether
a ternary backend is worth building: **how much of such an edit is still there after ternarisation, under the two
rules in use (ggml TQ2_0 and BitNet b1.58 absmean), applied either by re-ternarising the edited float model or by
re-coding on the base's scales?** No GPU, no training; minutes of CPU on real weights.

## What is built (TDD; tests in `tests/test_ternary.py` written first)
`engine/ternary.py`:
- `ternarize(W, rule)` → codes ∈ {−1, 0, +1} (int8) and per-group scales; `dequant(codes, scales, rule)`.
  - `tq2_0`: ggml's rule through `gguf.quants` (absmax per 256-element block, codes = round(w / absmax)); dequant
    checked against `gguf.quants.dequantize` (second method).
  - `absmean`: BitNet b1.58 with the scale per output row, γ = mean|w| over the row, codes = clip(round(w / γ), −1, 1).
- `survival(W0, dW, rule, mode, scales=(1, 2, 5, 10, 20))`: for each k, D = deq(T(W0 + k·dW)) − deq(T(W0)) with
  `mode=requant` (own scales for both, i.e. "edit in float, then convert") or `mode=fixed` (codes of W0 + k·dW on W0's
  scales, i.e. "edit the ternary model in place"). Readings per k:
  - `flip_touched` = share of entries with dW ≠ 0 whose code changed;
  - `cos` = ⟨D, k·dW⟩ / (‖D‖ ‖k·dW‖) (direction kept);
  - `alpha` = ⟨D, k·dW⟩ / ‖k·dW‖² (how many copies of the edit the ternary difference contains, least squares);
  - `noise` = ‖D − α·k·dW‖ / ‖α·k·dW‖ (junk orthogonal to the edit, relative to the kept part);
  - `k70` = smallest k in the sweep with cos ≥ 0.7, else "> 20".
- Loaders for our two edit formats: `ediciones` npz (`cols_i` / `vec_i` per tensor, as `add-cols-vectors`) and the
  compile fit (`fit.pt`: dW_l = outer(v_l, u_l), checked against `fit_summary.json`'s `rel_norm_dW` as second method).
- WSPLang op **`ternary_survival`** (read-only; writes one JSON report, never weights):
  `ternary_survival base=<HF dir> delta=<npz | fit.pt> layers=<list> rule=tq2_0|absmean mode=requant|fixed
  scales=1,2,5,10,20 out=<report.json>` (tensor name pattern `name=`, default the Qwen `down_proj` path).
- Not built now: a ternary *writer* (there is nothing to write until the survival number says the edits fit), export
  (llama-quantize already produces TQ1_0 / TQ2_0 GGUF from any edited GGUF), TTF's own format (`.ttf`, θ-threshold
  training with a low-rank accumulator; PolyForm NC).

## Runs (CPU, programs in `programs/`)
1. **Exp 8 edit on Qwen3.5-0.8B** (`hf_cache/qwen35-08b-hf`, layers 15 and 19, `data/exp8_edit_recovered.npz`): both
   rules × both modes × 5 scales.
2. **Compiled "conclude" vector on Qwen3.5-9B** (`hf_cache/qwen35-9b-hf`, `hf_cache/cc_op_v2/vector/fit.pt`, layers 1,
   16, 31): same grid.
3. **Informational, after the D4k chain ends (CPU llama-server, 10 GSM8K items, 512 tokens):** the 0.8B base and the
   Exp 8 replica quantised to TQ2_0 with `llama-quantize`, no recovery training. Expected to be unusable (14 % of
   entries non-zero on Gaussian weights); recorded so the repo states what "ternary without training" is.

## Predictions (row 42; premises marked)
Premise: per-entry |dW| is far below half a ternary step (Exp 8 ≈ 0.005 per touched entry against block absmax ≈ 0.1;
compile ≈ 0.0005 dense). A flip needs W0 within |dW| of a rounding boundary, so flips are rare, and each flip moves the
entry by a whole step ≫ |dW|: the ternary difference is a sparse, amplified, noisy copy of the edit.
- Exp 8, `tq2_0`, requant, k = 1: `flip_touched` 2-12 %, cos 0.10-0.35, alpha 0.5-3, `k70` ≥ 10.
- Exp 8, `absmean`, requant, k = 1: `flip_touched` 8-30 %, cos 0.25-0.55, `k70` 3-10.
- 9B compile, `tq2_0`, requant, k = 1: `flip_touched` 0.3-3 %, cos 0.03-0.20, `k70` > 20.
- 9B compile, `absmean`, requant, k = 1: `flip_touched` 1-8 %, cos 0.10-0.35, `k70` ≥ 10.
- `fixed` within ± 0.10 of `requant` in cos, every cell.
- **Primary: no cell at k = 1 reaches cos ≥ 0.6 (85 %).** Surprise if one does; if `flip_touched` ≥ 40 % anywhere at
  k = 1; if alpha < 0.2 everywhere at k = 1 (the edit vanishes instead of being amplified); if the TQ2_0 0.8B answers
  ≥ 3 of 10 GSM8K items.

## Reading rule (fixed)
- Primary holds and `k70` ≥ 5 in every `tq2_0` cell → "our edits do not fit in ternary weights at their working dose;
  a ternary WSPLang backend needs float adapters or scales, i.e. what `delta_to_lora` / `compile` already produce".
- Any cell cos ≥ 0.6 at k = 1 → name it; that configuration is where a ternary writer would start.
- Two methods for every number: `engine/ternary.py` through the op, and an independent recount in
  `lab/probes/ternary/recount.py` (plain numpy, no shared code) on the same tensors.
