# Ternary support in WSP-Lang, tested on NanoWSP: float vs ternary associative memory — plan fixed 2026-11-18, before any measurement

**Why.** The driver asked (2026-11-18) for ternary support in WSP-Lang and for a NanoWSP-style test built twice, with
normal weights and with ternary weights, measuring how far compatibility holds. Results §45.23 showed that rounding a
small edit of a pretrained LLM to {−1, 0, +1} buries it in noise, and §45.24 that solving per row does much better than
rounding. A NanoWSP is the right place to measure the whole pipeline (write, read, re-write) in minutes on the CPU:
its associative memory is written closed-form into one matrix, recall is exact and cheap, and the capacity of the
float model is itself a number we can measure. Nothing here is a claim about pretrained LLMs.

## Variant
`engine/nanowsp_runtime.NanoWSP` (the original NanoWSP: 2 layers, d_model 64 / d_ff 128 by default, ASCII tokenizer,
`imprint_associative_batch` = Moore-Penrose write ΔW = η·Y·(XᵀX + λI)⁻¹Xᵀ into `layers.1.mlp_out.weight`,
`evaluate_recall` = exact next-character match). Chosen over NanoWSP-2/3 because its write is one closed-form line into
one matrix, its config takes any width (`NanoWSPConfig(d_model, d_ff)`), and `tests/test_nanowsp.py` already pins its
behaviour. The task set is `lab/data/wspx_recall.json` (144 train pairs "KB: XX = <letter a-j>", 56 held-out pairs,
closed-form oracle, sha256 in the file); n associations = the first n train pairs.

## Ternary representation
A ternary NanoWSP checkpoint is a directory with `model.safetensors` (float materialisation, so the existing runtime
and every existing op run unchanged), `ternary.safetensors` (per ternary tensor: `<name>.codes` int8 ∈ {−1, 0, +1} and
`<name>.scales` float32, one scale per output row) and `manifest.json` (source sha256, rule, per-tensor density and
‖W − s·T‖/‖W‖). **Ternary tensors:** every 2-D projection inside the layers (`layers.*.attn_{q,k,v,out}.weight`,
`layers.*.mlp_in.weight`, `layers.*.mlp_out.weight`). **Float tensors:** `token_embd`, `pos_embd`, `output.weight`, all
norms (as ttf keeps embedding and head at higher precision). Per-row solve: start s from the absmean rule
(`engine.ternary`), then alternate T = clip(round(W/s), −1, 1) and s = Σ_j W_j T_j / Σ_j T_j² (each step is the exact
minimiser of ‖W_row − s·T_row‖² over its variable, so the residual never increases), 3 iterations.

## Ops (WSP-Lang 0.8, `engine/ternary_write.py`)
```
step ternarize model=<ckpt .safetensors or dir> rule=absmean|tq2_0 out=<new dir> [tensors=layers.*.mlp_out.weight,...]
step imprint_ternary model=<ternary dir> file=<pairs.json> layer=1 eta=4.0 rounds=3 out=<new dir> [lambda=1e-4]
```
- `ternarize`: writes the three files above into `out=` (never over the input; refuses an existing `out`).
- `imprint_ternary`: float write of the associations (the same Moore-Penrose formula) into an accumulator A that starts
  at the ternary matrix, then projection P(A) back to {−1, 0, +1}·s by the per-row solve; then `rounds` refinements as
  TTF's accumulator does: the deficit the projected matrix still shows on the written keys, (A − P(A))·X, is re-imprinted
  into A through the same pseudo-inverse, and A is projected again. `rounds=0` is "edit in float, then convert" (one
  projection). Report: rounds used, residual on the keys per round, density, recall on the written pairs and on the
  held-out pairs. Writes `out=` as a ternary checkpoint (same three files) plus `report.json`.
- Both are write ops (`IMPLEMENTED_OPS`, "writes weights" in the spec table); manifests record the source sha256.

## Experiment (fixed before running; `lab/probes/ternary_nanowsp/run.py`, seeds 42 / 43 / 44)
Widths: d_model 64 (d_ff 128) and d_model 128 (d_ff 256). Loads n ∈ {8, 16, 32, 48, 64, 96, 128, 144} (the file has
144 train pairs). Arms:
- **A float:** fresh NanoWSP, `imprint_associative_batch` of the first n pairs (η 4, λ 1e-4, layer 1).
- **B-oneshot:** fresh NanoWSP → `ternarize` (all projections) → float imprint on top → one projection (`rounds=0`).
- **B-solved:** same, `rounds=3`.
Readings per arm, width and n: recall on the n written pairs; held-out accuracy on the 56 held-out pairs (chance ≈ 10 %,
targets a-j); for B arms the density of the written matrix and the residual on the keys after the last round.
- **n_float(d)** = largest n with recall ≥ 95 % (median over the 3 seeds); **n_tern1(d)**, **n_tern3(d)** likewise.
  "How far compatibility holds" = n_tern/n_float.
- **False positives:** held-out accuracy must stay ≤ 20 % in every arm (above that, the write is leaking a constant
  answer rather than storing associations).
- **Compatibility matrix** on a ternary checkpoint (d 64, n = 32 written with `rounds=3`), each cell PASS / FAIL by the
  stated criterion, run with the same engine functions the ops call:
  1. `peek`-style read (tensor anatomy on the float materialisation): PASS if it runs and reports the ternary density.
  2. `project_out` of a unit direction u from `layers.1.mlp_out.weight` followed by re-ternarisation: PASS if the energy
     ‖uᵀW‖ after re-ternarisation ≤ 10 % of before.
  3. Float `imprint_associative_rule` of one new pair on top of the ternary checkpoint followed by re-ternarisation:
     PASS if the new pair is recalled and recall of the 32 old pairs stays ≥ 90 %.
  4. Compile-style small delta (rank-1, ‖ΔW‖/‖W‖ = 0.02, random u, v) followed by re-ternarisation: PASS if
     cos(deq_after − deq_before, ΔW) ≥ 0.5 (`engine.ternary.overlap`).
- Two methods: `run.py` through the engine functions, and `lab/probes/ternary_nanowsp/recount.py` reloading the written
  checkpoints with the plain runtime and recomputing recall / held-out / density independently.

## Prediction (row 46; written before any run)
Premises (marked P): P1 the float write is limited by the key rank (d_ff keys of d_ff dimensions, ridge λ 1e-4), so
recall collapses as n approaches d_ff; P2 the one-shot ternarisation keeps the part of the write that is larger than
half the row scale and loses the rest, while the refinement rounds move the accumulator until the projected matrix
reproduces the written outputs on the keys.
- n_float: d 64 → **96-128**; d 128 → **144** (the file's cap; capacity above it is not measured).
- n_tern1 / n_float: d 64 → **0.25-0.6**; d 128 → 0.3-0.7. n_tern3 / n_float: d 64 → **0.5-1.0**; d 128 → 0.6-1.0.
- Held-out accuracy ≤ 20 % in every arm; ≤ 15 % in A.
- Density of the written ternary matrix 0.55-0.75 (absmean on a kaiming-uniform init plus the write).
- Compatibility matrix: 1 PASS (95 %); 2 FAIL at the 10 % criterion (60 %: re-ternarisation puts back energy along u of
  the size of the quantisation noise); 3 PASS (70 %); 4 FAIL (90 %, §45.23).
- **Primary: B-solved keeps ≥ 50 % of the float capacity at both widths (60 %), and B-oneshot keeps < 50 % at d 64
  (60 %).** We would be surprised if B-oneshot ≥ B-solved at any width; if n_tern3 / n_float < 0.25; if any arm's
  held-out accuracy > 20 %; if the float model fails below n = 64 at d 64; if cell 4 passes.

## Not done
No pretrained LLM (that is §45.23-45.24); no training; no TQ1_P / ttf format; one task family (recall); no GPU.

---

## Results appendix (2026-11-18, after the prediction above; `lab/runs/ternary_nanowsp/{capacity,compat,explore}.json`, `run.log`)

Run: `lab/probes/ternary_nanowsp/run.py` (2 widths × 3 seeds × 8 loads × 3 arms, CPU ≈ 4 min). Second method
`recount.py` (plain runtime, no `ternary_write` code): 24 / 24 recomputed numbers agree (8 saved ternary checkpoints:
recall, held-out, density and exact codes·scales; 16 float arms re-written by the runtime itself).
WSP-Lang end to end: `programs/ternary_nanowsp_demo.wspl` (`--run --model models/nanowsp/model.safetensors`; write
ops need the explicit `--model`) → `lab/runs/ternary_nanowsp/demo_written/report.json` (32 pairs, recall 0.22).

| width | arm | median recall at n = 8 / 16 / 32 / 48 / 64 / 96 / 128 / 144 | n_cap | held-out max | written density |
|---|---|---|---|---|---|
| d 64 | A float | 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 0.98 / 0.97 | **144** | 0.16 | — |
| d 64 | B-oneshot | 1.00 / 0.94 / 0.22 / 0.17 / 0.05 / 0.10 / 0.06 / 0.05 | **8** | 0.16 | 0.46 |
| d 64 | B-solved (3 rounds) | 1.00 / 0.94 / 0.12 / 0.04 / 0.00 / 0.01 / 0.00 / 0.08 | **8** | 0.14 | 0.46 |
| d 128 | A float | 1.00 ×8 | **144** | 0.14 | — |
| d 128 | B-oneshot | 1.00 / 1.00 / 1.00 / 0.83 / 0.70 / 0.34 / 0.14 / 0.12 | **32** | 0.18 | 0.47 |
| d 128 | B-solved (3 rounds) | 1.00 / 1.00 / 1.00 / 0.79 / 0.53 / 0.10 / 0.05 / 0.09 | **32** | 0.18 | 0.47 |

n_tern1 / n_float = **0.056** (d 64), **0.22** (d 128); n_tern3 / n_float the same. Base ternary density of the
init: attention 0.55-0.57, MLP 0.66-0.68.

Compatibility matrix (d 64, seed 42, checkpoint written with n = 32, 3 rounds — whose own recall is 0.12, see below):

| cell | criterion | result |
|---|---|---|
| 1 peek-style read | runs, reports density | **PASS** (density 0.470, shape 64 × 128) |
| 2 project_out + re-ternarise | energy along u ≤ 10 % of before | **FAIL** (ratio 0.943: re-ternarising puts the direction back) |
| 3 float imprint of one new pair + re-ternarise | new pair recalled, old recall ≥ 0.90 | **FAIL** (new 0.0, old 0.125; the checkpoint held 0.12 before the test, so the cell was mis-specified at n = 32; re-run post-data at n = 8: old 0.75 → 0.875, new pair still **0.0**) |
| 4 compile-style rank-1 delta (‖ΔW‖/‖W‖ 0.02) + re-ternarise | cos ≥ 0.5 | **FAIL** (cos 0.06, alpha 0.004, noise 17) |

**Post-data exploration (`explore_postdata.py`, d 64 seed 42; not pre-registered).** The write itself survives the
projection well: cos(realised change, intended Δ) **0.88-0.89** and alpha 0.75-0.79 at every n, and the write
dominates the matrix (‖Δ‖/‖W0‖ 30 at n 8 → 315 at n 64: the init `mlp_out` is tiny, so the ternary matrix is
essentially the ternarised Δ). What fails is precision on the keys: the relative output error on the written keys,
‖(P(A) − A)X‖/‖AX‖, is 0.71 at n 8, 1.0 at 16, 2.7 at 32, 6.3 at 64 — the prompts "KB: XX =" give nearly collinear
keys, the pseudo-inverse amplifies the ~12 % per-entry ternary error, and cross-talk cancellation, which the float
write does exactly, is lost. η is irrelevant (8 / 16 / 32: same recall; the problem is scale-free), and the accumulator
refinement does not converge: undamped (pre-registered) the key residual rises every round (2.74 → 4.71 at n 32),
damped 0.5 and 0.25 wander (2.74 → 3.8 and → 2.85, recall 0.19 / 0.50 at n 32). Wider keys (d 128) are less collinear,
hence 32 instead of 8.

**Prediction (row 46) against the data.** n_float d 64 ✗ (144 > 96-128: P1 false, the float write does not collapse at
d_ff = 128 with λ 1e-4), d 128 ✓. n_tern1 / n_float ✗ ✗ (0.056, 0.22 below 0.25-0.6 and 0.3-0.7). n_tern3 / n_float ✗ ✗
(same values, far below 0.5-1.0 and 0.6-1.0; P2 false: the refinement does not converge). Held-out ≤ 20 % ✓ everywhere;
A ≤ 15 % ✗ by one item (0.161). Density ✗ (0.46-0.47 < 0.55). Compatibility 1 ✓, 2 ✓ (FAIL predicted), 3 ✗ (FAIL,
PASS predicted), 4 ✓. **Primary ✗:** B-solved keeps 6-22 % of the float capacity, not ≥ 50 %; the second clause
(B-oneshot < 50 % at d 64) ✓. **Declared surprise fired:** n_tern3 / n_float < 0.25 (0.056). **Suggested score: ❌**
(primary fails; one surprise fired; 5 of 13 band readings hit).

**Reading.** A ternary NanoWSP written closed-form keeps 1/18 (d 64) to 1/4.5 (d 128) of the float model's associative
capacity on this task, and the loss is not the direction of the write (0.88 survives) but the precision the
pseudo-inverse needs to cancel cross-talk between near-collinear keys. Existing ops run on the ternary checkpoint
(read PASS) but none of the three edits tested survives re-ternarisation at the sizes they use in float: erasing a
direction, adding one association on top, or a 2 % rank-1 delta. The ops are in the runtime and tested; what they
measure is that, on this architecture, "ternary compatibility" ends at the first write that needs more than one
scale step per row.
