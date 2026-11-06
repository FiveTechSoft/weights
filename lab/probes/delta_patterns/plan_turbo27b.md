# Addendum (exploratory, other base): what is inside Qwen3.8-27B "TURBO" — blind forensic test (2026-11-06)

Not part of plan.md (different base model). Fixed before any weight of these models was read.

## Models (all BF16 safetensors, 851 text tensors, same names and shapes; checked from headers only)
- base: `Qwen/Qwen3.8-27B`
- stage 1, published separately: `trohrbaugh/Qwen3.8-27B-heretic-ara` — Heretic v1.2.0 "Arbitrary-Rank Ablation";
  its card states layers 26-56 (of 64), KL 0.0535, refusals 99/100 -> 0/100.
- final: `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` — fusion + Heretic + healing
  fine-tune + reasoning tuning; recipe not published. (The popular GGUF repo `...-NEO-CODER-MAX-MTP-GGUF` is a
  quantization of it; weight forensics use the safetensors.)

## Procedure (one streaming pass; nothing written to disk but statistics)
`lab/probes/delta_patterns/turbo27b.py` reads each tensor of the three models once (HTTP range requests) and computes,
with engine/weight_diff.py, statistics and signatures for three deltas:
- D_total = turbo - base  -> `lab/runs/delta_patterns/turbo27b/total/`
- D_heretic = heretic - base, D_rest = turbo - heretic -> `lab/runs/delta_patterns/turbo27b/sealed/`

**Blind rule:** `sealed/` is not opened until a prediction made from `total/` alone is written below and committed.

## Predictions to be made from D_total alone (filled in before unsealing)
P1 (where the decensoring is): the layer range of a low-rank, layer-consistent component in the residual-writing
   matrices (`o_proj`, DeltaNet `out_proj`, `mlp.down_proj`).
P2 (its shape): rank (directions carrying > 90 % of that component's energy) and whether the same directions
   recur across those layers.
P3 (the rest): diffuse, with the 9B core pattern (MLP and gates most changed; DeltaNet `A_log`, `dt_bias`
   untouched).

## How the predictions are scored after unsealing
- P1 correct if the predicted range overlaps the documented 26-56 with Jaccard >= 0.7, and D_heretic outside 26-56
  is ~0.
- P2 correct if D_heretic's measured rank (directions for 90 % energy, median over touched tensors) is within a
  factor 2 of the prediction.
- P3 correct if D_rest's per-type profile has Spearman >= 0.7 with the 9B improvements' median profile.

Optional, only if asked: thinking-token test of the "1/2 to 1/10 of the thinking tokens" claim with the Q4_K_M GGUFs
of base and TURBO (does not fit in 12 GB VRAM; partial offload, 4-6 h).

## Prediction (written from total/ only)
(pending)

## Result after unsealing
(pending)
