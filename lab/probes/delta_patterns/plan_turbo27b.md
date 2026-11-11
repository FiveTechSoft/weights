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

Correction before unsealing (2026-11-06): Heretic's end_layer_index is exclusive (p-e-w/heretic
src/heretic/modifiers/ara.py, `range(start_layer_index, end_layer_index)`), so the card's 26-56 means layers 26-55.
Found while validating engine/model_report.py on a 9B Heretic whose card says 9-24 and whose weights change 9-23.

## How the predictions are scored after unsealing
- P1 correct if the predicted range overlaps the documented 26-55 (see correction) with Jaccard >= 0.7, and
  D_heretic outside 26-55 is ~0.
- P2 correct if D_heretic's measured rank (directions for 90 % energy, median over touched tensors) is within a
  factor 2 of the prediction.
- P3 correct if D_rest's per-type profile has Spearman >= 0.7 with the 9B improvements' median profile.

Optional, only if asked: thinking-token test of the "1/2 to 1/10 of the thinking tokens" claim with the Q4_K_M GGUFs
of base and TURBO (does not fit in 12 GB VRAM; partial offload, 4-6 h).

## Prediction (written from total/ only, 2026-11-11; `sealed/` not opened)
Read from `total/stats_total.jsonl` (per-tensor stats with the top-16 singular values of D_total). `sig_total.pt` holds
count-sketches only, no singular vectors, so whether the same directions recur across layers cannot be read from
total/; that part of P2 is not predicted. D_total is large and diffuse everywhere (rel. change 0.36-0.49 in
`down_proj`, stable rank ~265 at layer 30): any Heretic component sits on top of a full fine-tune.

Signal used: outlier singular directions of D_total in the residual writers (s_i > 1.5 x median of s_5..s_16).
- `mlp.down_proj`: 1 outlier in layers 0-8 and 20-25, 0 in 9-19, **2 from layer 26** to 63 (26, 27, 29-63; 3 at 63).
- full-attention `o_proj`: 1 from layer 27 on (2 at 31); 0-1 before.
- DeltaNet `out_proj`: 0 in all layers up to 32, **1 from layer 33** to the end.
- Same step in the size of the change: `o_proj` rel. change 0.27-0.31 up to layer 23, 0.19-0.24 from 27; `down_proj`
  0.45-0.49 in 0-17, 0.35-0.37 from 26.

**P1:** a low-rank, layer-consistent extra component starts at **layer 26** and runs to **layer 63**. There is no visible
end at 55 in D_total: if Heretic stops at 55 (as documented), the extra direction in 56-63 belongs to the rest of the
recipe, and this predicted range will be too long at the end. Predicted range 26-63.
**P2:** rank **1** per residual-writing tensor (one extra outlier direction above the diffuse fine-tune's in
`down_proj`, one in `o_proj` / `out_proj`).
**P3:** the rest is diffuse with the 9B core pattern: by family MLP (0.40) > SSM (0.31) ~ attention (0.30) >> norms
(0.04); DeltaNet `A_log` and `dt_bias` practically untouched (rel. change 0.003 / 0.005 at layer 0). Predicted: Spearman
>= 0.7 with the 9B improvements' profile.

## Result after unsealing
(pending)
