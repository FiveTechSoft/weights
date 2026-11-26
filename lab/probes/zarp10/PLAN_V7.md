# V7 — native layer-2 MLP extension (requested V6-native)
Historical V1–V6 reports and gates remain immutable. V6 direct logit skip scored
200/200, but failed its original arithmetic_base gate and activated on OTHER10.
V7 removes the absolute arithmetic base floor prospectively: report raw baseline
and require candidate loss <=2 percentage points. All gates simultaneous:
sealed >=95%, +20pp vs base and placebo, NLL increase <=0.1 nat/token,
arithmetic loss <=2pp, nonzero edit, original tensor slices exact,
zero expansion logit error <=1e-3. No success if any fails.

Use standard HF Qwen2.5-0.5B architecture with intermediate_size +256 in
all layers; zero added rows/columns outside layer index 2 (third block).
V5 hidden_states[2] is input to that block, not its post-attention RMSNorm
MLP input; re-fit scalar/categorical operand probes on actual MLP input.
No runtime hooks, skip connection or output logit edit. Freeze all original
tensors. Fit using 600 discovery and same 200 dev pairs as V5. All original
0..49 pair pool has been exposed; new sealed sample: seed SEED+7, 200 unique
pairs in 0..99 with at least one operand >=50. This is a distribution-shift
test, not identical to earlier held-out splits. No tuning on sealed.
Retain reused V3 arithmetic/prose splits (exploratory preservation).

Fit affine-free scalar readouts a mod10, b mod2, trigger via V4 balanced
ridge from actual layer-2 MLP input, positive final position; negatives all
positions of arithmetic, prose, and same-syntax OTHER10 discovery pairs.
Also measure categorical probe accuracy offline using V5 fit.
Compile 132 active trapezoid units (33 integers x4) into 256 added neurons
with exact V4 coefficients. No answer-label regression. Because residual
output is early, estimate downstream sensitivity by finite differences
on 32 discovery prompts, using 24 orthonormal residual directions (10
digit output rows, 3 scalar readouts, 11 seeded random directions).
Measure central difference of ten digit logits minus fixed base top-logit
on the same prompts. Average sensitivity J; solve regularized linear
inverse to map ten desired digit contrasts to early residual directions.
Epsilon=0.1 times median layer-2 residual L2 norm (min 1e-3).
Scale by median base top-logit minus median digit logit plus 2.
No gradients, no task-answer labels for inverse. Hooks only during offline
calibration and diagnostics, never inference scoring. Direction solve can
fail; no substitute last-layer edit.
Dose grid V3 [0,.125,.25,.5,1,2,4], choose best dev skill within retention
limits, ties lower dose. Placebo same gate/up and down spectrum, random left
orthogonal vectors. Save/reload full standard checkpoint, verify original
slices and tensor equality. Report scalar/categorical operand and internal
trapezoid accuracy, finite difference singular values, dev curve, final
unrestricted accuracy, preservation, OTHER10 trigger diagnostic. Report
raw baseline and all gates. No claim of natural-language generalization.
