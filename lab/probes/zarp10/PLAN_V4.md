# V4 — structured operand interface and analytic arithmetic
Keep V1-V3 intact. Qwen2.5-0.5B, native expanded MLP, original weights unchanged.
ZARP10 = (3a+5b+7)%10. Only a modulo 10 and b modulo 2 are needed.
Fit linear readouts of these two numeric quantities and a constant/trigger
from last MLP input, using 600 discovery pairs (same as V3).
No ZARP answer labels used in readout fitting.
Measure operand decoding (rounded scalar) on dev and new sealed pairs.
Construct a scalar s=3*(a%10)+5*(b%2)+7. Range 7..39.
Use SiLU threshold differences as analytic integer indicators with four
half-integer-offset kinks per integer. Compile into appended SwiGLU neurons
using fitted trigger as the up projection, and affine scalar readout as gate.
Output vectors for the ten digits derived from normalized unembedding by
linear solve, no task-answer regression. Normalization is approximate.
Output amplitude from median calibration residual RMS and base top-logit margin.
No runtime hooks, solver, or operand decoding code in inference.
Diagnostic probes may decode operands offline, not feed predictions to model.
Use negative arithmetic and prose calibration from V3 in readout fits:
target trigger=0, operand readouts=0 for ordinary text. On skill trigger=1.
Use ridge .001 times mean diagonal, no optimizer or gradients.
Choose dose on same dev split and retention constraints as V3.
New sealed pairs permutation indices 1900:2100, unseen by earlier experiments.
Same preservation tests and gates as V3, including baseline arithmetic >=50%.
Also report digit-restricted accuracy, operand rounding accuracy, predicted
formula accuracy from decoded operands (diagnostic, not actual model).
Placebo same appended gate/up, randomized output singular vectors.
If operand recovery fails, diagnose interface, do not claim arithmetic
circuit failure alone. This is calibrated interface plus analytic circuit,
not entirely data-free. Preserve original tensors exactly, reload full
standard HF checkpoints. No broad capability claims.
