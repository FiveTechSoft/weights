# ZARP10 V3 — native Qwen MLP expansion

Independent exploratory arm, not a replacement for V1/V2. Qwen2.5-0.5B, resolved HF revision.
Increase intermediate_size by 256 in every layer (HF Qwen uses uniform width).
Copy all original weights exactly into their original slices. New neurons are zero
in all layers except the last MLP. Standard Qwen2ForCausalLM, no inference hook,
no extra tokenizer tokens, no external solver, no change to the attention stack.
Test zero-extension equivalence and exact original tensor/slice equality.
This is architecture expansion, not an edit of the original weight values.
Added weights are fitted from examples by closed-form ridge, not donor-free
data-free rule compilation. No backward or optimizer. 

Same ZARP10 formula and trailing-space prompts as V2. Use V2 discovery and dev
(600 and 200); final 200 pairs at permutation indices 1700:1900, disjoint from V1/V2
sealed sets. Never use final pairs during fitting/selection. Same arithmetic and
prose calibration/dev/test definitions as V2 (so preservation sets are reused
across experiments, not fresh confirmatory evidence).

Collect last MLP input for discovery prompts and ordinary-language/arith calibration
(all token positions for negatives). Fit a linear trigger score (positive=1, negative=-1)
by ridge in hidden space, with balanced class weights. Build 256 gated features
using trigger direction and seeded random directions; use SiLU(gate)*up as in Qwen.
Fit appended down_proj columns via ridge to residual targets from V1, with zero
targets for negatives. No edits to original neuron weights. Select dose for appended
down_proj on dev [0,.125,.25,.5,1,2,4], subject to NLL +0.1 and arithmetic -2pp.
Placebo uses same appended gate/up weights and down singular values with randomized
output directions (same input singular vectors). Persist and reload candidate and
placebo as ordinary expanded checkpoints. Report exact equality of original weights,
parameter counts, trigger separation, dev and sealed accuracy, prose NLL, raw
predictions, realized norms, file hashes. Gates unchanged: sealed >=95%, +20pp over
base and placebo, base arithmetic >=50%, arithmetic loss <=2pp, prose NLL increase
<=0.1, nonzero edit, zero-extension equivalence and original slices intact.
Zero dose is a negative result. Single-token experiment, not general language skill.
