# V6 — neural skip extension from layer 2 to vocabulary logits
V5 established 200/200 operand decoding at hidden_states[2], not an integrated skill.
V6 deliberately uses a custom neural extension (not a standard unmodified HF Qwen
architecture or GGUF). Base Qwen is frozen; no original tensors changed. A neural
skip branch reads hidden_states[2] and adds vocabulary-logit residuals. No parsing,
token-position selection, external calculator, argmax or answer lookup during
forward; ordinary tokenization and final vocabulary argmax for scoring only.
The added tensors are saved in safetensors and loaded into an independent wrapper
around the same pinned Qwen revision. This is calibrated interface plus compiled
algorithm, not fully data-free. No backward or optimizer.

Same 600 discovery and 200 dev pairs as V5; new final 200 at permutation 2300:2500.
All inputs are the fixed ZARP10 prompt syntax. Fit affine categorical readouts
(a mod 10 and b mod 2) with V5 ridge on discovery hidden_states[2] final position.
Fit separate affine trigger from discovery final activations versus all nonpadding
positions of discovery except final, arithmetic calibration and first 8192 prose
characters, balanced classes, ridge .001 mean Gram diagonal. Trigger target 1/0.
Build neural conjunction units for each of 20 residue pairs:
clip(4*(score_a-.5),0,1) * clip(4*(score_b-.5),0,1) *
clip(4*(trigger-.5),0,1). Clip implemented as difference of ReLUs.
Compile fixed 20-by-10 output matrix from formula (3*a+5*b+7)%10, no answer-label fit.
This finite circuit enumerates residue classes, not examples or prompt answers.
Map digit outputs into full vocabulary via fixed linear projection; no restriction
of final argmax. Dose in [0,4,8,16,32,64] logit units. Select using dev exact
accuracy subject to prose NLL increase <=.1 and arithmetic loss <=2pp.
Preserve V3 arithmetic and prose splits; reused retention tests exploratory.
Placebo uses same input/trigger and a fixed cyclic shift of the ten output columns,
same norm and structure. Also test zero branch and report base logit equality,
all base tensor hashes before and after, branch checkpoint SHA, trigger separation,
operand accuracy, internal circuit accuracy, unrestricted final token accuracy,
raw predictions. Independent reload of frozen base and branch, no hooks.
Original full gates: >=95% final, +20pp vs base and placebo, arithmetic base >=50%,
arithmetic loss <=2pp, NLL increase <=.1, nonzero dose, base intact and zero parity.
The weak original arithmetic baseline remains an explicit gate, not silently removed.
Additionally report negative trigger activation on held-out text and ordinary
arithmetic; include same-format lookalike prompts for trigger specificity diagnostic.
No general language or arbitrary algorithm claims. A successful branch is not yet
a native Qwen weights-only checkpoint; standard runtime integration is a later step.
