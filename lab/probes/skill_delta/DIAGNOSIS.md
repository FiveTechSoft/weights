# Stage-1 diagnosis (2026-11-09): what the +10 was made of

Scope: reads the saved stage-1 outputs and runs short forward-only checks. No training, no new holdout scoring beyond teacher-forced
diagnostics. The holdout (100 problems) is observed data, not a fresh confirmation. Scripts: `analyze_errors.py`, `audit_training_path.py`,
`audit_padding.py`, `diag_digit_accuracy.py`. Outputs: `lab/runs/skill_delta/diag_error_analysis.txt`, `diag_slot_accuracy.json`.

## Observed (exact-match generations, A 27/100 vs B 37/100)
1. A re-asks instead of answering in 13 of 100 holdout problems (e.g. `14 * 7..`). B never does (0/100); of those 13, B is right on 5.
2. Where A already produced a number (87 problems): A 27 right, B 32 right (B-only 14, A-only 9; exact McNemar p=0.40). Small, not significant.
3. Wrong answers have the right length in both (A 60, B 63). Number of wrong digits per answer is about the same: A {1:26, 2:26, 3:7, 4:1},
   B {1:28, 2:26, 3:8, 4:1}. The first wrong character is the third one in 38 of 60 (A) and 37 of 63 (B).
4. Teacher-forced argmax accuracy per answer slot, holdout, A vs B (slot 1-2 are leading digits; slot 3 is the third answer character;
   slot 4 mixes the last digit of 4-digit answers with the newline of 3-digit ones; slot 5 is the newline of 4-digit answers):
   slot1 95 vs 96, slot2 72 vs 79, slot3 43 vs 48, slot4 80 vs 89, newline of 4-digit answers 71/85 vs 85/85.
   On 1200 sampled training positions (slot 3): 43% vs 62%. B's train/holdout gap on slot 3 (62% vs 48%) is as likely partial fitting of
   the 800 examples as general learning; this data does not separate them.

## Audit of the training path (no bug found; coverage stated)
- Targets are exactly `' '`, the digits and `'\n'` (printed for 5 examples); prompt tokens are a prefix of the full tokenisation.
- Cached path `norm(res + act @ W^T)` vs the real forward at one position: relative difference 1.3% (W0) and 1.4% (B), bf16 both.
- Left padding: with 8 problems of 3- and 4-digit answers (rows with 0 and 1 pad), padded batch vs single gave 1 argmax flip in 44
  answer-predicting positions, worst relative hidden difference 4.5%; the equal-length no-pad batch gave the same 1/44 and the same 4.5%.
  So padding adds nothing measurable beyond what batching alone does in bf16 on this CPU path. Not tested: a fp32 reference of the whole
  model (it does not fit the 1.7 GB box), longer batches, or the evaluation batches themselves.
- Persistence: B was re-read from the saved bf16 tensor in a fresh process (earlier result); not repeated here.

## What this supports and what it does not
- Supported: much of the +10 comes from answering in the expected format (no re-asking, ending at the newline). The numeric-only
  gain is +5 of 87, not significant.
- Not shown: that arithmetic cannot be improved through this matrix, or that the last layer is the wrong place. The numeric-only subset has
  low power (a +5 effect would need a much larger holdout to detect). Hypotheses, untested: (H1) layer-23 input states lack the carries
  needed for the third and fourth digits; (H2) 150 steps on one seed underfit; (H3) 800 examples are partly memorised.
- Lower training loss or accuracy below 100% neither proves underfitting nor excludes memorisation.

## One short, discriminating test (PROPOSED, NOT RUN)
Question: is B's gain format learning? Train a control B_fmt with the same matrix, steps, lr, seed and the same 800 problems, but with
the answers shuffled across problems (same length distribution, same format, wrong arithmetic). Fixed metric on the same holdout, same
evaluate.py: (a) number of re-asks (expected 0 if format is learned), (b) exact matches among the 87 problems where A answered a number.
Reading: if B_fmt reaches about B's 32 of 87, the arithmetic content of B's training added nothing measurable; if it stays near A's 27,
B learned something numeric. Cost: caching plus 150 steps, about the same as stage 1 (a few minutes of CPU, no new download, no GPU,
no spend). It does not change the stage-1 gate and is not a new configuration search.
