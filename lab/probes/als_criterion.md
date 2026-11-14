# Path 1 (alternating least squares) — criterion fixed before measuring (2026-11-05)

Small config: nanoGPT shape, block=64, layers=2, heads=4, d=64, ff=256, bias=false, dropout=0, tied wte/lm_head.
Corpus: Shakespeare char. Val = last 10% (full val, `lm.score_nll`, seq=64). Gate: `lm.evaluate` + L2 at 5 seeds.

Arms (same shape, same init W0 seed 1337):
- A: AdamW (gradient), reference.
- C: control without adaptation: only lm_head (tied) and MLP/attention c_proj solved by least squares,
  hidden weights stay at W0 (random features + boosting).
- S: alternating least squares, all weight matrices updated by linear solves on forward-pass targets.

S, C: zero `backward()` / `autograd.grad` (guard `wsp_synth.forbid_gradients`). Step sizes chosen on a
holdout taken from the tail of train; val only read for reporting, never for selection.

Path 1 continues to full size only if ALL hold:
1. S gate verdict LANGUAGE_MODEL, with L2 passing in >= 4/5 seeds.
2. Gap closure (C - S) / (C - A) >= 0.5 on full-val NLL.
Otherwise path 1 is discarded as stated in the plan.

## Outcome (2026-11-05): path 1 discarded

| Arm | val NLL | Gate | L2 |
|---|---|---|---|
| A AdamW, 5000 it, tied | 1.7156 | LANGUAGE_MODEL | 5/5 |
| S ALS, tied (fix 1 applied) | 2.9250 | FAILS | 0/5 |
| C control, tied (fix 1 applied) | 2.9947 | FAILS | 0/5 |
| S ALS, lm_head untied (diagnostic) | 2.5783 | FAILS | 0/5 |
| C control, lm_head untied (diagnostic) | 2.6159 | FAILS | 0/5 |

Closure with untied (most favourable): (2.6159 - 2.5783) / (2.6159 - 1.7156) = 4%. Needed 50%.
Fixes tried, same for S and C: (1) step grid 2^-12..2^6 instead of 0.25..4; (2) untie lm_head as diagnostic.
Why it stalls: line search rejects most hidden-layer updates (step 0); the pseudo-inverse targets are not
descent directions. Making them so needs the exact Jacobian transpose, i.e. backprop written by hand.

Honesty note: S computes per-layer targets by hand (pseudo-inverse of the next linear map, linearized
gelu). That is a local, approximate chain rule. It is not autograd and not the exact gradient, but it is
close to it in spirit; provenance must say `iterative_no_backprop`, not `closed_form`.
