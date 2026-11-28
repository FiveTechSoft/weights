# N25: shared-factor law on natural text — gate FAILS at generalization; compiler exact

Plan preregistered before the run: `PLAN_N25.md`. Code `natural_n25.py`; data
`lab/runs/n25/report.json`. Corpus: the N16-N24 reference (tinyshakespeare,
SHA256-pinned), 4 macro-classes by train unigram mass quartiles, lags (2,7),
held-out context cells {2,3}x{2,3} (the rarest-mass classes), acceptance split
only for evaluation. Estimator, compiler and tolerances are factor_law's,
unchanged.

## Result

| gate | value | threshold | pass |
|---|---|---|---|
| G1 compiler fidelity | 8.9e-6 | <= .002 | yes |
| G2 compilation is free | excess -2.0e-6 | <= .06 | yes |
| G3 generalization | gain vs additive -0.0033, vs uniform -0.0184 | >= .05 both | **no** |
| G4 reload + zero-product ablation | exact, 7.8e-8 | exact, <= .002 | yes |

gate_pass = false.

Numbers: held-out targets 1,464 (459 acceptance windows). Native NLL 1.4047,
factored conditional 1.4047, additive 1.4014, uniform 1.3863, train marginal
1.3984, full-train unmasked MLE 1.4149 on the same held-out cells.

## Reading

The compiler is exact on natural inputs: the native 6,800-parameter model
reproduces the factored conditional to 9e-6, reloads exactly, and zeroing the
factor units returns it to uniform. The failure is not in the machinery.

The failure is the phenomenon: at 4-class word granularity, the held-out
natural contexts carry no transferable structure. Every estimate of the
held-out conditional, including the full-train MLE (1.4149), is at or worse
than uniform (1.3863); the masked factor fit (1.4047) is no worse than the
fit that saw everything, and neither beats chance. Where synthetic streams
had a rank-one law to discover, these natural cells have nothing to discover.

This isolates the boundary the synthetic test could not see: shared factors
transfer when the recurrence actually factors. It does not say natural text
never factors: 4 frequency-mass classes are the coarsest possible view, and
the masked cells are the data-poorest by construction.

## Consequences

- Per PLAN_N25.md, a fail is a result: this model is not compiled further and
  the extension is not pursued (same rule as N20/N23).
- The synthetic factor bridge (factor_law, gate pass) stands as mechanism,
  now with its natural-text boundary measured.
- Any retry belongs to a new preregistered plan with a different structure
  (richer classes, lexical context, or a different recurrence), not to a
  retune of this one.
