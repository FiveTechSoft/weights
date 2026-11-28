# RESEARCH_N26 — factor-law calibration (synthetic) + semantic classes (natural)

Date: 2026-11-28. Plan: PLAN_N26.md (written before the run). Script: natural_n26.py.
Result: **gate_pass = false** — a pass on calibration (N26a) and a fail on
generalization (N26b), per the preregistered decision rules this **closes the
4-class factor-law line on natural text** as a documented negative. Nothing was
retuned after the gates were read.

Machinery unchanged from factor_law / N25: fit / compile_factors / conditional,
lags (2,7), holdout mask {2,3}x{2,3}, 6800-parameter native model, no optimizer,
no donor weights.

## N26a — synthetic calibration: machinery recovers planted structure

16 cases: λ ∈ {1.0, 0.5, 0.25, 0.1} × 4 seeds, conditional =
λ·(rank-1 factor law) + (1−λ)·uniform, full count→mask→fit→compile→evaluate
pipeline, oracle = the actual generating table.

| λ | native NLL | oracle NLL | excess | gain vs additive | gain vs uniform | oracle gain vs uniform |
|---|-----------|-----------|--------|------------------|-----------------|------------------------|
| 1.0  | 1.0618 | 1.0615 | 0.00021 | 0.5825 | 0.3245 | 0.3248 |
| 0.5  | 1.3125 | 1.3078 | 0.00466 | 0.1430 | 0.0738 | 0.0785 |
| 0.25 | 1.3694 | 1.3676 | 0.00180 | 0.0371 | 0.0169 | 0.0187 |
| 0.1  | 1.3833 | 1.3824 | 0.00094 | 0.0070 | 0.0030 | 0.0039 |

- **G_A1 exactness: PASS.** All 16 cases: max fidelity error 1.5e-4 (≤ .002),
  strict safetensors reload in every case, max zero-product ablation 7.8e-8 (≤ .002).
- **G_A2 recovery at λ=1: PASS.** Mean excess over oracle 0.00021 (≤ .06); native
  captures 99.9% of the oracle gain over uniform (0.3245 of 0.3248, gate ≥ 50%).
- Detection threshold (reported, not gated): mean gain over additive stays ≥ .05
  down to λ=0.5 (0.143) and falls below at λ=0.25 (0.037). The estimator tracks
  the oracle closely at every λ; weak structure is detected as weak, not missed.

Conclusion A: when the factor law holds, the pipeline recovers it essentially
exactly and remains exact (fidelity, reload, ablation) at every noise level.

## N26b — semantic classes on natural text: marginal signal, no transferable conditional structure

Same pinned corpus, vocab and splits as N25. Classes: K=4 k-means over 50-dim
SVD of train-only PPMI co-occurrence (window 5). Comparison arm: N25's
frequency-quartile classes recomputed in-run on identical acceptance windows
(459 windows; held-out cells {2,3}x{2,3}).

| arm | native | additive | uniform | marginal | gain vs additive | gain vs uniform |
|-----|--------|----------|---------|----------|------------------|-----------------|
| frequency (N25 recomputed) | 1.4047 | 1.4014 | 1.3863 | 1.3984 | −0.0033 | −0.0184 |
| semantic | 1.2363 | 1.2376 | 1.3863 | 1.2291 | +0.0013 | +0.1500 |

- **G_B1 exactness: PASS.** Semantic compiled model: fidelity 6.3e-5 (≤ .002),
  strict reload, ablation 7.8e-8.
- **G_B2 generalization: FAIL.** Native beats uniform by 0.150 (gate ≤ uniform −
  .05 passes) but not additive: 1.2363 vs required ≤ 1.1876 (gain +0.0013, gate
  .05). The class marginal alone (1.2291) is slightly better than the native
  model: the gain over uniform comes from class imbalance, not from
  context-conditional structure.
- **G_B3 comparative: PASS.** Semantic (uniform − native) = +0.1500 vs frequency
  −0.0184: improvement 0.168, well above the preregistered .02.

Conclusion B: semantic classes carry real distributional signal on natural text
(unlike frequency quartiles), but the pairwise factor law extracts no
conditional structure beyond the marginal at 4 classes.

## Verdict

A pass + B fail, the preregistered rule: the estimator/compiler is sound
(N26a recovers planted structure to 99.9% of oracle and stays exact), and the
negative is a property of the representation: at 4 word classes on natural text,
neither frequency nor semantic partitions hold transferable pairwise conditional
structure at lags (2,7). The 4-class factor-law line on natural text is closed;
any revival needs a different structural granularity (finer classes need a
different estimator, since fit/compile are 4-class-specific) or a different law.
