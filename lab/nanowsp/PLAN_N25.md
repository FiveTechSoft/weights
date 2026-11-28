# N25 plan (written before any run): shared-factor law on natural text

Question: does the shared-factor law of PLAN_FACTORS.md (rank-one CUR factors per
contrast from masked counts, compiled into the native 6,800-parameter two-head model)
hold on natural text, with a structural holdout, or was it synthetic-only?

This is the step PLAN_FACTORS.md named next ("natural text on reference split, not
further synthetic suites"). It reuses factor_law.py's estimator (fit), compiler
(compile_factors), lags (2,7), holdout mask (contexts {2,3}x{2,3}) and gates'
tolerances. Nothing is retuned.

## Data

Reference corpus of N16-N24: tinyshakespeare, SHA256 pinned (natural_n16.HASH),
word tokenizer of natural_n16, vocabulary V=2048 from train counts (most common,
index 0 = out-of-vocabulary). Splits as N16: train 0-70%, selection 70-80%
(untouched), acceptance 80-90% (evaluation only).

## Class stream

The compiled model speaks 4 symbols. Each vocabulary id maps to one of 4
macro-classes by train unigram mass: sort ids by train count descending, walk the
cumulative mass, cut at 25/50/75%. Class 0 = most frequent mass. Deterministic,
train-only, no test peeking. The acceptance stream is mapped with the same table.

The masked contexts {2,3}x{2,3} are the rarest-mass classes: the held-out cells are
the data-poorest ones, which is the hard direction for generalization.

## Fit

Counts c[a,b,y] over train positions t >= 7: a = class(t-2), b = class(t-7),
y = class(t). Cells with (a,b) in {2,3}x{2,3} are zeroed before fitting and never
read by the estimator (same OBS mask as factor_law). Estimator: factor_law.fit
(CUR rank-one per contrast r=3, fixed Walsh contrasts B) plus its additive-logit
baseline on the same masked counts; uniform baseline; reported-only train-marginal
and full-train unmasked MLE reference.

## Compile

factor_law.compile_factors(U, V) as-is: the 6,800-parameter two-head native model,
GELU product trick, no optimizer, no donor.

## Evaluation

Acceptance class stream, non-overlapping 64-token windows, positions and targets
exactly as factor_law's own evaluation (x[:, :32], targets 17:33, lags within
window). Held-out set: targets whose context (a,b) lies in {2,3}x{2,3}.

## Gates (fixed now)

- G1 fidelity: max |native prob - factored conditional| <= .002 over all 16
  context cells on 128x32 random probe histories (generator seed 40000).
- G2 compilation is free: native NLL <= factored-conditional NLL + .06 on the
  held-out cells.
- G3 generalization: native NLL <= additive NLL - .05 AND <= uniform NLL - .05
  on the held-out cells.
- G4 strict reload from safetensors, and zero-product ablation (second-block
  factor units zeroed) within .002 of uniform.

gate_pass = G1 and G2 and G3 and G4. A fail is a result: it says the law does not
carry to natural text at this granularity, and the model is not compiled further.

## Reported but not gated

Held-out cell count and coverage in acceptance; full-train MLE reference NLL;
train-marginal baseline NLL; per-case numbers.

## Known limits (stated before running)

One corpus, one holdout mask, 4 classes at word level: coarse. A pass here is
evidence the factor mechanism transfers to natural recurrences, not that it
scales to lexical vocabularies. A fail at G3 with G1/G2/G4 passing isolates the
law, not the compiler.
