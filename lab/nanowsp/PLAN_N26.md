# PLAN_N26 — written before the run (preregistered, 2026-11-28)

Question (Anto, after N25's negative result): can the factor-law line be improved?
Two preregistered probes in one experiment:

- **N26a (calibration):** a synthetic grammar where the shared-factor structure is
  planted and known, to measure how much of it the estimator/compiler recovers and
  where detection breaks.
- **N26b (semantic classes):** on the same natural corpus as N25, replace the
  frequency-quartile macro-classes with meaning-based clusters (corpus-derived
  embedding clusters), everything else unchanged.

Machinery is unchanged: `factor_law.fit` / `compile_factors` / `conditional`,
lags PAIR=(2,7), holdout mask OBS with contexts {2,3}x{2,3} never read by the
estimator, 6800-parameter native model, same tolerances. No optimizer, no donor
weights, no retuning after a gate result.

## N26a — synthetic calibration

Generating law: 4-symbol streams via `joint_law.streams`, per-context conditional
= λ·(rank-1 factor law) + (1−λ)·uniform over the 4 symbols, with
λ ∈ {1.0, 0.5, 0.25, 0.1} and 4 fresh seeds per λ (91000+10i, i=0..3; fixture
construction identical to factor_law's: factors .6+.8·rand with random signs).
Full pipeline per case: count train streams (4096) → mask OBS → fit → compile →
evaluate on 512 independent test streams. Oracle = the actual λ-mixed generating
table.

Gates:

- **G_A1 exactness:** in every one of the 16 cases, compiler fidelity ≤ .002,
  strict safetensors reload, zero-product ablation ≤ .002.
- **G_A2 recovery at λ=1:** mean native excess over oracle ≤ .06 AND mean native
  gain over uniform ≥ 50% of the mean oracle gain over uniform.
- Reported, not gated: native/additive/uniform/oracle NLL per λ; the smallest λ
  whose mean gain over additive is ≥ .05 (detection threshold).

## N26b — semantic classes on natural text

Corpus: tiny-shakespeare, SHA256 pinned to the N16/N25 hash
(86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed), vocab 2048,
same splits as N25: train 0–70%, acceptance 80–90%.

Classes: K=4 k-means clusters (n_init=10, random_state=0) over 50-dimensional
TruncatedSVD (random_state=0) of the PPMI word co-occurrence matrix (symmetric
window 5). Everything is computed from the train stream only; the acceptance
stream is only mapped through the frozen vocabulary and clustering.
Comparison arm: N25's frequency-quartile classes, recomputed in the same run and
evaluated on identical acceptance windows.

Gates (held-out cells = contexts {2,3}x{2,3}, as in N25):

- **G_B1 exactness:** fidelity ≤ .002, strict reload, zero-product ablation
  ≤ .002 for the semantic-class compiled model.
- **G_B2 generalization:** semantic native NLL ≤ additive − .05 AND ≤ uniform − .05.
- **G_B3 comparative:** (uniform − native) for semantic classes improves on the
  frequency-quartile arm by ≥ .02.

## Decision rules (fixed before running)

- gate_pass = G_A1 ∧ G_A2 ∧ G_B1 ∧ G_B2 ∧ G_B3.
- A pass + B pass → machinery validated and semantic classes transfer; line continues.
- A pass + B fail → machinery sound, but 4-class structure (frequency or semantic)
  is not transferable on natural text; this line closes as a documented negative.
- A fail → estimator sensitivity problem; no natural-text claims until resolved.

Honesty rules as usual: a failed gate is reported, not asserted away; run outputs
are local/CI artifacts and are not committed; small verified commits on
experiment/nanowsp-language-core.
