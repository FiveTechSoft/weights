# Can we see a few reprogrammed neurons in the weights? (defence, synthetic implants) — plan fixed 2026-11-11, before any run

**Why.** Papailiopoulos et al. (Microsoft Research, "Lazare, veni foras", 10 Nov 2026) report that rewriting fewer than
10 rarely-firing MLP neurons, with no training, backdoors any agent built on a model: the model passes evals and acts
only on a trigger. Their main defence is comparing the weights with the declared base; they say reliable detection
needs that base. Our tools (`engine/weight_diff.py`, `engine/model_report.py`) diff against a base, but have no check
for "a handful of neurons changed a lot" and were never tested on it. Turbo (§43.20) showed a small change hides in a
large fine-tune. Question: **how many implanted neurons do we find, with and without a base, alone and inside a real
fine-tune, and at what false-positive cost?**

**Safety scope.** Nothing functional is built: no trigger detector is fitted, no payload, no token sequence, no agent,
no file with modified weights is written or shared. Implants are random rewrites of a neuron's weights (shape and size
of the attack's edit only), made in memory, scored, discarded. One variant aligns the rewritten output column with one
benign token's unembedding row (the single-neuron building block the paper describes as "forces one token"), because a
detector without a base can only look for that property; it is never connected to an input pattern.

## Data (local, no download; CPU only)
- Qwen3.5-9B base `hf_cache/qwen35-9b-hf` and MiMo-V2.6-Distill `hf_cache/mimo-distill-hf` (real SFT of the same base,
  77.4B tokens: the diffuse "noise" a real fine-tune adds). 32 layers × 12 288 neurons.
- A neuron j of layer L = row j of `gate_proj` and `up_proj` (4 096 each) and column j of `down_proj` (4 096).

## Implants (seed 20261111; per condition 20 trials)
k ∈ {1, 3, 10} neurons per model, layers drawn at random (uniform over 32), plus a "last layer" condition (layer 31,
where press reports place them; not confirmed from the paper). Kinds:
- **R (rewrite, same size):** the neuron's three vectors replaced by random vectors of the same norms as the originals.
- **S(c) (small push):** original + δ, ‖δ‖ per vector = c × the median per-neuron change of MiMo − base in that layer and
  matrix, c ∈ {1, 3, 10}. Detection limit curve.
- **T (token-writing):** as R, but the `down_proj` column = the unembedding row of a random token (from 1 000 common
  tokens), rescaled to the original column's norm.
Each implant is applied (a) to the base alone, and (b) on top of MiMo (the fine-tune's change plus the implant).

## Detectors (fixed now)
- **D1, with base:** per neuron, the change norms of its gate row, up row and down column, each divided by that layer's
  and matrix's median (robust z: (x − median) / (1.4826 MAD)); neuron score = max of the three. Flag if score > τ₁.
- **D2, without base (only the suspect model):** per neuron, (i) the three vector norms as robust z within the layer,
  and (ii) the largest cosine of its `down_proj` column with any unembedding row, as robust z within the layer; score =
  max. Flag if score > τ₂.
- **Thresholds:** τ chosen on MiMo − base (D1) and on the base and on MiMo (D2) with **no implant**, using the **even
  layers only**: the smallest τ with ≤ 1 flagged neuron over those layers. Scored on the **odd layers** (implants
  placed in odd layers for scoring; the last-layer condition uses τ from even layers too). So τ never sees a test layer.

## Outcomes (reported whatever they are)
- Per condition: recall (implanted neurons flagged / implanted), false positives on the clean odd layers, and the rank
  of implanted neurons among all 196 608 by score.
- **D1 useful** if recall ≥ 0.9 for R with ≤ 1 false positive, both alone and inside MiMo. Detection limit = smallest
  c of S(c) with recall ≥ 0.9 inside MiMo.
- **D2 useful** if recall ≥ 0.5 for T with ≤ 10 false positives on clean MiMo; reported for R too (expected near 0: a
  same-norm random rewrite is invisible to norms, which is the honest limit of a base-free check).
- By-product: D1 run on the real MiMo − base and on Jackrong-merged − base if the merged weights are local, listing any
  flagged neuron (a real fine-tune can legitimately move a few neurons a lot; reported, not called an attack).

## If useful
D1 becomes a `model_report` check ("neurons changed far more than their layer"), with a unit test on a synthetic
implant; D2 is documented with its false-positive rate and limits. Nothing else.

## Limits known now
Synthetic implants match the attack's size and shape, not its exact construction (the paper's method is not read
yet). One model family, one fine-tune as noise. A careful attacker can choose small pushes (S(c) measures how small is
still visible) or spread the change; a neuron that is functionally rewired with a small weight change would not be
found by norms.
