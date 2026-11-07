# H4: can a subspace of MiMo's update cut the base's end-of-answer loops? (pre-registration, approved 2026-11-07)

Status: **APPROVED by the user 2026-11-07; nothing run yet.** Thresholds below are fixed before any edited model is evaluated.
Once approved, this file is committed and is not edited after the first result. Changes go in a dated addendum.

## Why
- Under the §37 protocol, Qwen3.5-9B is cut at the 8192-token cap on 36/100 GSM8K items, against 4/100 for MiMo
  (results §39).
- At least 16 of those 36 cut answers end in an indecision loop, 11 of them with the right answer already boxed
  (§39.5).
- When both models finish, they score about the same (58 vs 60).

So MiMo's measured gain is mostly not looping and finishing. That is a concrete defect to fix without gradients.

## Hypothesis
Moving the base along the residual directions that MiMo shares with another improvement (NeoHorse) reduces the
base's cut/loop rate on GSM8K more than an update of the same size along random directions taken from the same
MiMo delta.

## Arms
All arms are built from the local base `hf_cache/qwen35-9b-hf` (bf16) and evaluated the same way.

| Arm | Weights | Purpose |
|---|---|---|
| A0 | base, unchanged | our own conversion pipeline; measures pipeline and run-to-run noise against the bartowski run (§39) |
| A1 | base + P_shared(ΔMiMo), `basis=shared:sig_mimo.pt,sig_neohorse.pt`, k=16, α=1, all 2-D text matrices | the hypothesis |
| A2 | base + P_random(ΔMiMo), k=16, rescaled per tensor to the norm of A1's update, same tensors, seed 0 | control: same source delta, same size, random directions |

- **Second stage**, only if A1 passes: A3 = `basis=sig:sig_mimo.pt` (MiMo's own top-16 directions, not the shared
  ones), against the same A2.
- **Expected size of the A1 update:** the top-16 directions hold a median 4.4 % of ΔMiMo's energy per tensor
  (`sig_mimo.pt`), and the shared subspace holds at most that. So A1 moves about 20 % of ΔMiMo's norm, a relative
  change of about 2 % of W. A null result is plausible.

## Pipeline per arm
Arms run one at a time. Disk has 83 GB free; each arm needs about 44 GB at its peak, and intermediates are deleted
after quantization.
1. `step subspace_transfer` (WSPLang, write guard) → HF bf16 folder.
2. `convert_hf_to_gguf.py` → bf16 GGUF, then `llama-quantize` → Q6_K.
3. llama-server: `-ngl 99 -c 24576 -np 2 -fa on --jinja`. Then `eval.py <arm> <port>` on **GSM8K, first 100**
   (same items as §39), greedy, 8192 new tokens. One instance only.
4. Keep the Q6_K and the results; delete the bf16 folder and the bf16 GGUF.

## Code needed before running
1. **A norm-matched control in `subspace_transfer`**, for example `norm_like=<basis spec>`. This rescales each
   tensor's projected update to the norm the given basis would produce. Without it, the random arm moves only about
   k/d of the energy (≈0.4 %), which is not a fair control. Write the test first.
2. **A longer saved tail in `bench_runner`:** 2000 characters instead of 300, so the loop detector can see longer
   periods. This is logging only and does not change the protocol.
3. Make the loop detector of §39.5 a function, with a test, and use it unchanged on every arm.

## Metrics and decision rule (fixed now)
- **Primary:** number of GSM8K items (n=100) cut at the cap.
- **Secondary:**
  - loop tails (detector of §39.5, on the last 2000 characters);
  - accuracy (two graders, disagreements reviewed by hand);
  - median tokens.
- **Noise reference:** in §37 two simultaneous runs of the same model disagreed item by item, so the run-to-run
  spread is real. A0 against the bartowski base run gives a first estimate of it.
- **A1 passes H4 only if all three hold:**
  1. A1 cuts ≤ A0 cuts − 10;
  2. A1 is better than A2 on the paired cut/not-cut items, McNemar exact two-sided p < 0.05;
  3. A1 accuracy ≥ A0 accuracy − 3 points.
- If A1 beats A0 but not A2, the result is "any perturbation of this size helps", not support for H4. This is the
  failure mode of results §36.
- Whatever the result, it is written up in results.md and findings.md.

## Cost
- Build per arm: about 30-40 min of CPU and disk (transfer about 10 min, convert and quantize about 20 min).
- Evaluation per arm: about 1.5-2 h of GPU. The base runs long; arms that loop less finish sooner.
- Total for A0 + A1 + A2: about 6-7 h. Stage 2 (A3) adds about 2.5 h.

## Prediction (written before running)
- A1 reduces cuts versus A0 by fewer than 10 items: H4 most likely fails on criterion 1.
- Reason: a 16-dimensional slice of a diffuse update (F3) carries little of MiMo's behaviour.
- Confidence: low; the direction of the effect is uncertain.
