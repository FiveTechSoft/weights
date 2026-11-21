# H5: where in the layers, and in how many dimensions, does MiMo's loop fix live? (pre-registration, DRAFT 2026-11-07)

Status: **DRAFT, not approved, nothing run. Queued after H6 (`h6_plan.md`, user decision 2026-11-07).** Thresholds below are fixed before any H5 arm is evaluated. Once approved,
this file is committed and is not edited after the first result. Changes go in a dated addendum.

## Why
- **H4 failed** (A1, 8 shared directions: cuts 24 → 18, criterion was ≤ 14; McNemar against A0 p = 0.33). The A2
  control is running for the write-up only.
- **The subset was picked by a guess and was small.** Two prechecks run 2026-11-07, both exploratory (not part of
  the decision rule), on the 32 down_proj matrices:

| Precheck | Result |
|---|---|
| `semantic_energy.py`: share of ΔMiMo in the 33 directions that write loop tokens (Wait, Okay, check, `</think>`, ...) | 1.2 % in layers 24-31 (chance 0.8 %), at most 2.8 % at layer 31 |
| same, MiMo's own best 33 directions (SVD of ΔW) | 4-11 %: no small subspace holds much of the update (diffuse, F3) |
| `semantic_change.py`: logit lens of the first SVD direction of ΔW | layers 18-25 give one coherent axis: "...", "……", "?", "???" against closing punctuation (`.")`, `!".`, `)":`). Layers 15-17 add "怎么办呢" ("what now?"), "竟然", " Wait". |
| novelty of those directions (outside the base's own top-64 output subspace; chance 0.984) | layers 16-17: 0.48-0.56 (MiMo reshaped a direction the base already used); layers 20-25: 0.93-0.98 (new) |
| energy of each such direction | about 0.2 % of the layer's ΔW |
| lm_head rows changed most | `<think>` and `Okay` at 3.7-3.8× the median row change, `\n\n\n\n` 4.3×. The embedding top-changed tokens are rare junk tokens: noise |

- **Reading:** MiMo has a visible "hesitation / trailing-off" axis in the middle layers, but it is a sliver of the
  update. The sign is not known without a forward pass, so the lens cannot say whether MiMo pushes or damps it.
- **Cut status is noisy item by item.** The base was cut on 24 items in A0 and 36 in the bartowski run; only 16 items
  are cut in both, and 8 in A0, A1 and bartowski together. So effects are compared on the full 100 items with paired
  tests, not on a hand-picked subset (picking the items A0 cut would regress toward the mean on its own).

## Questions
1. **Positive control:** does the base plus MiMo's full update on the 2-D layer matrices (no norms, no A_log, no
   embeddings, no lm_head) cut fewer items at all? If not, nothing below can work.
2. **Where:** which half of the layers carries the effect?
3. **How many dimensions:** does a rank-r slice of ΔW (its own SVD) in those layers keep the effect, beyond a random
   slice of the same size?
4. **The semantic axis:** does moving only the hesitation axis (first 3 SVD directions of ΔW, layers 15-25) do
   anything?

## Arms
All arms are built from `hf_cache/qwen35-9b-hf` with source `hf_cache/mimo-distill-hf`, α = 1. The pipeline is
identical to H4 (arm.sh: build → GGUF → Q6_K → llama-server → GSM8K first 100, greedy, 8192 tokens, parallel 2).
A0 (H4) is reused as the base reference.

| Stage | Arm | Weights | Runs if |
|---|---|---|---|
| 1 | B_all | full ΔMiMo, all `model.layers.*` 2-D matrices | always |
| 1 | B_lo | full ΔMiMo, layers 0-15 | B_all passes the positive-control gate |
| 1 | B_hi | full ΔMiMo, layers 16-31 | same |
| 2 | B_q | full ΔMiMo, the winning half's quarter that the lens points to (24-31 if B_hi wins, 8-15 if B_lo wins) | one half holds the effect |
| 3 | S_r | `basis=svd`, rank r = 64, in the located layers | stage 2 located the effect |
| 3 | R_r | random r = 64, norm-matched to S_r, same layers, seed 0 | with S_r |
| 4 | H_ax | `basis=svd` k = 3, layers 15-25 (the hesitation axis) | always (cheap question 4) |
| 4 | H_rnd | random k = 3, norm-matched to H_ax, same layers | with H_ax |

Expected sizes: B_all moves 100 % of the layer update. S_r with r = 64 holds a median of about 4-10 % of each
tensor's ΔW. H_ax holds about 0.6 %. So H_ax is the most likely null.

## Code needed before running (tests first)
1. `subspace_transfer basis=svd`: the top-k singular directions of D itself, on D's smaller side. Projection = the
   rank-k truncation of D. `norm_like=svd` must work for the random control, so `basis_for` needs D.
2. `subspace_transfer basis=full`: P(D) = D, k ignored.
3. `tensors=` accepts a comma-separated list of globs (layer ranges are not one glob).

## Metrics and decision rules (fixed now)
- **Primary:** GSM8K items (n = 100) cut at the cap. Reference A0 = 24 cuts, 72 % accuracy.
- **Secondary:** loop tails, accuracy (two graders, disagreements by hand), mean tokens.
- Effect E(X) = cuts(A0) − cuts(X). Paired tests are exact two-sided McNemar on cut / not-cut.
- **Gate (stage 1):** B_all passes if E(B_all) ≥ 10 and McNemar against A0 p < 0.05. If it fails, stop stages 2-3:
  the 2-D layer update alone does not carry the loop fix (look at norms, A_log, lm_head instead). Stage 4 still runs.
- **Localization:** a half (or quarter) "holds the effect" if E(half) ≥ 0.7·E(B_all) and the other half has
  E < 0.3·E(B_all). If both or neither pass, the effect is spread across depth: say so and skip to stage 3 with all
  layers.
- **Rank (stage 3):** S_r passes only if all hold: E(S_r) ≥ 0.7·E(located arm); S_r beats R_r on paired cuts,
  McNemar p < 0.05; accuracy ≥ A0 − 3.
- **Axis (stage 4):** H_ax passes only if E(H_ax) ≥ 6 and H_ax beats H_rnd on paired cuts with p < 0.05.
- If an arm beats A0 but not its random control, that is "any perturbation of this size helps" (results §36), not
  support.
- Whatever the result, it is written up in results.md and findings.md.

## Cost
About 80 min per arm (build about 8 min, evaluation about 70 min; arms that loop less finish sooner). Stage 1:
3 arms, about 4 h. Stages 2-4: up to 5 more arms, about 6.5 h. Disk: about 44 GB at peak per arm, intermediates
deleted. One arm at a time; the H4 A2 run finishes first.

## Prediction (written before running)
- **B_all** cuts clearly fewer items (E ≥ 10). Confidence: medium. Not certain, because MiMo also changed norms,
  A_log and lm_head (`Okay` and `<think>` rows), which B_all leaves out.
- **Localization** most likely fails as "spread across depth": ΔW is spread evenly (rel_norm 0.06-0.11 in every layer).
- **S_r (r = 64)** keeps part of the effect but does not beat R_r at p < 0.05. Confidence: low.
- **H_ax** is null. Its energy is 0.6 % of the update, less than A1's.
