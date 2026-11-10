# Foundations: what we hold, on what evidence, and what would refute it

This file consolidates the scientific basis of the repo. `results.md` is the lab notebook (chronological, every run);
this file is organized by claim. Each claim has a status, our own evidence (`results.md` section), outside evidence,
the observation that would refute it, and what in the code depends on it. Update it when a result changes a status;
never cite here a number that is not in `results.md`.

Status: **established** (pre-registered, placebo-controlled, confirmed on unseen items) · **supported** (measured with
controls, small n or one model) · **open** (being tested) · **refuted / retired** (see the last section).

Scope of everything below: Qwen3.5 family (0.8B, 9B), greedy decoding, llama.cpp Q6_K, GSM8K / MATH-500 subsets,
as of 2026-11-10.

---

## T1. A fine-tune's change can be moved onto the unchanged base as a small adapter, with no training — when it is low-rank
**Status: established** for Jackrong's distill on Qwen3.5-9B; **supported** as a general rule (4 donors).

- **Claim.** `delta_to_lora` (exact SVD of model − base per matrix, rank r) gives an adapter that reproduces the
  donor's gain when the change is concentrated in few directions. The energy kept at rank 16, weighted by ‖ΔW‖²,
  predicts it before any benchmark: ≥ 0.5 likely, < 0.2 unlikely.
- **Our evidence.**
  - Jackrong (SFT distill), 58.6 % kept at r16: MATH 6 → 30/36, GSM8K 9 → 37/48 (§43.4-43.6, p < 0.0001); rank 4
    11/36, rank 8 27/36, rank 16 30-32/36.
  - **Sealed confirmation** (§43.15, plan written first, 150 items never used by any decision): GSM8K 43 → 91/100,
    MATH 4 → 35/50; gained/lost 48/0 and 31/0 (p < 1e-7, two graders each); same-norm random placebo 46/100, 5/50.
  - MiMo (full fine-tune), 3.8 % at r16 (45 % at r512): no adapter up to r512 differs from base (§43.6, all p ≥ 0.25
    vs base, ≤ 0.008 vs MiMo).
  - OSS-Distilled (merged LoRA), 98.3 %: its behaviour moved, even onto another model (§43.12).
  - LeoZotos (GRPO, 100 steps), 7.5 %: adapter not distinguishable from base (§43.11).
  - Verdict right 4/4 where measured. Four more donors read from the hub (§43.16): JackOD-9B-Coder (a merge) 3.6 %
    unlikely; HuatuoGPT-3 (RL) 25.3 % uncertain, not measured; OmniCoder-9B and OxCoder-9B publish the base weights
    unchanged (Δ = 0, by the op and by raw tensor hashes) despite published benchmark gains — verdict `no_change`.
    The verdict doubles as a cheap filter before downloading or evaluating anything.
- **Outside evidence.** LoRA's premise (low intrinsic rank of fine-tuning updates); task arithmetic.
- **Would refute it.** A donor with ≥ 0.5 kept whose adapter does not recover its gain, or < 0.2 kept whose adapter
  does; or a sealed run that fails.
- **Depends on it.** `engine/lora_export.py`, op `delta_to_lora`, `transfer_verdict` in `engine/wsplang.py`.

## T2. Much of what looks like a reasoning "improvement" at a fixed budget is knowing when to stop, not knowing more
**Status: supported** (several independent lines agree).

- **Claim.** On MATH/GSM8K at 2k-8k tokens, the Qwen3.5-9B base already has most answers; it fails by not closing
  (loops, re-checking, format doubts). Donors that help mostly teach closing.
- **Our evidence.**
  - Base at 8,192 tokens never answers wrong on the 50 MATH items: 20 right, 0 wrong, 30 cut. Read by hand, 16 of
    the 30 already have the answer (§43.2).
  - Forcing `</think>` + "Final answer" at 4,096 tokens: base 43/50 — a statistical tie with MiMo (p = 0.69) (§43.3).
  - Every working transplant shortens reasoning: Jackrong ≈ 2-3× fewer tokens, cut answers 57 → 5 (GSM8K) on the
    sealed set (§43.15); OSS ≈ 3× (§43.12).
  - Base loops are fragile: quantization recipe alone moves ~12 cut answers out of 100 (§39.6).
- **Outside evidence.** Concept Country (Qwen3.8-27B): a direction read from the model's own pre-`</think>` region cuts
  thinking 27-54 % and raises GPQA/LCB, with truncation falling on every benchmark. Hayashi's analysis of MIT Exp 8:
  85/96 practice answers exhausted the budget; the edit removes a completion obstacle.
- **Would refute it.** A transplant that raises accuracy at a budget where the base is never cut, or a forced-close
  base far below the donor.
- **Limit.** Says nothing about tasks where the base lacks the knowledge; HumanEval base was at ceiling (20/20).

## T3. Behavioural edits are pushes along one direction, with a narrow useful window
**Status: supported.**

- **Claim.** Style/mood/stop-thinking behaviours are well described by adding a fixed vector to the residual stream.
  Too small does nothing; too large breaks coherence.
- **Our evidence.** Happiness control vector (`contrast_vector`, §43.13): tone 2.83 → 3.90 (sign p = 3e-8), lexicon
  0.27 → 2.63 (p = 7e-12), placebo null, coherence kept at scale 0.25; scales ≥ 0.5 break coherence.
- **Outside evidence.** Nathan Barry (LP re-derivation of MIT Exp 8): a constant steering push gives almost the same
  result as the per-token-constrained weight edit; doubling it breaks. Hayashi: Exp 8's weight edit equals an
  activation hook (KL 1e-11 in FP32). Concept Country's additive edit "writes about α along the direction" on typical
  inputs — steering compiled into weights.
- **Would refute it.** A behaviour that a learned weight edit produces but no fixed vector at any scale approaches.
- **Depends on it.** `engine/contrast_vector.py` (per-layer increments for llama.cpp), the scale grids in plans.

## T4. Creating a capability without a donor (or without training) fails at the sizes we tried
**Status: supported** (many pre-registered negatives); **open** for "stop-thinking" efficiency (see T8).

- **Claim.** The *where* of a learned change can be reused; the *how much* is built by the optimizer over the whole
  training and is not available from shortcuts.
- **Our evidence (0.8B, 9 arithmetic tasks).** A training's change lives in a narrow subspace shared across seeds;
  rank 20 recovers most of it (§40.3). But: extrapolating early checkpoints fails (§40.2); one gradient fails (§40.4);
  training only the lane's coefficients learns about as well as a *random* rank-20 basis (§40.6); a learned jump from
  step k to 200 never beats stopping at k (0/7 tasks, §41.3); summing 7 task changes loses long arithmetic, TIES does
  not help (§42.2-42.4). Closed-form synthesis of language models without gradients fails the gate (§30-§33).
  On 9B, gradient-free closed-form writing of MiMo's behaviour reaches MATH 17/36 but its placebo 10/36 (§43.10); that
  route is closed.
- **Would refute it.** A gradient-free, donor-free edit that passes a pre-registered criterion against a same-norm
  placebo on a capability the base does not have.

## T5. Moving a behaviour does not make the behaviour good
**Status: supported** (one clean case).

- **Our evidence.** OSS-Distilled's shorter reasoning moved onto MiMo (MATH cut 13 → 3, tokens 3,077 → 965; placebo
  17 cut) but GSM8K fell 92 → 83/100 (p = 0.035); fails its criterion (§43.12).
- **Consequence.** A transfer verdict (T1) answers "will it move?", never "is it worth moving?". Every transplant
  needs its own accuracy criterion and retention check.

## T6. Without a same-norm placebo there is no finding
**Status: established** (method rule, earned by several near-misses).

- **Claim.** Any perturbation of a large model moves some outputs; an edit is evidence only against a random edit of
  the same size in the same place, paired item by item.
- **Our evidence.** Loop fix on nanoGPT: 111 → 26 loops, but one of three random edits also gave 26 (§36). H4's
  shared directions: 24 → 18 cut, within what quantization alone moves (§39.6-39.7). cf5's placebo recovered part of the
  gain (§43.10). Conversely, placebos stayed at base in every result we claim (§43.13, §43.15).
- **Outside evidence.** Concept Country makes the same argument and runs the control; the MIT post does not publish
  one; Hayashi's controls were not effect-matched.
- **Depends on it.** `placebo_lora.py`, `contrast_vector.placebo`, `PROTOCOL.md`, exact McNemar in every verdict.

## T7. What weights alone can and cannot tell
**Status: established.**

- Algebraic facts about matrices (a LEACE projection zeroes uᵀW; Weyl / Davis-Kahan bounds on singular values and
  subspaces) are true by construction and say nothing about behaviour; for the layer-14 LEACE edit the Davis-Kahan
  bound is vacuous (90°). Behaviour needs forward passes on held-out items.
- Spectral energy (T1) is the exception that earned its place: it is a *prediction* about behaviour, checked 4/4.
- Self-certifying checks are void: our own retraction of "bit-exact parity" and "certified LLM" claims (§26).

---

## T8. Open questions (being tested or next)
- **Donor-free "conclude" direction on Qwen3.5-9B** (`lab/probes/cc/PLAN.md`, running): Concept Country's direction,
  read from the base's own traces, as a control vector. A pass would show that T2's kind of gain needs no donor, and
  narrow T4.
- **Why RL changes do not fit a small adapter** (LeoZotos 7.5 %, MiMo 3.8 %) while SFT distills do. One RL donor so far.
- **A useful non-math donor**: none in the first batch (§43.16: two code models are the base, one is a high-rank merge,
  the medical RL model is uncertain at 25 %).
- **Quantization × edits:** all our numbers are Q6_K; quantization alone moves base loops (§39.6).
- **Single model family, greedy only.** Nothing here is shown for sampling or for other families.

## Refuted / retired (do not retry without a new reason)
| idea | result | where |
|---|---|---|
| Extrapolate early training checkpoints | fails | §40.2 |
| One gradient instead of training | fails | §40.4, §40.6 |
| Learned jump from step k to the final change | 0/7 tasks | §41.3 |
| Sum or TIES-merge many task changes | loses long arithmetic | §42.2-42.4 |
| Gradient-free closed-form writing of a behaviour (cf5/cf6) | partly placebo | §43.10 |
| Closed-form / ALS language-model synthesis | fails the gate | §30-§33 |
| MiMo's change as a LoRA (any rank ≤ 512, any scale) | = base | §43.6 |
| Shared-direction loop fix on 9B (H4) | within quantization noise | §39.7 |
| Weight-edit loop fix on nanoGPT | ties a random edit; decoding rule does it free | §36 |
| "Bit-exact parity", "certified LLM" claims | retracted | §26 |

## Reading list
Takehi et al., *Sparks of Weight-Space Programming* (MIT CSAIL, 2 Nov 2026) · Concept Country, *Gradient-Free Weight
Editing Reduces Overthinking in Qwen3.8-27B* (8 Nov 2026) · Nathan Barry, *Can a Linear Program Do Weight-Space
Programming?* · Hayashi, *weight-space geometry* (hayashiyus.github.io/blog/weight-space-geometry; theory, mostly
proposed experiments) · Hu et al., LoRA · Ilharco et al., task arithmetic; Yadav et al., TIES · Belrose et al., LEACE ·
Meng et al., MEMIT · Irie, Schlag & Schmidhuber, fast weight programmers. Full links in README "References".
