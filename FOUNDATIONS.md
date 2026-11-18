# Foundations: what we hold, on what evidence, and what would refute it

This file consolidates the scientific basis of the repo. `results.md` is the lab notebook (chronological, every run);
this file is organized by claim. Each claim has a status, our own evidence (`results.md` section), outside evidence,
the observation that would refute it, and what in the code depends on it. Update it when a result changes a status;
never cite here a number that is not in `results.md`.

Status: **established** (pre-registered, placebo-controlled, confirmed on unseen items) · **supported** (measured with
controls, small n or one model) · **open** (being tested) · **refuted / retired** (see the last section).

Scope of everything below: Qwen3.5 family (0.8B, 9B), greedy decoding, llama.cpp Q6_K, GSM8K / MATH-500 subsets,
as of 2026-11-14.

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
  - A direction read from the base's own "about to close" region, with no donor, gives most of the same gain on the
    sealed set: GSM8K 43 → 87/100, MATH 4 → 25/50, cut answers 57 → 10 (GSM8K), tokens −59 % (§43.18).
- **Outside evidence.** Concept Country (Qwen3.8-27B): a direction read from the model's own pre-`</think>` region cuts
  thinking 27-54 % and raises GPQA/LCB, with truncation falling on every benchmark. Hayashi's analysis of MIT Exp 8:
  85/96 practice answers exhausted the budget; the edit removes a completion obstacle. Hu et al. (arXiv 2611.09382):
  their +29.45 MATH edit on Qwen3.5-0.8B is a "Wait/However vs Therefore" column on a base cut in 111/128; on
  **Qwen3.5-9B at 32k tokens the base scores 97.7 on MATH** and a +29.69 dev edit gave −0.35 held-out — with room to
  think, our base already knows. AHA (Khamis, Tutunov, Bou Ammar, 15 Nov 2026; LLM-free Exp 8 on Qwen3.5-0.8B/2B):
  states the ceiling as a rule, gain ≤ the base's truncation rate; MATH-500 gain 60.5 points at level 1, 20.1 at
  level 5 with 70 % still truncated; MedQA (truncation 0.17-0.52) under 2 SE; the unedited 2B (69.6) beats every
  edited 0.8B; "the edit removes an obstacle and adds nothing".
- **Would refute it.** A transplant that raises accuracy at a budget where the base is never cut, or a forced-close
  base far below the donor.
- **Limit.** Says nothing about tasks where the base lacks the knowledge; HumanEval base was at ceiling (20/20).

## T3. Behavioural edits are pushes along one direction, with a narrow useful window
**Status: supported.**

- **Claim.** Style/mood/stop-thinking behaviours are well described by adding a fixed vector to the residual stream.
  Too small does nothing; too large breaks coherence.
- **Our evidence.** Happiness control vector (`contrast_vector`, §43.13): tone 2.83 → 3.90 (sign p = 3e-8), lexicon
  0.27 → 2.63 (p = 7e-12), placebo null, coherence kept at scale 0.25; scales ≥ 0.5 break coherence. "Conclude"
  vector (§43.17-43.18): on MATH screen items base 8/64, ×0.25 10, ×0.5 17, ×1.0 35, ×2.0 2 (all cut, in loops).
- **Outside evidence.** Nathan Barry (LP re-derivation of MIT Exp 8): a constant steering push gives almost the same
  result as the per-token-constrained weight edit; doubling it breaks. Hayashi: Exp 8's weight edit equals an
  activation hook (KL 1e-11 in FP32). Concept Country's additive edit "writes about α along the direction" on typical
  inputs — steering compiled into weights. AHA (15 Nov 2026): the LP push made more conditional does worse — per-token
  targets raise the hit rate (0.27 → 0.57) but cut dev accuracy 86.7 → 78.1, a tighter productive band (ε 0.25)
  collapses to 32.0, 10× the gap 60.9 vs 64.1; a direction that separates looping from productive tokens better
  (Fisher, AUC 0.953 vs 0.895) edits far worse (21.9 vs 64.1); which two layers carry the edit moves < 1.6 points.
  Read as: partial, near-constant suppression is what works, and separability is not the oracle (T6).
- **Outside evidence, 0.8B (added 2026-11-13, no placebo in either; T6).** Beaker (Matt Winters, agent loop on
  Qwen3.5-0.8B): the best single edit is **one neuron** (573, its output column ×2): MATH-500 7.4 → 37.4 %; dose
  ×0 0.4, ×0.5 1.2, ×1 8.2, ×2 36.6, ×4 38.0, ×8 34.3; forbidding the token "Wait" without touching weights gives
  37.6 %, the same. shawnghu (LessWrong, Fable agent repeating MIT's eight experiments): 3-11 neurons, the same
  "Wait" brake plus a push to `</think>`, MATH 8 → 60 while MMLU-Pro / GPQA barely move (40 → 42-45). Two things to
  note, not to claim: (a) in the 0.8B the closing mechanism is local (a neuron) while in the 9B it is a spread direction
  (§43.23: no handful of neurons; §43.22: the direction cannot be re-derived in the 0.8B); (b) their useful window is
  wide (×2-×8 all > 34 %) where ours is narrow (×2 breaks): scaling an existing column keeps the model's geometry, adding
  a constant vector does not. Both are hypotheses for a later line, not findings.
- **Would refute it.** A behaviour that a learned weight edit produces but no fixed vector at any scale approaches.
- **Depends on it.** `engine/contrast_vector.py` (per-layer increments for llama.cpp), the scale grids in plans.

## T4. Creating a capability without a donor, or with a one-shot gradient-free shortcut, fails at the sizes we tried
Two claims with different evidence (split 2026-11-11 after an outside review; before, the title mixed them).
**Status: supported, narrow scope** — one model (0.8B), one task family (arithmetic), plus one 9B case. **Narrowed**
(§43.18): efficiency at a fixed budget — closing in time — *can* be obtained without donor or training.

**What counts as a "new capability" here (operational).** A task from a fixed bank where the base fails at any
budget because the answer is direct (3-shot, no reasoning tokens): H6 `mul3x3` (base 28-30 %) and the H7 tasks that
pass G1/G2 (§41.1). On reasoning benchmarks a gain is *not* a new capability unless the base also fails with a large
budget (T2: at 32k tokens the base already solves MATH). The bank is arithmetic only; a second, non-arithmetic family
is missing and is needed before "supported" means more than "supported for arithmetic on 0.8B".

### T4a. Without a donor
- **Claim.** No route that uses only the base (its own traces, its own directions, no stronger model's weights or
  outputs) has created a new capability.
- **Evidence.** The only donor-free route that passed anything is the "conclude" direction (§43.17-43.18), and it
  creates no capability (see the exception below). Every other route had a donor: cf5/cf6 write MiMo's behaviour
  (§43.10); transplants and LoRA come from fine-tunes (§43.8, §43.11-43.12).
- **Would refute it.** A donor-free edit that passes a pre-registered criterion against a same-norm placebo on a task
  of the bank above. A better donor does not bear on T4a; it bears on T1.

### T4b. Without gradients, in one shot
- **Claim.** The *where* of a learned change can be reused; the *how much* is built by the optimizer over the whole
  training and is not available from one-shot or linear shortcuts.
- **Evidence (0.8B arithmetic).** A training's change lives in a narrow subspace shared across seeds; rank 20
  recovers most of it (§40.3). But: extrapolating early checkpoints fails (§40.2); one gradient fails (§40.4);
  training only the lane's coefficients learns about as well as a *random* rank-20 basis (§40.6); a learned jump from
  step k to 200 never beats stopping at k (0/7 tasks, §41.3; fitted on only 8 tasks per fold); summing 7 task changes
  loses long arithmetic and TIES does not help (§42.2-42.4) — those changes were already rank-20 (ΔW = U·C), so
  restricting rank does not remove the interference of a sum; pairs do sum (B1, median retention 0.89). Closed-form
  synthesis of language models fails the gate (§30-§33). Caveat: H6's gate G2 was not met (+23 points, 30 asked, §40).
- **9B, with a donor.** Closed-form writing of MiMo's behaviour into `down_proj` (cf5) reaches MATH 17/36 and GSM8K
  25/48 and **beats its same-norm placebo** (10/36, 14/48; McNemar MATH 8/1 p = 0.039, GSM8K 12/1 p = 0.003; §43.10).
  So it is a real but small effect, not a zero; the route was closed because its ceiling stays far below
  distillation (27/36, 41/48) and the layer-by-layer linear fit, not the matrix written, sets that ceiling.
- **Iterative search without gradients: not reachable here (H6g, §44.1).** The SPSA estimator matches the exact
  gradient only at rank 1 in fp32 (rank 20 and all bf16 fail the ratio gate), and with exact backprop the rank-1
  coefficients themselves stay below the bar (R 0.035 `down_proj`, 0.192 all; 0.25 needed), so no SPSA arm was run.
  A few preconditioned closed-form writes per round (H6h, §44.2) also fail: R 0.01 / 0.08 / 0.04 against 0.25.
- **Would refute it.** A gradient-free edit that passes a pre-registered criterion against a same-norm placebo on a
  task of the bank **and costs less compute than the equivalent training** (forwards per example, counting a backward
  as 2 forwards). A gradient-free route that works only at a much higher cost narrows the claim, it does not refute it.
- **Where the outside frontier is (added 2026-11-13).** iWSP (Mehta & Singh, MSR; `csinva/weight-space-programming-
  autoresearch`): an agent writes weights and features in Python from scratch, no training, no pretrained weights, one
  validation score per iteration, 30-50 rounds. Algorithmic tasks: 26/26 runs at 100 % on a hidden set, and 100 % again
  when a new session rebuilds the model from a ≤ 200-word description; the solved models have a median of 166 parameters
  with Opus 5.5 as the writer, 41,600 with Sonnet 5.5 and 100,000 with Haiku 5.5 (same tasks, same loop: the writer's
  size decides how compact the written algorithm is). TREC 87.3 % (beats majority in 3 of 4 NLP tasks). Brain
  encoding: early visual cortex 0.443 vs 0.400 for the best interpretable baseline (ties AlexNet; loses in ventral,
  0.317 vs 0.391); the story-fMRI gain (0.067 → 0.104) was a position confound, caught by the legibility test, not by
  the benchmark. Together with Beaker and shawnghu the picture is one line: **writing from scratch works for a closed
  algorithm or an interpretable feature; whatever has to be learned from data still needs training or a donor.** That is
  T4 stated from the other side. Missing in all three: a same-size random control (irrelevant at 100 %, pertinent for
  brain encoding and for the "Wait" neuron). Ok & Lee (POSTECH, 17 Nov 2026; reading list) add the largest from-scratch
  run so far: three frontier LLMs given ten ARC examples all design a plain-text fact table instead of a Transformer
  (54.4 % on 3,530 ARC items, written Transformers below BM25 in 20/23 runs, fine-tuned pretrained 0.8B 82.1 %) — the
  knowledge written is the writer's own, and the donor-plus-gradients route stays 28 points ahead.

### The exception (§43.17-43.18)
Concept Country's "conclude" direction, read from the base's own finished traces (last 48 tokens before `</think>`
minus the rest, `contrast_vector` with `think`/`focus` rows), applied as a control vector ×1.0: sealed GSM8K 43 →
87/100, MATH 4 → 25/50 (gained/lost 46/2, 21/0; p < 1e-5, two graders each); the same-norm random placebo falls
*below* base (23/100, 2/50). It contradicts neither claim: it creates no knowledge, it removes a completion obstacle
(T2); with 32k tokens the base already solves MATH.

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
  gain (§43.10). Conversely, placebos stayed at base in every result we claim (§43.13, §43.15), or fell below it: the
  same-norm random control vector costs 20 GSM8K items (§43.18), so a push of that size is harmful unless aimed.
  Nine seeds, not one (added 2026-11-14, §45.10): eight more same-norm random directions on the 64-item MATH screen
  score 3-8 (base 8, vector 38), none reaches half the gain, none beats the base, the vector ranks first of nine.
- **Outside evidence.** Concept Country makes the same argument and runs the control; the MIT post does not publish
  one; Hayashi's controls were not effect-matched. Hu et al. (2611.09382): on 0.8B dev, edit 20/48, base 6/48, equal-norm
  random direction 10/48 — a random push gets part of the way; no held-out placebo, and dev gains often vanish held-out.
  Their own search variance makes the same point: three independent searches on MATH gave +29.45, +5.00 and +27.30
  (1,023, ~1.04 M and ~1.98 M values changed), and 6 of 16 runs found no improvement — one search is not evidence.
  Kocabay, *Weight-Space Reading* (13 Nov 2026): the same rule on the reading side — LLM readers given weights
  predicted a behaviour change for 53-88 % of 40 edits that changed nothing in Muse-30B (6 of them no-op controls),
  so a reader's hit rate means nothing without inert edits and a baseline that exposes guessing.
- **Depends on it.** `placebo_lora.py`, `contrast_vector.placebo`, `PROTOCOL.md`, exact McNemar in every verdict.

## T7. What weights alone can and cannot tell
**Status: established.**

- Algebraic facts about matrices (a LEACE projection zeroes uᵀW; Weyl / Davis-Kahan bounds on singular values and
  subspaces) are true by construction and say nothing about behaviour; for the layer-14 LEACE edit the Davis-Kahan
  bound is vacuous (90°). Behaviour needs forward passes on held-out items.
- Spectral energy (T1) is the exception that earned its place: it is a *prediction* about behaviour, checked 4/4.
- Self-certifying checks are void: our own retraction of "bit-exact parity" and "certified LLM" claims (§26).
- Blind forensics from weights only (§43.20, Qwen3.8-27B TURBO): where a Heretic ablation starts (layer 26, exact) and
  that the rest is an ordinary fine-tune (profile Spearman 0.93) are readable from the fused model; the ablation's rank
  (13, predicted 1) and its end (hidden by later changes) are not. An edit alone is read exactly (concept27b).
- **Outside evidence (Kocabay, *Weight-Space Reading*, 13 Nov 2026, reading list; code released):** the direct test of
  this thesis with LLM readers and a behavioural oracle. Given a target's weights, a surgical edit and a Python sandbox
  (no forward pass), Qwen3.8-27B predicts up / down / same for five behaviours of subtle edits to its own weights at
  balanced accuracy 0.730 vs 0.578 from a text description (+0.152, CI [+0.056, +0.256]; 10 changed edits, one of four
  pairs significant); recall of real changes 0.667 (text 0.278) at the price of specificity 0.793 (0.878). What the
  paper puts first: destructive edits are damage detection (another family reads them as well as the model itself,
  0.594 vs 0.593); no self-reading advantage survives the crossed contrast; and on Muse-30B 40 edits including 6 no-ops
  changed nothing (post-sublayer RMSNorm cancels uniform scaling) while readers called 53-88 % of them changed. Reading
  for T7: weights add *some* information about an edit's effect, measurable only against guessing, inert edits and the
  architecture; behaviour remains the ground truth, which their own protocol needs to score anything. Matches §43.20
  (an edit's place read exactly, its rank and its effect not) and §45.3 (the norm row changes what a write does).
- **Outside evidence (Chaudhuri, X thread, 14 Nov 2026, reading list; no code yet):** the *gradient route itself* is
  readable from the input-embedding matrix. Rows of tokens never in the data do not move; a row seen in one step moves
  by ≈ 31.8 × that step's learning rate in nearly every coordinate (Adam with no history keeps only the sign), so it is a
  timestamp; untouched rows shrink by (1 − lr·wd) per step, exact in fp32. Claimed: exact token list, schedule and step
  count in 15/15 fine-tunes, a public fine-tune replayed bit for bit on 25.5 M weights, 38/40 swapped documents caught.
  Of the public-model claims we checked one: Pythia-1B's config says `lr 0.00025`, its card says 3.0e-4. Two lessons for
  this thesis: (1) a weights-only read can be exact when it reads the *optimizer's* arithmetic rather than behaviour —
  our closed-form writes leave none of these marks, which separates them from fine-tunes (and from the implanted
  neurons of T8 only if the implanter also trained the embeddings); (2) rows that belong to no token are a base-free
  reference inside the checkpoint, the thing our implant check lacks. Our own study filed MiMo's top-changed embedding
  rows as "rare junk tokens: noise" (`lab/probes/delta_patterns/h5_plan.md`); by this mechanism they are the timestamps.
  **Tried on the one donor we hold in HF form (§45.18, 2026-11-17, prediction row 39 ❌):** Jackrong's distill of
  Qwen3.5-9B has `embed_tokens` and `lm_head` identical to the base, and only the linear projections differ — a merged
  LoRA, so no optimizer footprint can exist there. The mechanism stays untested by us; the per-tensor `equal` that
  found this is itself the cheapest weights-only read in the repository (training recipe from the checkpoint).

---

## T8. Open questions (being tested or next)
- **Compile the "conclude" direction into weights** (confirmed as a control vector, §43.18): **passes on dev (§43.19)**.
  A closed-form rank-1 write per layer into `down_proj` (31 layers, no gradients), merged into the Q6_K checkpoint with
  no inference flags: GSM8K 50 → 87, MATH 6 → 34, not different from the control vector (p 0.45 / 0.73); the same
  write in a random direction damages (28 / 6). **Confirmed on sealed set 2 (§43.21):** GSM8K 46 → 89, MATH 7 → 26
  (43/0, 19/0, two graders each), same as the vector (p 1 / 0.29), placebo merged below base (30, 4). Retention: the
  merged model costs far less outside maths than the vector (HumanEval-20 19 vs 15/20, perplexity ×1.18 vs ×2.3),
  probably because Q6_K keeps the direction but ~half the magnitude. Re-deriving the direction in the 0.8B fails: it
  closes 1/256 traces at 4 096 tokens (§43.22). Its MLP part is not a handful of neurons: median k₅₀ 119 vs
  128 for a random split (§43.23; one exception, layer 26, 4 neurons). **Redone with the corrected vector (§45.4,
  §45.8, 2026-11-13):** the first vector's layer 31 was the post-norm row (§45.3); the re-derived vector confirms on
  sealed set 1 (86 / 33) and its compilation on sealed set 2, second use (89 / 25 vs base 46 / 7, placebo 34 / 5,
  HumanEval-20 19/20). Paired on the same 150 items the corrected layer 31 changes nothing in correctness (3/3, 3/4) and
  halves the loop tails: the signal was in layers 1-30. Composition with the Jackrong adapter is redundant in accuracy
  and additive in cost (§45.7: tokens ×0.39, cuts 9 → 0). **A closing sentence in the prompt does not do it (§45.9):**
  base + "stop reasoning once you have the answer" gives GSM8K 50 → 37 (p 0.019), same tokens, more cut answers; a
  neutral sentence gives 29. The thinking phase does not react to an instruction in the user turn at this budget, and
  any added sentence lengthens it. Prediction row 28 failed (we expected 62-75). Still open: other budgets, other
  models, sampling, an instruction placed inside the thinking or as a system prompt. **Not a token budget (§45.11,
  §45.13, 2026-11-14/15):** at 8,192 tokens the base repeats 72 / 21 item by item (§39.6), the vector scores 85 / 37
  with a quarter of the tokens (16/3, p 0.004) and the compiled weights 90 / 41 (19/1 vs base; 6/1 and 6/2 vs the
  vector, equal or better); the compile's half dose closes early less often than the vector (6 vs 13 finished-wrong).
  Prediction row 31 (base 82-92 at 8,192) failed; row 32 partial, the compile did more than predicted. **On Phi the budget
  explains more (§45.17, 2026-11-16):** at 2,048 tokens (2×) the base goes 59 → 81 / 6 → 26, the merged model keeps +10 on
  GSM8K (12/2, p 0.013) and +3 on MATH (7/4, not significant; the base still cuts 24 of 50 there), the vector +4 / +9
  (13/4, p 0.049). Every merged win but two is an item the base cut: AHA's ceiling rule (gain ≤ base truncation rate) held
  on both benchmarks, as it did on Qwen. The family-level reading: the direction is "finish after reasoning" on both, but
  how much of the fixed-budget gain survives a larger budget is model-dependent (Qwen most of it at 4×, Phi half of GSM8K
  and little of MATH at 2×). Prediction row 36 partial (14 of 18 readings).
- **The conclude direction across models through activation maps (gate 3b, §45.12, 2026-11-14):** the 9B's contrast
  mapped per layer into Qwen3.5-0.8B (gate 1 maps, fitted on raw GSM8K text, not refitted) finishes 83/100 GSM8K
  answers at 2,048 tokens where the base finishes 0, and 7 are correct (7/0, p 0.016); the same-norm placebo 3 / 1,
  the embedding-map control 0 / 0, the 0.8B's own contrast over the same 9B texts 37 / 13, the MIT Exp 8 edit 13 / 13.
  It passes the pre-registered rule (gain 10 ≥ half the reference's 16), and the prediction (row 30: would not pass)
  failed. What crosses is the **stop** component, not the **finish** component: the ported answers have a median of 8
  characters of thinking and 8 % of the finished ones are right (own 35 %, Exp 8 100 %, the 9B with the same direction
  ≈ 90 %). T2's two parts (stop when done, and *knowing* when done) separate here; the map carries the first. Open:
  a rule that counts correct-among-finished, a scale between 1.0 and 2.0, other pairs. [2026-11-15, §45.14:] the dose
  curve (finished 0 / 3 / 15 / 50 / 52 at 1.0 / 1.25 / 1.5 / 1.75 / 2.0) shows a threshold, and from 1.5 up every
  finished answer skips the thinking phase; the correct-among-finished rule fired on direct answers (≈ 40 % right
  without thinking), so the thinking length of finished answers must be a criterion, not a report, in any next rule.
- **The conclude direction on a second family (Phi-4-mini-reasoning 3.8B, §45.15, 2026-11-15):** same recipe and code on
  another lab's model (`phi3`, fused `gate_up_proj`, tied embeddings, another tokenizer, a template where the model writes
  `<think>` itself), 1,024-token budget (it does not run out at 2,048), scale 0.5 chosen on screen items: dev GSM8K
  59 → 85, MATH 6 → 19; **sealed set, first use for this model: 59 → 82, 4 → 19** (27/4, 16/1, two graders each), tokens
  −34 % / −12 %; same-norm placebo 58 / 6 (dev 47 / 5, never above base). Every finished answer at 0.5 keeps its
  thinking (median 2,072 → 1,318 characters); at Qwen's scale 1.0 the model skips it (screen 5/64 with 63 finished): the
  T2 "stop" regime of gate 3b is reachable in Phi by dose alone. What changes between families: the scale (vector norm
  27.2 vs 18.1) and the per-layer profile (peak at layer 30, last row 0.87× the one before; Qwen ≈ 2×). What does not:
  the placebo's sign. Prediction row 34 partial (direction, controls, verdict right; magnitudes larger than predicted).
  **Compiled (§45.16):** the same op writes the Phi direction (× 0.5) into `down_proj` of 31 layers, G0 0.044, cos(written,
  dW) 0.74 after Q6_K; served with no flags: dev 81 / 19 (vector 85 / 19, base 59 / 6; C1-C4), sealed set 2 (first use)
  base 62 / 7 → merged 86 / 14 (26/2, 7/0), vector 88 / 15 on the same items (2/4, 2/3 against it: equal), compiled
  placebo 60 / 2; no finished answer without thinking; HumanEval-20 11 → 16 (n = 20, reported). Row 35: 18 of 22 readings
  in band. The compile primitive now holds on two families; what differs is the dose written (× 0.5), not the op.
  **Budget doubled (§45.17):** base 81 / 26, merged 91 / 29 (+10 GSM8K, p 0.013; MATH +3, n.s.), vector 85 / 35; the
  gains are the base's cut items (T2). Open: a 4,096 / 8,192 arm for MATH, the sealed set at 2,048, sampling.
- **A third family where the gain does not confirm at scale 0.5 and does at 0.35 (DeepSeek-R1-Distill-Llama-8B, §45.19-45.21, 2026-11-17):** same
  recipe, code and pre-registered criteria (`llama`, 2,048 budget, scale 0.5; 1.0 breaks it as on Phi). Dev passes
  (GSM8K 74 → 76, MATH 22 → 32 with 13/3, p 0.021, two graders; tokens −46 % / −26 %; placebo 76 / 23) and **sealed set 1
  does not confirm**: base 83 / 28, vector 76 / 26 (7/14, 7/9), placebo 82 / 23; pooled over both sealed sets 24/29 and
  17/14. Mechanically the direction behaves as in Qwen and Phi (cuts 13 → 2, −41 % tokens, placebo inert, no finished
  answer without thinking), but the items it loses are ones the base finished right and it finishes wrong, as many as it
  recovers from the base's cuts (sealed 1 GSM8K: 14 lost, 0 cut, 0 loops; Qwen and Phi lost 2-4). Compiled with the
  unchanged op (G0 0.030, cos 0.74, ‖dW‖/‖W‖ 0.012) the merged file equals the vector item by item on sealed set 2 (7/7,
  4/5) and like it does not beat the base (75 / 23 → 77 / 27); HumanEval-20 14 → 13. Rows 37 and 38 ❌ on the primary
  reading: the first dev PASS of the series the sealed set rejected. What this moves: the accuracy gain of T2's exception
  is shown on two families of three; the compile primitive (T8) and the same-norm placebo's null (T6) on three. On this
  family the direction is a token saving at equal accuracy. Hypotheses, not measured: a base that needs its long
  thinking more (sealed base 83 with 13 cuts: ceiling 13, the vector loses 14) and a different profile (last row 3.2×
  the one before). Untested: other budgets, sampling (card says T 0.6), scales between 0.25 and 0.5.
  **Dose (Part C, §45.21, 2026-11-17):** at 0.25 and 0.35 on dev the vector gives 84 / 31 and 81 / **37** (15/0 on MATH);
  a pre-registered gate picks 0.35; **sealed set 3 (200 + 100, first use) confirms**: base 158 / 40 → vector 156 / **58**
  (14/16, p 0.86; 20/2, p 0.0001 with both graders), same-norm placebo 152 / 38 (n.s.), tokens −31 % / −18 %, C3 (GSM8K
  ≥ base − 2) at its boundary. 19 of the 20 MATH items gained were cut by the base (58 cuts of 100; among the 42 it
  finishes, 40 → 39): T2's exception in its purest form. On GSM8K the 14 gained are recovered cuts and the 16 lost are
  base-correct items finished wrong, the same mechanism at half the rate (10 % of base-correct finished items vs 17-20 %
  at 0.5). So "dose, not direction" holds halfway: the lower dose halves the cost and lets the cut recovery reach
  significance on a sealed set; it does not remove the premature closure. The sealed gain is now shown on three families,
  on Llama-8B only at 0.35, only on MATH, only where the base cuts. Third sealed set spent on this family (the scale was
  chosen on dev after 0.5 failed on the first two; the gate was pre-registered). Row 40 failed on its primary reading (we
  gave "does not confirm" 60 %).
- **Rebuild-from-description test (idea from iWSP, not run):** do ≤ 200 words of recipe ("contrast of the base's own
  traces, last 48 tokens before `</think>` minus the rest, pre-norm last row, layers 1-31, scale 1.0") let a new session
  rebuild the conclude vector and get dev 83 / 35? Today we only know the *program* is deterministic. Cheap,
  descriptive, after the queue.
- **Implanted neurons (defence, §43.24):** with the declared base, a same-size rewrite of a few neurons ranks 1st of
  393 216 even inside a full SFT (D1 recall 1.0), but the SFT itself flags ~5 legitimate neurons, so it is a short
  review list, not a clean alarm (strict rule: not useful); pushes ≤ the fine-tune's typical change are invisible;
  without a base only a token-writing column is seen (0.6-0.7), not in the last layer. `model_report` check not added.
  **Lazarus (MSR, 11 Nov 2026, reading list)** writes exactly there: seven dormant neurons of the final FFN block as a
  triggered backdoor, closed form, and reports agent-built detectors at 91.4 % / 9.7 % FP without a base. Cheap next
  check (not run): a Lazarus-style edit on our Qwen3.5-0.8B (a few < 1 %-firing neurons of the last block, closed-form
  payload) through `implant_detect` with and without the declared base; prediction to be written before running.
- **Why RL changes do not fit a small adapter** (LeoZotos 7.5 %, MiMo 3.8 %) while SFT distills do. One RL donor so far.
- **A useful non-math donor**: none in the first batch (§43.16: two code models are the base, one is a high-rank merge,
  the medical RL model is uncertain at 25 %).
- **Quantization × edits:** all our numbers are Q6_K; quantization alone moves base loops (§39.6).
- **Three model families tried (Qwen3.5-9B; Phi-4-mini-reasoning since §45.15; DeepSeek-R1-Distill-Llama-8B since
  §45.19), greedy only.** The conclude direction and its compile into `down_proj` are the only results shown outside
  Qwen; the accuracy gain is confirmed on a sealed set on all three, but on Llama-8B only at scale 0.35, only on MATH and
  as recovery of the base's cuts, with GSM8K at base − 2 (§45.21; at 0.5 it saves tokens at equal accuracy, §45.19-45.20);
  the transplants, the activation maps and the implant checks are Qwen-only. Nothing here is shown for sampling.

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
Programming?* · Khamis, Tutunov & Bou Ammar, *AHA! Who Needs an LLM to Program an LLM?* (haithamb.github.io/aha,
15 Nov 2026; Exp 8 with the direction from the model's own rewind-and-resample pairs and Barry's LP) · Alapan Chaudhuri,
*A model's weights keep a record of its own training* (X thread, 14 Nov 2026; AdamW's footprint in the embedding rows:
token list, LR schedule, step count, document steps read back from a fine-tune; no code yet) · Kocabay (Poindexter Labs),
*Weight-Space Reading* (alphaXiv, 13 Nov 2026; LLM readers predict an edit's behavioural effect from weights: 0.730 vs
0.578 balanced accuracy on subtle self-edits, false alarms on inert edits, damage detection on destructive ones;
github.com/suayptalha/wsr) · Marasini, Khadka & Rahimi, *WeightBench* (arXiv 2611.18553, 13 Nov 2026; 42 tasks, five
tracks; Opus 5.5 macro 0.535; abstract only) · Hu et al., *Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations*
(arXiv 2611.09382) · Hayashi, *weight-space geometry* (hayashiyus.github.io/blog/weight-space-geometry; theory, mostly
proposed experiments) · Hu et al., LoRA · Ilharco et al., task arithmetic; Yadav et al., TIES · Belrose et al., LEACE ·
Meng et al., MEMIT · Irie, Schlag & Schmidhuber, fast weight programmers · *A Survey of Weight Space Learning:
Understanding, Representation, and Generation* (arXiv 2603.10090, cited in `docs/wsplang_0.7_spec.md` §9) — the map of
the field: our work is mostly *understanding* (spectral energy T1, shared subspaces §40) plus edits of one model
measured against placebos, which the survey does not cover; its *generation* branch (hypernetworks, generative models
over model zoos) is trained, so it is another route than the gradient-free synthesis T4 rules out · Lu et al., *The
Surprising Effectiveness of Deleting Weights in LLM Post-Training* (arXiv 2611.10866, NeurIPS 2026) — a learned binary
mask zeroing < 0.05 % of `gate_proj` weights matches full fine-tuning on GRPO/SFT/on-policy distillation at ≥ 7B, with
little forgetting: support for T2/T8 (what post-training changes on these benchmarks fits in very little; rank-1 per
layer in §43.19). Caveats: the mask is trained with gradients; no same-fraction random-deletion placebo (T6), and LoRA
r1 also matches, so the tasks need little information rather than masks being special · Zheng et al., *Correctness
vs Efficiency: Behavior Control with Weight-Space Extrapolation in Code RL* (arXiv 2605.28751, NeurIPS 2026) — two RL
runs from one SFT checkpoint differing only in reward strictness; their weight difference steers reward hacking, and
extrapolating past the strict run cuts resource-limit failures 14.5 → 8.0 % without more RL; too far and outputs
collapse (repetition, reasoning that never ends): T3's narrow window in weight space, and T5 (a "correctness tax", total
solve rate unchanged). Read from the author's thread; no same-norm placebo there (T6) · Winters, *Weight-space
programming* (withbeaker.ai/blog, 11 Nov 2026) — agent loop on Qwen3.5-0.8B, neuron 573 (T3 note) · shawnghu,
LessWrong (10 Nov 2026) — MIT's eight experiments repeated with a Fable agent, raw outputs on Hugging Face
(`shawnghu/fable-weight-space-programming`); the author says he did not review outputs or method closely · Mehta &
Singh (MSR), *Interpretable weight-space programming builds transparent transformers and brain encoding models* (iWSP,
Nov 2026; `csinva/weight-space-programming-autoresearch`) — T4 note · Naik, Liu & Gangal, *Auto Weight-Space
Programming, Part 1* (atharva-naik.github.io/blog/awsp-part-1, 13 Nov 2026; alphaXiv copy submitted 15 Nov 2026 for
versioning, same abstract: Qwen3.5-4B, 99.8 % on the four-way logit-lens test; code, LoRAs and item files released;
co-author's thread 15 Nov 2026, x.com/VarunGangal/status/2121771960300634620, frames it as "sparks of life for a
trainable regime", nothing beyond the blog) —
the **reading** side: a rank-32 LoRA teaches Qwen3.5-4B to give the logit lens of 128-number slices of its own
`down_proj` columns pasted as digits (99.8 % four-way on held-out neurons, chance 25 %; shuffled numbers → chance);
naming the tokens without candidates is rare (6.6 %), **writing weights fails** (1.0-1.4 % of requested tokens vs
27.7 % for an analytic writer), predicting activations and circuits is at chance, and single neurons do change
sycophancy but a linear probe over the weights cannot say which (their 7.4 = our T7 and T3: the effect needs the
forward pass). Their framing matches ours: the tool (logit lens) is a shortcut, the oracle is the behavioural
experiment; "no self-advantage" — reading another model's weights works as well, so the *auto* in auto-WSP is not
yet shown to matter. One seed per run, a four-way choice with same-layer distractors; honest "needs more effort"
list · rizumita / CAPH TECH, *Qwen3.8 Flash Next: Chinese mixed into Japanese answers removed by rewriting the output
layer, 27 % → 0 %* (zenn.dev/caphtech/articles/qwen38-zh-ban, 13 Nov 2026; tool `CAPHTECH/kestrom`) — 25,451 `lm_head`
rows of Chinese tokens overwritten with the bytes of an unused row (id 248077, past the 248,077-token vocabulary);
byte diff shows only those rows changed; greedy outputs on 60 prompts identical to the base where the ban did not fire.
A hard constraint written into the weights, not a learned direction: the right control is output identity where it
does not fire, and they ran it. Gaps they state: 60 medical prompts (27 % has a 17-39 % interval), mixing measured
under sampling (temperature 1.0) while identity was checked greedy, logit margins unmeasured, token-level bans can
break Japanese words spelled with banned tokens (seen on Qwen3.5-9B), no GGUF/llama.cpp run. Gap they do not state:
correctness of the 40 JMMLU answers before and after is not reported (T5). Side note worth keeping: mlx-serve gave
different answers in 49/60 greedy reruns without reloading the model (prompt-cache numerics) — same lesson as our
§45.11 (`-np 2` batching: two servings of the same base differ item by item) · emeraldant, *WSP-Gym* (thread 14 Nov
2026; emeraldant.github.io/WSP-Gym, code and test sets on GitHub) — the first **trained** weight-writer we know of:
Inkling 2 Small (Thinking Machines) with a LoRA via Tinker, supervised on 57,600 synthetic small networks with one
planted fault (weights pasted as text plus one line on the function; answer = at most 3 edits, no tools, no forward
pass), then RL; repairs 83 % of held-out networks in ~40 tokens with no reasoning (untrained with reasoning 76 %,
GPT-6 Astra 97 % with ~110-170 reasoning tokens, weight-statistics heuristic 4 %). Generalization drops where it
should: larger networks 70 %, unseen function families 59 %, gradient-trained networks 67 %, **unseen fault types
20 %**. Disguised networks (permuted, rescaled, split neurons, decoys, noise) rule out memorization of weights. Our
reading: closed-form toy networks with a known fix, so it sits with iWSP in T4 ("what has a closed algorithm can be
written"); the 20 % on unseen faults is the same wall as Naik's writing failure and our H6/H7 — the skill learned is
the fault catalogue, not weight semantics. No LLM-scale model was edited · Naik (thread 14 Nov 2026,
x.com/AtharvaRNaik/status/2121629639458324625) proposes a WSP **benchmark suite** and names the baseline: "auto-research
agents equipped with interp tools" — the same bar we hold ourselves to (the agent must beat the gradient / control-vector
route on behaviour, not merely produce an edit). He quotes jfonsecarivera (13 Nov 2026): Sloef's *sparse rewiring*
(trains only when a neuron fires, not what it writes; 89 of Qwen3-8B's 442,368 neurons reproduce an SFT persona) applied
to risky financial advice gives emergent misalignment with **47 neurons** (misaligned answers 0.3 → 14.5 % on the 8
standard questions); the neurons promote "risky / gamble / exploit / manipulate / brutal" and 22 of the 47 fire on the
framing word of a request ("should") harder than anywhere in 131k tokens of normal chat; on a LoRA the misalignment
drops 26 → 0 % when the adapter is switched off at chat-template tokens, and masking those tokens in training makes
the model find another route. Trained edits, not WSP; relevant to our implanted-neuron defence (T8, §43.24: a
same-size rewrite of a few neurons ranks first when the base is declared) and to T3 (a behaviour as a push through few
sites, which moves when blocked) · Zeng, Severi, Bullwinkel, Horvitz & Papailiopoulos (MSR), *Lazare, veni foras:
Reprogramming Dormant Neurons to Backdoor AI Agents* (arXiv 2611.15118, 11 Nov 2026) — closed-form edit of rarely-firing
neurons in the **final** FFN block, one payload token per neuron when a trigger is in context: 7 of 12,000+ neurons of
Nemotron-Terminal-8B fitted on 23 trajectories, payload on ≥ 98.8 % of triggered test trajectories, no leakage seen on
benign ones; Qwen3.6-27B and Gemma-4-31B "equally vulnerable"; agent-built detectors with no reference model flag 91.4 %
of edited checkpoints at 9.7 % FP. The adversarial twin of our implanted-neuron check (T8, §43.24): ours ranks a
few-neuron rewrite first when the base is declared, and without a base sees only token-writing columns **not in the last
layer** — the block Lazarus uses. Idea credited to Claude Code (Opus 4.5, Feb 2026) · Tensor-Slayer / areu01or00,
*Reprogramming dormant neurons: Qwen TTS* (thread 14 Nov 2026; repo `areu01or00/qwen-tts-neuron-mapping`, WIP) — an
agent maps 417 talker-MLP neurons of Qwen3-TTS-0.6B (0.48 %; 304 of them in layer 27; 55-61 "dormant" = strong in < 1 %
of speech frames) and pushes their `down_proj` columns along one fixed direction per layer (rank-1, ×8-12): breathing,
laughs, whispers, a second speaker; ×16+ gibberish, ×10 on built-in voices stuck 13/36. Listening only, no WER or quality
metric, no random-neuron control: T3's narrow dose window again, and a demo rather than a finding by T6 · Ok & Lee
(POSTECH), *Weight-Space Programming Doesn't Need Transformers* (alphaXiv, 17 Nov 2026; code and 56 written models
`effl-lab/Weight-Space-Programming`, MIT; project page wsp-no-transformer.github.io; thread
x.com/HyunjongOk/status/2122518170359706090) — the **from-scratch** branch, not editing: three frontier LLMs (Opus 5.5,
GPT-6.1 Sol, GPT-6 Astra) in a sandbox with ten labelled ARC questions either write the weights of a given architecture
or design their own; lineages keep the best-validation replicate per generation (5-8 generations). Free to design, all
three write a **sparse table of plain-text facts** compiled by a build script and read by fixed code, 54.4 % on 3,530
ARC test items (Opus, 3.35M "parameters", chance 25 %) against 43.1 % for the best of 15 released architectures the same
LLM wrote (Qwen3-1.7B shell, 48.7 %) and 36.1 % for a written two-layer Transformer; their own associative-memory table
(pair rows at computed addresses, 120 rows per question) 52.4 % and 17× faster on GPU than a size-matched Transformer.
The numbers that matter for us: written Transformers score **below BM25 over the same text in 20 of 23 runs** (36.1 vs
44.1), and Opus's first table beats BM25 by 3.7 points only (50.7 vs 47.0), so the written "model" is mostly the text
and the architecture is a retrieval index; a fine-tuned pretrained Qwen3.5-0.8B gets 82.1 % and gradient training from
scratch ≤ 32.7 %; on raw CIFAR-10 pixels ≤ 28 % (96.1 % only on DINOv2 tokens); and when the writer starts from a
pretrained checkpoint it changes **under 1 % of the weights, mostly readout and embeddings**, and full overwrites never
pass 72 %. Deleting one written fact moves 0.5-1.8 answers in a table against 4.8 in a written Transformer
(superposition cost, read from the writing side). Our reading: T4 stated once more from outside — writing from scratch
works where the knowledge is already explicit in the writer (here, science facts the LLM knows) and loses by 28 points
to a pretrained donor plus gradients; the "1,800× fewer parameters" is against a written Transformer that keyword search
beats. Nothing here edits a pretrained LLM, so it does not touch T2/T3/T8; the pretrained-checkpoint rows agree with
Naik's writing failure and with our rank-1 compile being the size of what hand-writing can place. The term "WSP" now
covers two different things (editing a donor; designing a table), worth keeping apart when reading claims · alphaXiv /
Daniel Kim, *minimizing-wsp-edits* (OpenResearch project, 17 Nov 2026; marimo notebook `alphaXiv/weight-edit-minimization`;
thread x.com/askalphaxiv/status/2122744324354097172) — an agent (Inkling 2 Small) shrinks two MIT WSP blog edits. Exp 8
on Qwen3.5-0.8B: the notebook recovers Astra's 130,188-entry edit as **one direction u subtracted through 64 `down_proj`
neurons per layer** (W = W₀ − 6·u·(m ⊙ uᵀW₀), layers 15 and 19) and keeps the 4,096 largest entries per layer (8,192,
6 %): 58.2 vs 58.6 % on a 256-problem MATH-train screen (one sample, T 0.6, 32k cap, SE ≈ 3 pp), 54.2 % on MATH-500 × 4
held out against the blog's 48.1 % from another evaluation; side effects = cut rate and a 16-gram loop share only. Car-ramp
PPO actor: 6,073 entries → 55 functional (the rest zero embedding rows) → 11 by drop-one audit, and 10 written from
scratch; every rung solves the level. We checked the Exp 8 claim on our own recovery of the edit (§9.1,
`data/exp8_edit_recovered.npz`): rank-1 per layer indeed (s₁/s₂ 1,771 and 1,552, all 64 columns collinear with u, the two
layers' directions at cos 0.85), and the 4,096 largest entries hold 74-76 % of ‖ΔW‖² — so the pruning keeps most of the
push, and our §16.2 note that called the edit "not rank-1" was wrong (corrected there). Our reading: the blog's edit is
the same primitive as our compile (T8: rank-1 in `down_proj`, few neurons, the push magnitude is what matters, not the
entry count); the "no degradation" is a screen-resolution claim with no same-norm placebo and no other benchmark (T6); the
car policy is a closed toy network (T4), where 10 weights written from scratch beat PPO's 6,073 because the fix is a
known algorithm · laxman / bandr.ai, *DAWE* (write-up and code `bandr-ai/dawe`, 17 Nov 2026; quoted by Rikiya Takehi) — a
closed-form readout edit: ridge regression from final-layer states to the end-of-turn logit, added to one or two rows of
`lm_head` (2,557 / 5,064 values on Qwen3.5-4B), repairs a reply-format break (plan-first 20.1 → 88.2 %, retrain 88.7;
old harness unchanged; same-norm random row edit 68.1 %). 204 items, greedy, single runs, own scorer (± 5 pp), no
constant-bias control (they say so), tied embeddings, test aggregates visible to the agent. T8: the same primitive as our
compile, placed on the readout and carrying only the stop / continue component (what our port carried in §45.12); T2 /
T3 / T6 from outside. The control they lack is one we lack too: a constant logit bias on the stop token against our
"conclude" vector (roadmap gap 14) · rizumita / CAPH TECH, *kestrom inspect* (zenn, 17 Nov 2026; `CAPHTECH/kestrom`,
Mac / MLX) — backdoor reading from the fine-tune difference against the official base: per-tensor SVD (LoRA s₁ ≈ 1-2.5,
implants 5.85-25.1), token rises on plain text, component-to-token lens, trigger search, GGUF via a same-way requantised
base. Pre-registered, planter separate from reader: 0 / 32 clean flagged, 15 / 16 found, 14 / 16 targets, 4 / 16
triggers on Qwen3.5-0.8B / 4B; the miss is a double-quantised GGUF whose rounding buried the rank-1 change. Agrees with
§43.24 (with the base a rank-1 write is first of 393,216 neurons; a real fine-tune moves a few legitimate neurons a lot;
without the base only token-writing shows) and adds the lens and the requantised-base trick we do not have ·
oguzeldereli, *ttf* (ternary runtime + llama.cpp patches; `ttf-Qwen3.8-27B-GGUF` 6.74 / 8.20 GB; PolyForm NC) — Qwen3.8-27B
at 1.6 bits plus 72M tokens of recovery training: GSM8K 0.895 = FP16 (200 items), HumanEval 0.841 vs 0.933, TriviaQA on
Flash-Next 0.538 vs 0.776. For us a 27B that fits 12 GB for activation-space work (control vectors), not a weight-space
base: ternary matrices and recovery training make differences against the official weights unreadable (T7) and leave
our rank-1 writes no float target; the knowledge loss at 1.6 bits is itself a weight-space fact (what survives is coarse
structure). Full links in README "References".
