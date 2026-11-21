# DSpine and behavior extraction: verified findings and a small multiplication probe

Status 2026-11-09: research review and design only. **Nothing here has been executed.** The prototype code
in this directory compiles and its data sets are unit-checked (disjointness, ranges, answers), but the
training/evaluation paths are untested (no torch on the review box); they mirror the measured memory
patterns of `lab/probes/skill_delta/train_down23.py` and must be smoke-tested before any real run.
Owner's scope: research and prototype only; no integration of DSpine into WSPLang without measured benefits.
Revised after a static leak review: arm C added (conventional sequence model), teacher-label variant added,
split hygiene and dependence framing made explicit (section 3).

## 1. What DSpine is (verified against official sources)

DSpine is a parallel drafter for speculative decoding, from Tencent YouTu Lab with Shanghai Jiao Tong
University and Xiamen University.

| Claim in the proposal | Verified |
|---|---|
| Official repository | https://github.com/Tencent/DSpine (created 2026-10-15, Apache 2.0, C++/CUDA/Python, SGLang plugin) |
| Paper | arXiv:2609.36173, "Draft in Parallel, Condition Through Depth: Adjacent Causal Injection for Speculative Decoding" (2026-10-10), https://arxiv.org/abs/2609.36173 |
| Official model | https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16 (also tencent/Qwen3-8B-DSpine-b16) |
| Five-layer drafter | Confirmed: "a five-layer draft model for Qwen3.8-27B, with block size 16 and an 8,192-token sliding window" (model card) |
| ~2B parameters | Confirmed: HF metadata reports 2B params, BF16 |
| Block size 16 | Confirmed (model card and repo usage flags) |
| Uses intermediate hidden states | Confirmed and stronger than stated: the drafter **reads hidden states from five layers of the target model** on every drafting round (GGUF card and paper) |
| Predicts blocks in parallel, target verifies | Confirmed; on Qwen3.8-27B it additionally uses DSpark-style dynamic verification and ReplaySSM |
| GGUF + llama.cpp | Third-party community port: https://huggingface.co/aj9o9/Qwen3.8-27B-DSpine-b16-GGUF needs the `dspine` branch of https://github.com/Ajay9o9/llama.cpp; upstream llama.cpp does not support DSpine. Unofficial; not audited here |

Mechanism (paper): (a) **gated adjacent injection** - at every drafter layer, each position receives a gated
message built from its predecessor's predicted feature, so causal dependencies are carried through network
depth while all 16 positions update in parallel; (b) a **unified transfer space** built from the target
model's output embeddings, used both for the layer-wise injection and for predecessor-conditioned decoding;
(c) **layer-wise output-embedding supervision** that aligns the drafter's predicted features with the
target's token-embedding space, promoting useful predictions already in shallow layers; (d) a transition
cache that precomputes adjacent candidate-pair scores in parallel.

Results (not reproduced): on Qwen3-8B, mean acceptance length 7.83 at T=0 across seven benchmarks vs 7.55
for DFlash2 (same training data/budget); +23.3% SGLang throughput vs DFlash; the 27B card reports 2.7-3.8x
end-to-end speedups on agent benchmarks. These are serving numbers from the authors, on CUDA.

## 2. The decisive fact for WSPLang

**The DSpine drafter is not a standalone capability.** It reads the target's hidden states (five layers) at
every drafting round, its predictions live in the target's output-embedding space, and the target verifies
every proposed block. Remove the target and the drafter cannot generate at all. DSpine is a mimic trained
for inference acceleration, and the paper's own framing (draft -> verify) depends on the large model being
present and correct.

The proposal in the owner's message already states this ("its parameters are not directly portable"),
and the sources confirm it. What DSpine can give WSPLang is not its weights or its serving stack but three
ideas:

1. **Layer-wise readouts.** The paper's empirical study finds that early positions already form recoverable
   token predictions in shallow layers, and that predecessor information helps more when it enters earlier.
   This is direct evidence about *where* capability information concentrates - the same question our probes ask.
2. **The target's output embedding as a shared transfer space.** Supervising an auxiliary module so its
   outputs align with the target's token-embedding space is cheap and exact. Our cached-CE path already does
   the equivalent: `train_down23.py` trains through the frozen final RMSNorm and the full tied 248k-way head.
3. **Conditioning through depth (gated adjacent injection).** Relevant only for multi-token parallel
   prediction; not needed for a first single-position probe.

What DSpine does **not** provide: any method for converting an extracted behavior into a weight-space edit
of another model. Question 3 of the proposal (standalone weight-space modification) remains open and is the
business of the `skill_delta` surgery line, not of DSpine.

## 3. Proposed probe: 2-digit multiplication with a small Qwen target

Deliberately small, reusing the machinery and pinned artifacts of the running teacher experiment so the
numbers are comparable. Same model (Qwen3.5-0.8B @ 2fc0636), same seeded sets
(`lab/probes/closed_form_math/data.py`: train 2000 / selection 100 / holdout 100 / control add2, operand
pairs disjoint), same metric (free greedy generation, exact match before the first newline), same
persistence rule (save bf16, reload in a fresh process, re-measure; in-memory numbers do not count).

Four conditions (the three comparisons the owner asked for, plus the target itself):

- **Target baseline (A):** the unmodified model, evaluated by `skill_delta/evaluate.py`. Already being
  measured by the teacher experiment; do not rerun without need.
- **Arm H (conditioned auxiliary, DSpine-inspired):** `extract_hidden.py` caches, at each answer position,
  the outputs of decoder layers 3, 11, 19, 23 plus the final hidden state (~50 MB for 800 examples).
  `train_aux.py --arm h` trains a two-layer MLP (5,244,928 params) that maps the concatenated layer states
  to an update of the post-norm hidden state; logits through the frozen tied head; exact CE. Evaluation
  (`eval_aux.py --arm h`) is *dependent mode*: one target forward per generated token, the aux proposes
  without target verification - the gap vs the target measures what verification was buying.
- **Arm X (auxiliary without target representations):** identical training signal and parameter count
  (5,244,415 params, matched within 0.01%), but the input is the masked mean of the prefix token
  embeddings - it sees the full problem, never the target's internals, and no token order. Fully standalone
  at evaluation: no decoder layer runs.
- **Arm C (conventional small model, same dataset):** a GRU over the prefix token embeddings plus a linear
  head input (~2.89M params). C is NOT parameter-matched to H/X (5.24M): different architecture, count
  reported exactly, and it is a reference baseline, not a matched ablation. It sees order and the full
  problem without any target hidden state, and is also standalone. X alone would be too weak to be
  informative (mean pooling cannot represent multiplication); C is the real conventional baseline.

All arms share the frozen tokenizer, embedding table and tied head (static lookups, no target computation):
the only thing that distinguishes H from X/C is access to the target's computed hidden states.

Leak and hygiene review (static, before any run):

1. **No label leak through the hidden states.** Attention is causal: the state at position t-1 attends only
   to tokens 0..t-1, and the target token t never enters its own input (same convention as train_down23.py).
2. **Dependence is not extraction.** The hidden states at t-1 contain the target's own computation of the
   answer. A high arm-H accuracy therefore means the capability can be *read out* of the target's internals
   by a 5M-parameter probe - not that the aux holds an independent capability. That is exactly DSpine's
   nature. The informative contrasts are H vs X/C and H vs H-zero-hidden; the expected result is that H
   collapses without the target, which would confirm dependence rather than refute the probe.
3. **Exposure bias (report, do not fix).** Training states are computed over ground-truth prefixes (teacher
   forcing); evaluation states are computed over the aux's own continuations. Affects arm H only.
4. **Labels: ground truth by default, teacher outputs as a secondary variant.** GT answers the capability
   question ("can a small component learn 2-digit multiplication from these signals"). A teacher-labeled
   cache (`extract_hidden.py --teacher-labels`, one extra generation pass over the train prompts, ~90 s per
   100) answers the DSpine-faithful mimicry question ("can it reproduce the target, errors included") -
   less interesting while A's own accuracy is low, so it is not the main arm.
5. **Split hygiene - qualified, not claimed harmless.** The H/X/C holdout contrast is first-use for those
   variants, but this design was shaped with knowledge of the target, its measured difficulty and the A->B
   teacher results: human decisions about this target are already informed by those numbers, so the shared
   holdout cannot support a claim of fully blind confirmatory evaluation. Comparability with A still
   requires the same holdout, so both numbers are reported with this caveat attached. Any tuning choice
   uses the selection set only. If the design changes in ANY way after results are seen (retraining, width,
   layers, arms, thresholds), the fresh-seed confirmation set (`make_confirm_set`, disjoint from every
   existing mul2x2 set, checked by test) becomes required, not optional, for the final number. The
   selection set's prior use by the A->B stage-3 choices does not contaminate this probe, whose choices are
   already fixed in code.

Measurements (answers to the proposal's three questions):

1. *Can a small component learn the capability from internal representations?* Exact-match accuracy of arm H
   on the holdout (100 problems, used once), vs arms X, C and A. Interesting only if H beats the better of
   X/C by >= 15 points (the same distinguishable-gain logic as the teacher gate; a 100-problem holdout
   cannot resolve less). If C alone reaches high accuracy, the honest conclusion is that the task is simply
   learnable from 2,000 examples and hidden states add nothing - that is a result too.
2. *Does it generalize?* `range_data.py` builds a seeded 3-digit x 3-digit set (disjoint by construction).
   Also: `--zero-hidden` evaluation of arm H (states zeroed, skip kept) quantifies how much of the accuracy
   is the hidden states rather than the skip connection plus frozen head. Selection set is used for any
   choice; holdout stays final; the fresh-seed confirm set backs up any post-holdout decision.
3. *Weight-space conversion?* Not attempted here. If arm H shows a real gain, a follow-up could distill the
   aux's function into a rank-constrained edit of an existing matrix and measure recovery - future work,
   not promised by this probe and not implied by anything in DSpine.

Also recorded per arm: parameter count, peak memory, seconds per prompt, and retention after bf16
save/reload. Controls: add2 control set; the 30-prompt text control stays with the main experiment.

## 4. Feasibility and limits

- No GPU, no CUDA, no SGLang here: DSpine itself cannot run on this box (its MLX path needs Apple Silicon
  with ~21 GB; the GGUF port needs a GPU and the 27B target). Direct replication is out of scope.
- The probe reuses the CPU-measured patterns of the teacher run: extraction ~2 min for 800 examples,
  aux training ~5 min per arm for 150 steps at 128 positions/step (chunked-head CE), each 100-prompt greedy
  evaluation ~90 s. Roughly 30-40 min for all three arms plus evaluations, sequential, on the same class of
  2-CPU / ~1.7 GB box. Abort on memory errors; do not explore wider scopes by repeated OOM.
- Execution needs the machine that holds the model (the teacher experiment's box); the review box that
  wrote this design has no torch and ~1 GB RAM. Coordinate timing so the probe does not compete with the
  running A->B experiment.
- The torch code paths are untested; first step on the target box is a smoke run (small --n, few steps)
  before any measured run. The data-set helpers are unit-checked.

Not covered by this probe (explicit): block-parallel prediction and gated adjacent injection, DSpine's
headline mechanism (a single-position probe does not exercise it; a follow-up block-drafting probe is only
worthwhile if H shows signal); a teacher-labeled main run (available via flag, secondary); any weight-space
conversion (question 3, the skill_delta line's business).

## 5. Files

- `extract_hidden.py` - cache target hidden states / final hidden / prefix embedding mean / prefix token ids
  at answer positions; optional teacher labels
- `train_aux.py` - train arm H (conditioned), X (parameter-matched ablation) or C (conventional GRU)
- `eval_aux.py` - greedy exact-match evaluation on holdout / selection / control / range / confirm, in
  dependent, standalone, and zero-hidden modes
- `range_data.py` - seeded 3-digit x 3-digit generalization set and fresh-seed 2-digit confirmation set
  (both disjointness-checked by test)
