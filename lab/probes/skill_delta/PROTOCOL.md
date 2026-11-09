# WSPLang 0.1: reproduce a trained capability by weight surgery (2-digit multiplication)

Status 2026-11-09 07:10: protocol and trainer only. NOTHING of this experiment has been evaluated. Written before any evaluation;
thresholds below are fixed and are not changed after seeing any holdout number. The skill (2-digit x 2-digit multiplication) was
named by the project owner; it was not selected on results. The illustrative pairs in the owner's proposal (17 x 23, 19 x 27, ...)
are known examples, not a secret holdout; the holdout is the seeded set below.

## Model and stages
Model A: Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17 (Apache-2.0). Model B: A after a small conventional training.
1. Teacher: B must beat A on new problems. If not, the experiment stops here (no surgery).
2. Delta: apply delta = B - A to A. Must recover B (a sanity check of the code, not a result).
3. Filter / subspace: smaller edits (top-magnitude entries, low-rank per matrix) must reproduce the gain.
4. Another task: not claimed by stages 1-3.

## Data (lab/probes/closed_form_math/data.py, seeded; operand pairs disjoint across sets, a*b and b*a count as the same problem)
train 2000 (use first N), selection 100 (stage 3 choices only), holdout 100 (final numbers, used once per model variant),
control 100 two-digit additions (seed 6). The 3-prompt smoke pairs are excluded everywhere. Text/fact control: 30 fixed short
prompts with a gold substring, controls_text.json (sha256 prefix 0ae7c2e133fc4424, committed in c7759ce). Disclosure: the unpatched
A text control (22/30) was run right after the file was written and before it was committed; the file was not changed afterwards.

## Metrics
Free greedy generation (<= 6 new tokens), prompt "Q: {a} * {b} = ?\nA:", exact match of the text before the first newline. Never
NLL or target probability as a hit. Paired comparison with exact McNemar on the holdout.

## Gates (fixed now)
Origin of each threshold: the owner's proposal asked for (i) B better than A on the evaluation set, (ii) >= 90% of that improvement recovered,
(iii) controls not worse by more than 2 points, (iv) results kept after saving and reloading. The ">= 15 points, McNemar p < 0.05" size of (i)
is the engineer's addition (a gain that a 100-problem holdout can distinguish from noise), not the owner's number.
- Stage 1: B - A >= 15 points on the holdout, McNemar p < 0.05, and control (2-digit addition) not worse than A by more than 2 points.
- Stage 2: the A+delta model gives the same greedy output as B on >= 99% of holdout + control prompts, and tensors equal B up to the stored dtype.
- Stage 3: a reduced edit recovers >= 90% of (B - A) on the holdout (point estimate; the confidence interval is reported), with control
  loss <= 2 points. Reduced edits are chosen on the selection set only.
- Persistence: every number above is re-measured after saving the edited tensors in the SAME dtype and loading them in a fresh process.
  In-memory results do not count. (Reason: the external qwen0.8B-wsp repo reports that a check on bf16-merged saved weights fell from 2/5 to 0/5.)
Passing says something about this model, skill and teacher only.

## Scope of the teacher (clarified)
The teacher trains ONE matrix: layers.23.mlp.down_proj.weight (1024 x 3584, 3.67M parameters). It is not the whole of layer 23: that
configuration (and layers 20-23) ran out of memory at batch 4-8 and is not used. Because layer 23 is the last decoder layer and its MLP,
the final norm and the head are position-wise, caching the residual stream entering the MLP and the MLP activation at the answer
positions is EXACT for this matrix (train_down23.py header). Training is real CE through the final RMSNorm and the full 248k-way head.
It would not be exact for any earlier matrix, so the cache path is limited to this one.

## CPU feasibility (measured 2026-11-09, box with 2 CPU and ~1.7 GB free RAM)
- Real greedy generation works: 100 prompts in about 90 s.
- Smoke test of train_down23.py (40 examples, 237 answer positions, 30 steps of 128 positions): caching 5 s; peak anonymous memory
  1.54 GB; ~1.9 s/step (the 0.5 GB head is re-read in chunks every step); training loss 0.71 -> 0.17-0.21 on those 237 positions.
  This is a training-set loss on a smoke run, not a result. Two earlier runs of the same script were killed for memory before it fitted
  (the kernel kill shows as "Killed"); repeated OOM exploration of wider scopes is not done.
- Planned run (not executed): cache 800 examples (~100 s), 150 steps (~5 min), then evaluate A and B on holdout (100 prompts, ~90 s each),
  control additions (~90 s each) and text control (~50 s each). About 15 minutes in total.
- No GPU is used. Whether the owner's RTX 3060 is reachable is unverified. No GitHub Actions workflow has been added.
