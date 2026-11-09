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
prompts with a gold substring, to be written and hashed before the first evaluation (open item).

## Metrics
Free greedy generation (<= 6 new tokens), prompt "Q: {a} * {b} = ?\nA:", exact match of the text before the first newline. Never
NLL or target probability as a hit. Paired comparison with exact McNemar on the holdout.

## Gates (fixed now)
- Stage 1: B - A >= 15 points on the holdout, McNemar p < 0.05, and control (2-digit addition) not worse than A by more than 2 points.
- Stage 2: the A+delta model gives the same greedy output as B on >= 99% of holdout + control prompts, and tensors equal B up to the stored dtype.
- Stage 3: a reduced edit recovers >= 90% of (B - A) on the holdout (point estimate; the confidence interval is reported), with control
  loss <= 2 points. Reduced edits are chosen on the selection set only.
- Persistence: every number above is re-measured after saving the edited tensors in the SAME dtype and loading them in a fresh process.
  In-memory results do not count. (Reason: the external qwen0.8B-wsp repo reports that a check on bf16-merged saved weights fell from 2/5 to 0/5.)
Passing says something about this model, skill and teacher only.

## CPU feasibility (measured 2026-11-09, box with 2 CPU and ~1.7 GB free RAM)
- Real greedy generation works: 100 prompts in about 90 s.
- Training works only for a very small scope. With the frozen base in bf16 (996 MB) plus the process, ~1.7 GB is all there is:
  layer 23 down_proj, batch 2: ~0.3 s/step, loss decreases from ~1.03; layers 20-23 or the whole of layer 23 at batch 4-8: killed by the
  kernel (out of memory), and backward through the linear-attention layers (reference PyTorch kernel) is slow. No teacher was trained;
  the loss values above are a smoke test, not a result.
- Path to a teacher on this CPU (not done): precompute and store the layer-23 inputs of the train/selection/holdout prompts once, free
  layers 0-22, and train layer 23 alone (about 18M parameters) with the head recomputed in chunks. Capacity is small, so the stage 1
  gate may well fail; that outcome is informative and would stop the experiment.
- Alternative: run `lab/probes/skill_delta/train_teacher.py` with wider scope on a machine with a GPU. Whether the owner's RTX 3060 is
  reachable is unverified and no access has been used. No GitHub Actions workflow has been added.
