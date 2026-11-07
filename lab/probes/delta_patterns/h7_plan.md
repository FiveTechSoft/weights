# H7: a dataset of training steps, to learn to learn (APPROVED by the user 2026-11-08)

Date: 2026-11-07. Follows H6 (results §40, findings F29-F33). Thresholds below are fixed before any H7 result;
changes go in dated addenda.

## Why (user, 2026-11-07: "el dataset de pasos, para aprender a aprender")
H6 found:
- where a small training moves: a narrow lane (F29, F31);
- that the "how much" per direction cannot be guessed:
  - from the first frames (F30);
  - from one gradient (F32, H6f-a);
- that training only 20 directions per matrix learns most of the skill, even in a random basis (F33).

**Learning to learn needs several tasks.** For one task there is nothing to predict: a second run of the same task
can simply reuse the first run's weights. The question is whether the trajectories of past tasks teach a rule that
lets a **new** task skip most of its steps.

## Design
**Common coordinates.**
- Per matrix, one fixed random orthonormal basis U of rank 20 on the smaller side (seed fixed), shared by all tasks.
  ΔW = U·C, so every task's state is a vector C of 10.4 M numbers in the same coordinates.
- F33 shows that a random rank-20 basis learns nearly as well as the lane. A task-agnostic basis keeps the held-out
  task's own training out of the coordinates.

**Tasks** (3 solved examples in the prompt, answer then newline, exact-match grading). M0 3-shot accuracy measured
on 100 items (exploratory precheck, 2026-11-07); kept if between 5 and 50 %:

| Task | Prompt | M0 | Family |
|---|---|---|---|
| mul3x3 | `a*b=` (3×3 digits, H6) | 29 % | multiply |
| mul4x3 | `a*b=` (4×3) | 10 % | multiply |
| sq4 | `a^2=` (4 digits) | 17 % | multiply |
| mul2x2x1 | `a*b*c=` (2, 2, 1 digits) | 28 % | multiply |
| muladd1 | `a*b+c=` (2, 1, 2 digits) | 21 % | mixed |
| div6_2 | `a//b=` (6 by 2 digits) | 24 % | divide |
| mod3_1 | `a%b=` (3 digits, b in 3-9) | 16 % | divide |
| add3x3 | `a+b+c=` (three 5-digit numbers) | 11 % | add |
| rev4 | `reverse a=` (4 digits) | 48 % | string |

Too easy (rejected): add7 93 %, sub6 97 %, sq3 94 %, cube2 94 %, div4_1 99 %, mul4x2 68 %.
Too hard (rejected): mod4_2 1 %, muladd (2,2,3) 0 %, sub_neg 0 %, rev8 0 %, digsum9 3 %, digsum4 5 %.

**Per task, one run:**
- Data: train 20 000, dev 200, test 500, disjoint, deterministic.
- Optimizer: AdamW on C only; lr 1e-5 (H6f probe); 200 steps of batch 40; 10 % warmup, then cosine.
- Same loop and numerics as H6f-b.

**Logged per run (the dataset).** About 1 GB per task, about 9.5 GB in total; 37 GB free.
- C: every step for steps 1-10, then every 5 steps (fp16).
- AdamW moments m, v at steps 10, 25 and 50 (fp16).
- Per step: loss, lr, and per matrix |g_C|, |u|, cos(g, u) and cos(u_t, u_{t-1}).
- Dev accuracy every 25 steps; test at steps 0, 25, 50, 100 and 200.
- Batch indices, so any step can be replayed.

## Gates
- **G1:** M0 on the 500-item test set is between 5 and 50 %. Otherwise the task is dropped before training.
- **G2:** acc(C_200) − acc(M0) ≥ 10 points on the task's test set. Otherwise the task is dropped from the held-out
  evaluation (nothing to recover) but its trajectory is kept as training data.
- **G3:** a re-run of mul3x3 for 10 steps gives a bit-identical C_10.

## Questions answered (descriptive, no threshold)
1. **Shape in time.** For each task:
   - cos(C_t, C_200) against t;
   - the share of trajectory variance in its top 1-3 temporal principal components;
   - whether coordinates follow a common profile, C_t ≈ a(t) · C_200.
2. **Across tasks.**
   - Cosine between tasks' C_200, overall and per matrix kind.
   - Does the multiply family share coordinates, and is rev4 apart?
   - Which layers and kinds move per task.
3. **Interference matrix.** Every task's C_200 evaluated on every other task's test set (9 × 9).

## H7, learned jump (pass/fail)
Leave one task out: the predictors are fitted on the trajectories of the 8 other tasks and applied to the held-out
task after k of its own steps, k ∈ {10, 25, 50}.

**Predictors:**
- **P_scale:** C_pred = α · C_k, where α is the median over the training tasks of ⟨C_k, C_200⟩ / |C_k|². This is the
  E_k analogue in coefficient space.
- **P_lin:** a per-coordinate linear rule shared by all coordinates:
  - C_200,i ≈ w · φ_i, with φ_i = (C_k,i, C_k/2,i, m_k,i, m_k,i / (√v_k,i + ε), 1);
  - w fitted by least squares on the training tasks (one w per matrix kind).
  - It is a tiny learned optimizer that jumps from step k to step 200.
- **P_mlp:** the same features, a 2-layer MLP (hidden 32) shared by all coordinates, trained on the training tasks.
- **"+dev" variants:** one extra scale γ ∈ {0.5, 0.75, 1, 1.5, 2} on the prediction, chosen on the held-out task's
  dev set (forward passes only).

**References:**
- B_k: stop at step k (C_k as it is);
- B_200: the real run;
- R_traj = (acc(pred) − acc(M0)) / (acc(C_200) − acc(M0)) on the held-out task's test set.

**Pass:**
- at k = 25 (1/8 of the steps), the best of P_lin / P_mlp (chosen once, on the mean over held-out tasks at k = 10)
  reaches mean R_traj ≥ 0.7 over the held-out tasks that pass G2;
- and it beats B_25 with McNemar p < 0.01 in at least 2/3 of those tasks.

**Also reported:**
- every predictor at every k;
- the "+dev" variants;
- retention NLL;
- the predicted against the real C_200: cosine and relative error.

**Prediction:**
- P_scale fails (R ≈ 0.1-0.3), as E_k did.
- P_lin reaches 0.4-0.7 at k = 25, so it fails.
- P_mlp is a little better.
- The multiply tasks predict each other better than rev4.
- Confidence: low.

## Code needed (tests first)
- `lab/probes/h7/tasks.py`: the task registry, the splits and the prompt.
- `lab/probes/h7/run_task.py`: coefficient training with the logging above, reusing `h6f.coef_*`.
- `lab/probes/h7/predict.py`: the predictors, leave-one-task-out, and evaluation.
- `lab/probes/h7/geometry.py`: the descriptive questions.

## Cost
- GPU, one job at a time.
- About 30 min per task: about 4.5 h for 9 tasks, plus the 9 × 9 interference evaluations (about 1 h).
- Predictor fitting is on CPU, minutes.
- Every predicted model evaluated: about 1 min per task × predictors × k.

## Addendum 2026-11-08 a (before any H7 code ran)
- **Train split = 8 000 per task** (not 20 000). 200 steps × batch 40 use exactly 8 000 examples, one pass with no
  repeats. Several tasks have small item spaces (rev4 and sq4: 9 000 distinct questions). Dev 200 and test 500 are
  unchanged.
- **mod3_1:** b ∈ {3, ..., 19} instead of {3, ..., 9}. The old space (6 300 questions) cannot hold 8 703 distinct
  items. The precheck's 16 % no longer applies; G1 on the test set decides.
- **Basis:** U_n = random_basis(d, 20, seed = 70 000 + index of the matrix in sorted order). Identical for every task.

## Addendum 2026-11-08 b (code details fixed before any H7 trajectory finished)
- C_k/2 uses the nearest stored snapshot: k = 10 → 5, k = 25 → 10, k = 50 → 25 (25 is k, so the "half" is 10 → 25
  would repeat; the mapping is fixed as listed).
- P_mlp training: 200 000 random coordinates per training task. Features 1-4 are divided by their std over those
  rows, and the target by its std. Adam lr 1e-3, 2 000 steps of batch 4 096, seed 0.
- B_k: for k = 25 and 50 the test evaluation logged during training (fp32 C). For k = 10, the fp16 snapshot C_10
  evaluated afterwards. Predictions are built from the fp16 snapshots and moments.
- The McNemar comparison for the pass rule counts only wins (predictor better than B_25 with p < 0.01).
