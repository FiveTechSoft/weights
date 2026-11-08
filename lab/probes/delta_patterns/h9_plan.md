# H9: warm start — does starting from other tasks' learned changes shorten the training of a new one? (DRAFT)

Date: 2026-11-08. Follows H7 and H8 (results §41-42, findings F34-F37). Thresholds below are fixed before any H9
result; changes go in dated addenda.

## Why
"Learning to learn" by skipping training has failed three ways (F30, F32, F34), and F36 explains why: the learned
change only works as a whole. Reuse does work: a learned C is copied instantly, and two summed C keep ~0.89 of each
task's gain (F37).
H9 asks the practical version: if the jump cannot be skipped, can **reuse make it shorter**? Start a new task's
training from the sum of the other tasks' learned C instead of from zero, and count the steps it saves.

## Design
- Same model, basis, data and loop as H7 (`run_task.py`): Qwen3.5-0.8B, W = W0 + U·C, shared random rank-20 basis
  U, AdamW on C, lr 1e-5, 200 steps of batch 40, 10 % warmup then cosine, same seed and same batch order.
  **Only the starting point C_0 changes.** The optimizer state starts fresh, as in H7.
- Tasks: the 7 that passed H7's G2. Each is the held-out task once (leave one out).
- **Arms per held-out task h:**
  - **Cold:** C_0 = 0. This is H7's existing run, reused as is (same seed and data order); nothing is re-run.
  - **W_sum:** C_0 = λ·Σ_{j≠h} C_j(200), the six other tasks' final C, λ = 0.5. λ is fixed now: it is the value B2
    chose on dev in H8, so no selection is done on h.
  - **W_rand (control):** C_0 = a random tensor with the same global norm as W_sum's C_0 (seed fixed). It separates
    "a useful starting point" from "any non-zero start of that size".
- Logged as in H7: dev every 25 steps; test at 0, 25, 50, 100 and 200; retention NLL at 0 and 200; C snapshots at
  25, 50, 100 and 200 only (disk).

## Pre-registered criteria
Accuracy gain over the task's M0: g(arm, t) = acc_test(arm, t) − acc_test(M0). The comparison is at fixed step
counts, on paired test items (exact McNemar).
- **HW1 (warm start helps early):** at step 50, W_sum beats Cold by ≥ 5 points on average over the 7 tasks, **and**
  is better with p < 0.01 in ≥ 4 of 7.
- **HW2 (it saves steps):** W_sum at step 50 reaches at least Cold's step-100 test accuracy in ≥ 4 of 7 tasks
  (≥ 2× fewer steps).
- **HW3 (no final cost):** at step 200, W_sum is not worse than Cold by more than 3 points in more than 1 of 7 tasks.
- **HW4 (content, not size):** at step 50, W_sum beats W_rand on average by ≥ 5 points. If W_rand helps as much,
  the effect is a scale or regularisation effect, not reuse.
- **H9 passes** if HW1, HW2 and HW4 hold, and HW3 is not violated.
- **Descriptive:**
  - step-0 accuracy of W_sum on h: the interference of the other six, which H8 measured only with h included;
  - dev curves every 25 steps;
  - retention NLL;
  - per task, whether a related task in the sum (e.g. mul3x3 for mul2x2x1) goes with a larger gain.

## Cost and safety
- 14 new runs (7 W_sum + 7 W_rand). In H7, one run took ~27 min including evaluations, so this is about 6.5 h of GPU.
- One job at a time, chained under nohup, from a frozen copy (`lab/probes/h9_run/`).
- Disk: four C snapshots per run (~25 MB each in fp16, ~100 MB per run, ~1.4 GB in total).
- No original model file is modified.
- Runs **after** the Jackrong vs MiMo comparison that is now using the GPU.

## Code (tests first)
`lab/probes/h9/run_warm.py`: H7's trainer with `--init sum|rand` and `--lam`. Pieces with tests:
- `warm_init(task, held, lam)`: per-matrix sum of the other tasks' C_final, times λ. Tests:
  - it excludes h;
  - it equals the manual sum;
  - it is deterministic.
- `rand_like_norm(C0, seed)`: random, same shapes, same global norm, deterministic.
- `snapshot_steps` override (25, 50, 100, 200).
- The first logged step must reproduce H7's Cold run when `--init zero` is given. A check on mul3x3 for 10 steps
  must give a bit-identical C_10 to H7's G3 run: this is G0.

## Not in H9
- No new tasks.
- No choice of λ on test data.
- No learned optimizer: that is option 2 for a later plan.
