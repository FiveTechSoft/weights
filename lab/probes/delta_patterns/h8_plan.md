# H8: where the accuracy lives in C, and whether learned changes add up (APPROVED by the user 2026-11-08)

Date: 2026-11-08. Follows H7 (results §41, findings F44-F45). Uses only H7's saved coefficients
(`hf_cache/h7/<task>/`); nothing is trained. Thresholds below are fixed before any H8 result; changes go in dated
addenda.

## Why
H7 found:
- the coefficient trajectory is nearly straight and C_25 already points the right way (cos(C_25, C_200) 0.66-0.92);
- yet no predictor stretched from step k beats stopping at k (F44);
- different tasks use nearly orthogonal C in the shared basis (F45).

Two questions follow:
- **A.** Is the accuracy in the part of C_200 along C_k (which a predictor could reach by scaling) or in the part off
  it (which only the late steps add)? This explains the H7 failure.
- **B.** If tasks are nearly orthogonal, can already-learned changes be **combined without training**? This is
  "applying changes synthetically" by the route that works: reuse, not prediction.

## Common setup
- Qwen3.5-0.8B, W = W0 + U·C with H7's shared basis U (`run_task.shared_basis`).
- Tasks: the 7 that passed H7's G2 (mul3x3, mul2x2x1, muladd1, div6_2, mod3_1, add3x3, rev4).
- Evaluation on each task's 500-item test set, the same as H7. acc0 = M0 test accuracy; accT = acc(C_200).
  R = (acc − acc0)/(accT − acc0).
- Exact two-sided McNemar test on paired items where two arms are compared.

## Part A: decompose C_200 along C_k
For k ∈ {10, 25, 50} and each task:
- d = C_k / |C_k| (flattened over all matrices);
- a = ⟨C_200, d⟩; parallel part P = a·d; off-line part O = C_200 − P.

Arms:
- **A_par:** W0 + U·P. This is the best any rescaling of C_k can do, because the scale a is the oracle one.
- **A_off:** W0 + U·O.
- Logged for each arm: energy share |O|²/|C_200|².

**Pre-registered reading (k = 25, mean over the 7 tasks):**
- **HA "the accuracy lives off the line":** mean R(A_par) < 0.5 **and** mean R(A_off) ≥ 0.5.
- **HA' "the line is enough":** mean R(A_par) ≥ 0.8. Then H7's failure is a scale problem, not a direction problem.
- Otherwise the result is mixed: both parts are needed, and neither is enough alone.

k = 10 and k = 50 are descriptive.

## Part B: a library of learned changes
**B1, pairs:**
- For each of the 21 pairs (i, j): W0 + U·(C_i + C_j), evaluated on the test sets of both i and j (42 evaluations).
- Retention r_i = (acc_i(pair) − acc0_i)/(acc_i(C_i) − acc0_i).
- Reference for each pair, computed from the H7 interference matrix with no new run: the additive prediction
  acc0_i + Δ_ii + Δ_ji.

**B2, all 7:**
- W0 + U·λ·ΣC, with one global λ ∈ {0.5, 0.75, 1} chosen on the sum of dev accuracies over the 7 tasks.
- Then the test set of every task, plus retention NLL on GSM8K (the same function as H6/H7).

**Pre-registered criteria:**
- **HB1 (pairs add up):** median r over the 42 (task, pair) cells ≥ 0.8.
- **HB2 (the library holds 7 skills):** mean r over the 7 tasks ≥ 0.7 at the chosen λ, **and** retention NLL ≤
  base + 0.02 (base 1.321).
- **Descriptive:**
  - worst cell;
  - whether pairs with higher cos(C_i, C_j) lose more;
  - whether the additive prediction from the interference matrix is close (mean absolute error in points).

## Not in H8
- No training, no new tasks.
- No comparison with joint multi-task training. That would need new runs, and goes in a later plan if B passes.

## Cost and safety
- About 120 evaluations of 200-500 items each, at about 8 s per evaluation in H7, so about 20-30 min of GPU.
- One job, under nohup, run from a frozen copy (`lab/probes/h8_run/`).
- Writes `lab/runs/h8/decompose.json` and `lab/runs/h8/library.json` after every evaluation.
- Reads H7 files only and modifies no model file.

## Code (tests first)
`lab/probes/h8/h8.py`:
- `decompose(C_k, C_T) -> (P, O, a, share)`, with tests: P + O = C_T, ⟨O, d⟩ = 0, a planted scale recovered;
- `retention(acc, acc0, acc_own)`, with tests;
- `add_coefs([C...], lam)`, with a test of the per-matrix sum and scale;
- `pick_lambda(dev_sums)`: largest dev sum wins, and a tie goes to the smaller λ (as in `synth.pick_alpha`);
- main `--part a|b|all`.

## Addendum a (2026-11-08, after B2 failed; before any TIES result): B3, TIES merge of the 7 task C
User asked for a TIES arm (the Delta League leader on Qwen3.5-0.8B uses TIES).
- TIES (Yadav et al. 2023) on the 7 flattened task vectors C_i (global over all matrices):
  1. trim: keep in each C_i its top-d fraction of entries by |value|, zero the rest;
  2. elect: per coordinate, sign of the sum of the trimmed values;
  3. disjoint mean: per coordinate, mean of the trimmed values whose sign agrees with the elected one (0 if none);
  4. W = W0 + U·(λ·merged).
- Grid chosen on the sum of dev accuracies over the 7 tasks: d ∈ {0.2, 0.5, 1.0}, λ ∈ {1, 2, 3, 5}. Ties go to the
  smaller d, then the smaller λ. Disjoint mean averages, so λ up to the number of tasks is in range.
- Then: test of every task, retention NLL.
- **HB3:** mean r ≥ 0.7 **and** retention NLL ≤ base + 0.02 (same as HB2).
- Descriptive: per-task r against B2. Paired McNemar per task against B2's test masks is not available (B2 kept
  only accuracies), so the comparison is by r only.
