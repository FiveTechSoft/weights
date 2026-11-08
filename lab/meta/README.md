# lab/meta: measuring "learning to learn" for WSP experiments

Question: does experience from earlier experiments let a proposer choose better interventions
on tasks it has not seen? This directory measures that instead of asserting it.

- `grid.py`: real runs. For each task (a set of key-value facts) and each configuration
  (target layer, eta, regularisation) it imprints the facts into a NanoWSP-3 and measures recall.
  36 configurations x 21 tasks. The result is cached in `grid_scores.json` (reproducible: seeds fixed).
- `meta_learner.py`: the proposer (static ranking, nearest-task ranking) and the controls
  (shuffled experience, exact random baseline). Pure logic, unit tested in `tests/test_meta_learner.py`.
- `evaluate.py`: criteria fixed before looking at held-out results. Held-out tasks use seeds and fact
  counts (20, 60, 200) absent from the experience (10, 30, 100, 300).

## Result (2026-11-08, first run)

Mean trials to reach recall >= 0.95 on 9 held-out tasks (lower is better):

| policy | trials |
| :--- | :--- |
| random order | 2.05 |
| paper default config first | 1.22 |
| static ranking from experience | 1.00 |
| nearest-task ranking from experience | 1.00 |
| shuffled-experience control | 1.00 |
| oracle | 1.00 |

Online stream (experience grows task by task): first third 1.41 trials vs 2.12 random, last third
1.00 vs 2.10.

Reading, as pre-registered: experience does reduce the search cost compared with no experience
(criterion a holds). The claim "the proposer learned to learn" is **not supported**: the shuffled
control and a context-free static ranking do just as well (criteria b and c fail). On this testbed
one configuration (eta 3.5 or 8, small regularisation) is near-optimal for every task, so there is
nothing for task context to add. A harder testbed, where the best intervention depends on the task,
is needed before a contextual proposer can show an advantage. Not tested: any model other than
NanoWSP-3, any intervention other than the closed-form imprint.

Reproduce: `python lab/meta/grid.py` (about 7 minutes on CPU, resumes from the cache) then
`python lab/meta/evaluate.py`.

## Second testbed: interference (`interference.py`, 2026-11-08)

The first testbed was too easy. Here a NanoWSP-3 already stores `n_old` facts (at `old_layer`) and a
new batch is imprinted with a chosen (layer, eta, regularisation). Score = min(recall of new facts,
retention of old facts), so writing over the old memory is penalised. 30 configurations x 15 tasks
(9 train, 6 held out with new seeds and sizes), real runs again. Threshold 0.8 and budget 10 were fixed
on development runs, before any held-out task was scored. Same criteria a, b, c as above.

Held-out results (`python lab/meta/evaluate.py interference`):

| policy | mean trials (censored at 11) | score of first proposal |
| :--- | :--- | :--- |
| random | 9.60 | 0.159 |
| paper default first | 8.67 | n/a |
| static ranking | 6.50 | 0.357 |
| nearest-task ranking | 6.00 | 0.733 |
| shuffled-experience control | 7.87 | 0.303 |
| oracle | 6.00 | 0.733 |

The three pre-registered criteria hold. The nearest-task proposer proposed the oracle-best configuration
first on 6 of 6 held-out tasks; the static ranking did on 2 of 6. What it picked up from the experience:
best results come from writing the new facts at the same layer as the old memory, with eta 1.0 and a
small regularisation. That is a task-dependent rule (it follows `old_layer`), not a global default.

Limits, stated plainly:
- Only 3 of the 6 held-out tasks have any configuration reaching 0.8, so the trials-to-success numbers
  rest on 3 solvable tasks (the other 3 are censored for every policy, oracle included). The
  threshold-free "score of the first proposal" column is the cleaner evidence, and it points the same way.
- 6 held-out tasks, wide confidence intervals (see `evaluate.py` output); held-out old layers are the ones
  seen in training, only the sizes and seeds are new. Unseen layers or other architectures are untested.
- Still one model (NanoWSP-3) and one intervention family (closed-form imprint). This shows that a
  proposer can learn a task-dependent rule from past experiments there; it does not show it for LLM-scale
  weight edits.
