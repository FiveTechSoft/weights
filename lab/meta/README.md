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
