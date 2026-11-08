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

## Protocol v2: unseen old layers (`interference_v2.py`, 2026-11-08)

New protocol, not a re-run of v1: grid over layers 1 to 5 (50 configurations), budget 15 (30% of the grid),
10 held-out tasks including 4 with old layers 2 and 4 that never appear in the experience. The extra
criterion d (threshold-free score of the first proposal) and the reporting rules were committed in
`ca08974` before any v2 held-out task was scored. Full output: `interference_v2_results.txt`.

Result: mixed, and negative where it matters.
- Nominal criteria a, b, c hold on all 10 held-out tasks (contextual 11.5 trials vs random 13.9, static 12.3,
  shuffled 12.9), but the confidence intervals overlap almost completely ([7.0, 16.0] for contextual) and
  5 of the 10 tasks are unsolvable for every policy.
- Criterion d holds on all held-out tasks (first-proposal score 0.454 vs static 0.312, shuffled 0.233), but
  fails on the unseen-layer tasks: **0.035 for contextual, below static (0.268), shuffled (0.106) and random
  (0.156)**. The oracle reaches 0.787 there.
- Reading: the nearest-task proposer does not generalise to old layers it has not seen. It averages the
  scores of absolute configurations of neighbouring tasks, so it cannot express the rule it should learn
  ("write at the old layer"), which is relative to the task. The v1 result (6/6 first proposals) was
  therefore interpolation between old layers seen in training, not an abstract rule.
- Next, as a new protocol (v3) with fresh held-out tasks: encode a configuration relative to the task
  (layer minus old layer) so the experience can transfer across layers. That idea comes from reading the v2
  failure; it must not be scored on the v2 held-out tasks.

## Protocol v3: task-relative configuration encoding (fresh held-out tasks)

Pre-registered in 5a920d7 (before any v3 held-out task was scored). Results: `interference_v3_results.txt`
(`python lab/meta/evaluate_v3.py`). Held-out tasks use new seeds and sizes; v2 held-out tasks are not reused.

| policy (first-proposal score, 10 held-out / 4 unseen layers) | all | unseen layers |
|---|---|---|
| random | 0.184 | 0.160 |
| static (absolute) | 0.360 | 0.293 |
| absolute contextual (v2 proposer) | 0.544 | 0.175 |
| **relative contextual** | **0.731** | **0.643** |
| relative, shuffled experience | 0.351 | 0.322 |
| relative static (pooled, no task similarity) | 0.731 | 0.643 |
| fixed rule (layer = old layer, ties in grid order, no experience) | 0.115 | 0.079 |
| oracle | 0.750 | 0.690 |

Criteria d, e, f hold nominally. Honest reading:
- The relative encoding and the offset-0 idea are OUR design decision, not discovered by the system. It came from reading the v2 failure.
- Within that representation, experience supplies the rest (which eta/reg work at offset 0): the experience-free fixed rule scores 0.115 on its first proposal.
- Task similarity adds nothing: relative static equals relative contextual. Learning here is pooled experience in a hand-designed representation, not task-conditioned meta-learning.
- Trials-to-success is capped by the 5/10 held-out tasks that no policy (oracle included) solves; solvable-only: relative contextual 1.00, fixed rule 3.00, static 5.80, absolute contextual 4.40, random 10.30.
- The fixed-rule baseline is weak in its eta/reg tie-break (grid order); a stronger hand rule was not pre-registered and was not added post hoc.
- Next honest step: let the system discover the encoding itself (e.g. choose among candidate encodings by leave-one-task-out), with a new held-out set.

## Protocol v4: automatic choice among designed encodings (fresh held-out, negative on the key criterion)

Pre-registered in 4c66efe before scoring; results in `interference_v4_results.txt` (`python lab/meta/evaluate_v4.py`).
Leave-one-old-layer-out validation picked `rel_old` (0.750 vs absolute 0.119 vs distractor rel_size 0.152), the
selection is sound. On 10 new held-out tasks (4 with unseen layers) the selected policy scores 0.724 first-proposal
(unseen: 0.693), versus 0.468 / 0.053 for the absolute contextual proposer and 0.302 for static, but it is identical to
control A (best designed encoding, learned static ranking) and control B, by construction. Criterion g (additional
utility over control A) is FALSE. Reading: selection among candidates we wrote recovers our own best design; it shows
that validation can reject bad encodings, not that the system invents representations. 5/10 held-out tasks are
unsolvable for every policy (oracle included). Open: candidate encodings that are generated, not listed, and a benchmark
where the best designed encoding differs between task families, so that selection has something to add.

## Protocol v5: two task families (fresh held-out, negative on criterion g)

Pre-registered in add366e before any held-out task was scored; results in `interference_v5_results.txt`
(`python lab/meta/evaluate_v5.py`). 20 new held-out tasks (10 per family, 8 with unseen layers, 7/20 unsolvable by
every policy). The selector picked `rel_old` in both families, and `rel_old` was also the best fixed encoding in
hindsight in both (first-proposal 0.876 overall, 0.905 on unseen layers; absolute 0.635 / 0.301; static 0.449;
shuffled-experience selection 0.386). Selected = best fixed control, so criterion g (additional utility) is FALSE;
h and i hold (the selector matched the hindsight best in each family). The pre-registered expectation that family B
would prefer a non-relative encoding (from the training tasks) did not hold on held-out tasks: the two families do
not need different encodings here, so this benchmark cannot show a benefit of selection. Open: a family where a
designed encoding is actually wrong (needs a mechanism-level difference, not a scoring change), or generated rather
than listed candidates.

## Balance v1 to v5 (what was learned, what was designed)

| protocol | held-out | headline | verdict |
|---|---|---|---|
| recall testbed | 1 family | all policies reach 1.00 trials | not testable: one config fits every task |
| v1 interference | 6 tasks, seen layers | contextual first pick best 6/6 | interpolation between seen layers; metric added post hoc |
| v2 (budget 15, unseen layers) | 10 | first-proposal 0.035 on unseen layers vs static 0.268 | NEGATIVE: absolute encoding does not transfer |
| v3 relative encoding | 10 fresh | 0.731 (unseen 0.643) | works, but encoding is our design; task similarity adds nothing |
| v4 selection among designed encodings | 10 fresh | selected = best designed control | no additional utility |
| v5 two families | 20 fresh | selected = best fixed in both families | no additional utility; v5 did NOT implement candidate generation, only the two families |

Post hoc analysis (all cached scores, families A and B; not a test): mean score by offset (layer minus old layer)
is flat except a small bump at offset 0, but at offset 0 the eta knob dominates: eta 1.0 gives 0.76 (A) / 0.87 (B),
every other eta at most 0.25. So what experience mainly transmits is "use eta about 1.0 and write at the old layer",
a nearly constant setting in a 50-config grid. Learned by the system: which eta works (from the score tables) and
that a relative layer encoding beats an absolute one. Supplied by us: the config space, the relative encoding, the
candidate encodings, the families, the threshold and the scoring.

Best-founded next step (not started): within-task adaptation. Instead of ranking configurations once from past tasks,
use the scores of the first trials on the NEW task (sequential search with a meta-learned prior from past tasks, e.g.
a kernel over eta on a finer grid) and compare it against the same search with a flat prior, which is the honest
control for "experience helps". It needs no new encodings and no new families. Validation cost locally: a finer eta grid
(about 15 values) costs ~5 s per config-task pair, ~20 fresh held-out tasks about 25 to 30 minutes of compute, plus the
pre-registration commit before scoring. A failure mode to report in advance: the knob is one-dimensional, so a
flat-prior search may already find eta 1.0 in few trials.
