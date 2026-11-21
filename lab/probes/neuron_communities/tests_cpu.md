# What was verified on CPU before any GPU run (2026-11-18, night; fork agent, no commits)

## Files
- `communities.py`: PMI co-attribution graph, Louvain (`networkx 3.7`, seed 2026, cap 128 by weighted degree),
  always-on set, candidate community, `merge_topk` (global top-k across layers), the PLAN reading rule (`verdict`).
- `attribute.py`: `positions_for`, `attribute_sequence` / `group_backward` (layer groups, one vector-Jacobian
  product per position), `end_to_end_grads` (reference), `ablate_probabilities` (one-row route) /
  `full_forward_prob` (reference), condition builders, and the subcommands `run`, `communities`, `ablate`, `check`.
- `tests/test_neuron_communities.py`: 11 tests, all pass with the system python (`torch 2.15 cpu`, `transformers
  5.12.1`, `networkx 3.7` are all importable from the system python; the GPU venv is only needed for `run`/`ablate`
  on the 9B). `PYTHONUTF8=1 python -m unittest tests.test_neuron_communities` → `Ran 11 tests ... OK` (18 s).

## Two-method checks (PLAN "Two-method check first")
1. **Layered gradient route vs end-to-end autograd**, tiny random Qwen3.5 text model (4 layers, 3 linear + 1 full
   attention, float32): gradients at every group input and `a · ∂log p/∂a` at every `down_proj` input agree to
   relative error < 1e-3 for groups of 1, 2 and 4 (test), and are bit-identical in float32 on CPU (max abs diff 0.0,
   norms ≈ 8.5); a different target token changes the gradient by 129 % (control: the comparison is not trivial).
   Causality holds (no gradient after the position).
2. **The same check on the real 0.8B** (`attribute.py check --n 2 --max-tokens 256`, float32, 18 positions per
   trace, groups of 4 over 24 layers): max relative error of the group-input gradients **0.0** at layers 20, 16, 12,
   8, 4, 0, max relative error of the attributions **0.0**, for both traces (41-53 s each route).
3. **One-row ablation route vs full hooked forward** (tiny model): p(token) at t_e equal to 5 decimals for 5
   conditions including "zero 60 of 64 neurons in layer 3" and "zero all 64 in layer 2"; batching of conditions
   does not change the result.
4. **End-to-end dry run on the 0.8B** (CPU, 2 discovery traces cut to 200 tokens, groups of 6, keep 16, then
   `communities --min-co 3`, then `ablate --heldout pos --limit 2 --placebos 2`): every step runs and writes
   `attributions.{npz,json}`, `communities.json`, `summary.json` with the PLAN's numbers and a verdict (the numbers
   themselves are meaningless at 200 tokens: p(</think>) ≈ 1e-13 there). Output removed afterwards.

## Clarifications of the PLAN as implemented (not changes of criteria)
- **Data layout found:** `pos.jsonl` and `neg.jsonl` are the *same* 53 traces: pos = the whole finished reasoning
  with focus on its last 135-140 characters (next token `</think>`), neg = a prefix cut of the same reasoning
  (focus [0, cut)). So one forward per trace serves both: pos event = last token of the reasoning, neg event = last
  token of the neg cut (the model is mid-reasoning there), each with its 8 predecessors → 18 positions per trace.
- **[rev] Held-out set:** the 53 pos traces are exactly the 53 finished traces of `lab/runs/cc/traces.jsonl`, and
  9 of them fall in cc_weights' 28 held-out (seed 20261111). Using "the 28 held-out traces" as the test would
  test on discovery traces. Implemented: discovery = the 44 pos traces outside the held-out set; test = the 9
  finished held-out traces (their close event = last reasoning token). n = 9 for the A1/A2 medians, small; the
  summary records per-trace numbers. `--heldout pos` reproduces the same 9 from pos.jsonl (same sequences).
- **Ablation position:** neurons are zeroed at the close-event position only (the prediction being tested), not
  at every position; that is what makes the one-row route exact (causality). All positions would also change the
  context the prediction reads from; not done.
- **Attribution kept:** `same` = a(t) · ∂log p_t/∂a(t) (the PLAN's formula) drives the graph and the masses;
  `total` = Σ_{s ≤ t} a(s) · ∂log p_t/∂a(s) is stored alongside in the npz for reading, not used by the rule.
- **A2's random control** uses placebo set 0 (|C_l| random neurons per layer) plus the always-on set.
- **Always-on:** neurons in the global top-64 at ≥ 90 % of the 792 selected positions (44 × 18).
- **Candidate masses:** |attr| summed at the 44 pos close events minus at the 44 neg events (event positions only;
  the graph uses all 792 positions).
- Edge rule as planned: PMI > 0 and co-selected ≥ 5 (`--min-co`); cap 128 keeps the highest weighted degree.

## Cost estimate for the 9B run (from the real tokenisation)
- 44 discovery traces, 63,893 tokens (median 1,272, max 3,094). Group-boundary cache 0.52 GB per group, 4.2 GB for
  8 groups of 4; gradient store (18 positions × T × 4,096, bf16, one file per trace, overwritten per group) 9.4 GB;
  ≈ 14 GB on disk under `lab/runs/neuron_communities/cache`, deleted at the end unless `--keep-cache`.
- GPU: 4 layers (2.2 GB) + head and norm at the top group (2 GB) + the retained graph of 4 layers at T ≤ 3,094.
  If memory fails, `GROUP=2 bash chain.sh` halves the graph (and doubles the disk writes of gradients).
- Time: forward ≈ 5 min; backward = 8 groups × 44 traces × (1 recompute + 18 vJPs) ≈ 60-100 min; ablation on 9
  held-out traces (987-3,395 tokens) with 2 + 20 + 2 + |layers(C)| conditions in batches of 8 ≈ 10-20 min.
- Held-out finished traces: 9, tokens [1114, 1607, 1114, 1161, 987, 2351, 957, 3395, 992].

## Not done
No GPU run; no edit of PLAN.md (the [rev] above is for the coordinator to write in); no ablation at all positions;
`networkx` Louvain only (no Leiden; igraph/leidenalg not installed).
