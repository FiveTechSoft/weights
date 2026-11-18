# Is "emit `</think>` now" a neuron community? (Goodfire's unit on our 9B) — plan fixed 2026-11-18, before any code or run

**Why.** Goodfire (Shilov, Fel, Haklay, Bushnaq & Lewis, arXiv 2611.23112, 17 Nov 2026) finds, on Llama-3.1-8B over
10M prompts, thousands of *neuron communities*: a few MLP neurons across layers (often < 10, cap 128) that are
attributed together to one next-token event, **necessary** (ablating them changes the prediction) and **sufficient in
their layers** (ablating every other MLP neuron of those layers, except a few always-on neurons, keeps the
prediction). Our §43.23 asked a different question: how many `down_proj` columns carry the *direction* "conclude"
(mean residual difference), and found ~100 per layer (k₅₀ 119 vs 128 for a random split; layer 26: 4). A direction
can be spread while the *event* it promotes, emitting `</think>`, is carried by a sparse community. If it is, WSP-Lang
gets a smaller and readable edit target (`edit_neurons` on those columns, with the op's structural placebo) and a
second mechanism to compare with the rank-1 compile (§43.19 / §45.8). If it is not, the rank-1 term stays the right
object and Goodfire's unit does not apply to this behaviour. The driver asked on 18 Nov for the WSP-Lang application;
this is it. **Nothing here runs without the driver's OK** (GPU busy with queues 5-6 until ≈ 03:15 on 19 Nov).

## Data (exists; nothing new generated)

`lab/runs/cc/pos.jsonl` (53 traces that close with `</think>`) and `neg.jsonl` (53 that do not), the rows the conclude
vector was derived from (§43.17); the 28 held-out traces of §43.19 for the held-out test. Model: `hf_cache/qwen35-9b-hf`
bf16 (32 layers × 12 288 neurons), loaded as in `lab/probes/cc_neurons/decompose.py` (one layer on the GPU at a time).

## Phase A — attribution, graph, communities, three tests (GPU ≈ 2 h; CPU for the graph)

**Positions.** For each `pos` trace, the position whose next token is `</think>` (the *close event*), plus the 8
positions before it; for each `neg` trace, the position at the same relative depth (fraction of the trace) and its 8
predecessors. Target logit: `</think>` at every position (what Goodfire attributes is the model's own next-token
prediction; here the event of interest is fixed so pos and neg are comparable).

**Attribution (exact gradient, not the direct path).** attr_{N,j}(t) = a_{N,j}(t) · ∂ log p(`</think>` | t) / ∂ a_{N,j}(t),
a = `down_proj` input (SwiGLU unit, the same as Goodfire's and §43.23's). Computed layer by layer: forward with the
inputs of every layer cached to disk (≈ 60 k tokens × 4 096 × bf16 × 32 layers ≈ 16 GB in
`lab/runs/neuron_communities/cache/`, deleted at the end), then backward from layer 31 down with one layer on the GPU,
re-running each layer's forward with `requires_grad` on its input. **Two-method check first:** the logit gradient of
layer 31 by this route must equal the one-layer autograd gradient on the GPU for 2 traces (relative error ≤ 1e-2,
bf16); and Σ_j attr_{N,j} must correlate with the MLP's first-order contribution (sanity, not a criterion). If the
cache does not fit in RAM/disk, fall back to 24 + 24 traces and say so before running.

**Graph.** Per position, the top-64 neurons by |attr| (global rank across layers). Edge weight = PMI of co-selection
(count together / expected from each neuron's own selection rate), kept if PMI > 0 and co-selected ≥ 5 times.
Communities: `networkx.community.louvain_communities(seed=2026)`, cap 128 neurons; the **candidate community C** is
the one with the largest total |attr| at close-event positions of `pos` minus the same at `neg` positions. Always-on
set S₀: neurons in the top-64 at ≥ 90 % of all positions (Goodfire's "always-on").

**Tests (held-out: the 28 traces of §43.19, close-event positions only; p = p(`</think>`) at those positions).**
- **A1 necessity:** zero the `down_proj` input of C's neurons in their layers → median relative drop of p.
  Placebo: 20 random neuron sets of the same size per layer (seed 2026), median drop.
  **Pass if** drop(C) ≥ 25 % and ≥ 5× the placebo median, and placebo median ≤ 10 %.
- **A2 sufficiency:** in C's layers, zero every MLP neuron except C ∪ S₀ → p retained / p original.
  **Pass if** median retained ≥ 50 %, and the same ablation with a random set of |C| neurons (plus S₀) retains ≤ 25 %.
- **A3 layer necessity (reported, not a criterion):** per layer of C, drop of p when that layer's part of C alone is zeroed.
- Also reported: |C|, its layers, its neurons' Δa in §43.23 (are layer 26's 4 neurons in C?), top-20 tokens where
  each fires in the held-out traces, overlap of C with the top-k₅₀ neurons of §43.23 per layer.

**Reading rule (fixed now).** *Community found* if A1 and A2 pass → phase B. *Necessary, not sufficient* (A1 only):
written as partial, no phase B (the event depends on the dense remainder). *Nothing sparse* (A1 fails): §43.23's
reading stands for the event too; stop.

## Phase B — only if A1 and A2 pass: the community as an edit (GPU ≈ 2 × 1.7 h)

`edit_neurons` (engine/column_edit.py, 0.8): `mode=scale`, `alpha=1.5` and `alpha=2.0` on the `down_proj` columns of C
in each of its layers, `placebo=true` (structural placebo of the op: same columns, same per-column norm change,
random direction; for `scale` the op's documented fallback applies and must be stated in the report). Into Q6_K as in
§43.19 (`ediciones` program, `add-cols-vectors`). Arms on `dev-2048` by `bench_run` (battery on), against the base and
the band of §45.32 (GSM |Δ| 4 with ~24 flips, MATH 1, HumanEval 4-5; **not** the identical-rerun band of §45.30).
- **B1** the knob acts: tokens or cuts move by more than 2× the base band for at least one alpha.
- **B2** GSM8K not below base − 4 and HumanEval not below base − 5 for the alpha that passes B1.
- **B3** the structural placebo stays inside the band on tokens, cuts and GSM8K.
- If B1-B3 pass: compare with the vector at 1.0 (dev 83 / 50) and the compile (§45.8) on the same items; a sealed set
  would need a new one (the three existing sets are spent on Llama), to be decided then.

## Not done here
No search over k, PMI threshold, cap or alpha beyond the values fixed above; no other event (the plural "-s", "an",
dates) unless the driver asks; one model (Qwen3.5-9B); no attribution on generic text (Goodfire's 10M prompts is the
discovery setting; ours is one event on 106 + 28 traces, which is a *targeted* use of their tests, not a replication).

## Prediction
`predictions.md` row 50 (written 2026-11-18, before any code).
