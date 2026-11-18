# FiveTechSoft / weights: Weight-Space Programming (WSP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Website: GitHub Pages](https://img.shields.io/badge/Website-Research_Portal-cyan.svg)](https://fivetechsoft.github.io/weights/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: MiMo-V2.6-Distill-Qwen-9B](https://img.shields.io/badge/Model-MiMo--V2.6--Distill--Qwen--9B-blue.svg)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)

A toolkit and a research log for **Weight-Space Programming (WSP)**: reading, comparing and editing the weights of
LLMs directly, with no gradient step.
- It inspects and edits quantized GGUF files (Q6_K) and safetensors checkpoints.
- It diffs a fine-tuned model against its base and reports what changed, including outside the weights.
- It runs declarative experiments written in WSP-Lang (`.wspl`).

Reading and editing weights needs no forward pass. **Behavioural claims do**: every behavioural result here was
measured by running the model (llama.cpp / llama-server, usually on an RTX 3060).

Starting point: MIT CSAIL's blog post [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
(Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, 2 Nov 2026; a blog post, not peer-reviewed).

## Main result (2026-11-09)

**A fine-tune's skill can be moved onto the unchanged base model as a small adapter, with no training.**
WSP-Lang's `delta_to_lora` op reads the difference between a fine-tune and its base, matrix by matrix, keeps the
strongest directions (exact SVD) and writes a normal LoRA adapter. On Qwen3.5-9B with Jackrong's reasoning distill as
the donor, rank 16, 87 MB, a few minutes:

| Qwen3.5-9B, greedy, 2,048-token budget | base | base + adapter (×0.5) |
|---|---|---|
| MATH-500, 36 held-out items (16 the base cannot finish + 20 it solves) | 6/36 | **30/36** |
| GSM8K, 48 items (another benchmark; never used to pick anything) | 9/48 | **37/48** |
| GSM8K problems the base already solved, kept | 6/6 | **6/6** |
| **Sealed confirmation** (2026-11-10): GSM8K 100 + MATH-500 50 never used by any decision | 43/100, 4/50 | **91/100, 35/50** |

Paired against the base (exact McNemar), both p < 0.0001. The sealed run (plan written before it, results §43.15) confirms with two graders per benchmark (gained/lost 48/0 and 31/0) while a same-norm random placebo stays at the base (46/100, 5/50); much of the gain is that the adapter reasons 2-3× shorter and finishes within the budget. Before any benchmark the op's `energy.json` says whether a
change will move: rank 16 keeps 58.6 % of Jackrong's change (it moves) and 3.8 % of MiMo's (it does not, at any rank
up to 512). Demo and commands: [`demos/math_transplant/`](demos/math_transplant/). It **needs a donor**: every attempt
to create the skill without one (synthesis, learned jumps, merges, self-distillation) failed; see the table below. The
one exception is efficiency: a direction read from the base itself gives most of this gain (next section).
**Does the verdict generalize?** Four donors so far (results §43.11-43.12), verdict given before any benchmark:

| donor | change kept at rank 16 | verdict | what the adapter did |
|---|---|---|---|
| Jackrong reasoning distill (SFT) | 58.6 % | likely | MATH 6 → 30/36, GSM8K 9 → 37/48 |
| iAmBoosted OSS-Distilled (merged LoRA r16), read from the hub | 98.3 % | likely | moved its *behaviour* onto another model (MiMo): ≈ 3× shorter reasoning, MATH answers cut at 8,192 tokens 13 → 3, a same-norm placebo does not; but GSM8K 92 → 83/100 (p = 0.035), so it fails its pre-registered criterion |
| LeoZotos math GRPO (RL) | 7.5 % | unlikely | not distinguishable from the base (GSM8K 15/48 vs 9, p = 0.21) although the full donor improves it (21/48, p = 0.012) |
| MiMo 2.6 (full fine-tune) | 3.8 % | unlikely | does not move at any rank up to 512 |

The verdict was right each time; moving a change does not make the change good (OSS). The main result is confirmed
on a sealed set (`data/sealed_confirm.json`, sha256-checked, disjoint from everything used so far; results §43.15) and next transplants will be too.

> **Re-tested with the corrected op (2026-11-12): holds.** The `contrast_vector` op took the last layer's row after the final norm (transformers' `hidden_states[-1]`), so `direction.31` of the first vector was the norm's rescaling, not a residual change (fixed in 4aeb453, results §45.2-45.3). The vector re-derived with the fix (layers 1-30 identical, layer 31 norm 13.2 → 18.1) was run again under a plan fixed before it (`lab/probes/cc/PLAN.md`, addendum 2026-11-12; prediction row 24 of `predictions.md`): dev GSM8K 50 → 83, MATH 6 → 35; sealed set 43 → 86, 4 → 33, placebo 28 / 1 (results §45.4). The numbers below are for the first vector. The compiled weights were recompiled from the corrected vector (2026-11-13, results §45.8; prediction row 27): layers 1-30 byte-identical to the first compilation, only layer 31 differs; dev GSM8K 85, MATH 30; sealed set 2 (second use) 89 / 25 vs base 46 / 7, placebo 34 / 5, HumanEval-20 19/20. Paired on the same 150 sealed items the corrected layer 31 changes nothing in correctness (3 won / 3 lost, 3 / 4) and halves the loop tails. A closing sentence in the prompt does not do what the direction does: base + "stop reasoning once you have the answer" falls to GSM8K 37 with the same token count, and a neutral sentence to 29 (results §45.9, prediction row 28 failed). The placebo is not one lucky draw: eight more same-norm random directions (seeds 1-8, screen set of 64 MATH items, 2,048 tokens) score 3-8 against base 8 and vector 38; none reaches half the vector's gain, the vector ranks first of nine (results §45.10, prediction row 29). Nor is it a token budget: the base re-run at 8,192 tokens repeats 72/100 item by item while the vector at 8,192 scores 85 with a quarter of the tokens (16 won / 3 lost, p 0.004; results §45.11), and the compiled weights at 8,192 score 90 / 41 (19 won / 1 lost against the base; equal or better than the vector, 6/1 and 6/2; results §45.13).

**Without a donor: the base's own "conclude" direction** (results §43.17-43.18). Replicating Concept Country's idea on
Qwen3.5-9B: the `contrast_vector` op reads, from the base's own finished reasoning traces, the mean residual stream of
the last 48 tokens before `</think>` minus the rest of the thinking, and applies it as a llama.cpp control vector at
scale 1.0 (picked on unused items). No donor, no training. Same sealed set, same 2,048-token budget:

| Qwen3.5-9B, sealed set (150 items never used by any decision) | base | + "conclude" vector | same-norm random vector |
|---|---|---|---|
| GSM8K /100 | 43 | **87** | 23 |
| MATH-500 /50 | 4 | **25** | 2 |
| answers cut at the budget (GSM8K) | 57 | 10 | 76 |

Gained/lost vs base 46/2 (GSM8K) and 21/0 (MATH), exact McNemar p < 1e-5 with two graders each; plan written before the
run. A push of the same size in a random direction *hurts*, so the gain is the direction. It is efficiency, not new
knowledge: the base already has most answers and fails by not closing (with 32k tokens it solves MATH, arXiv
2611.09382).

**Holds on a second family (results §45.15, 2026-11-15).** Same recipe, same code, on Microsoft's Phi-4-mini-reasoning
(3.8B, `phi3`, another tokenizer and template), 1,024-token budget (the model runs out of budget there, not at 2,048):
scale 0.5 picked on screen items; sealed set (first use for this model) GSM8K 59 → **82**/100, MATH 4 → **19**/50 (27 won /
4 lost, 16 / 1, two graders each), tokens −34 % / −12 %; same-norm random vector 58 / 6. No finished answer skips the
thinking phase at 0.5; at scale 1.0 (Qwen's) the model does skip it and scores 5/64 on the screen set. Plan written before
the run (`lab/probes/cc_phi/PLAN.md`; prediction row 34 partial: direction and controls right, magnitudes larger than
predicted).

**And compiled into its weights (results §45.16, 2026-11-15).** The same `compile_control_vector` op, unchanged, writes the
Phi direction (× 0.5) into `down_proj` of 31 layers (G0 0.044; cos(written, dW) 0.74 after Q6_K); the merged file served
with no flags scores dev GSM8K 59 → **81**/100, MATH 6 → **19**/50 (vector 85 / 19), C1-C4 pass; sealed set 2 (first use for
this model) 62 → **86**, 7 → **14** (26 won / 2 lost, 7 / 0), the vector 88 / 15 on the same items (merged vs vector 2/4 and
2/3: equal), same-norm compiled placebo 60 / 2 (dev 54 / 5). No finished answer skips the thinking phase. HumanEval-20
11 → 16 (reported, n = 20). Prediction row 35: 18 of 22 readings in band, all the primary ones.

**Budget control on Phi (results §45.17, 2026-11-16).** At 2,048 tokens the base recovers much of the gap (GSM8K 59 → 81,
MATH 6 → 26; 25 of the 42 GSM8K items it cut at 1,024 now right), the merged model still scores **91** / 29 (12 won / 2 lost
vs base on GSM8K, p 0.013; 7/4 on MATH, not significant) and the vector 85 / 35 (11/7; 13/4, p 0.049); merged vs vector 9/3
and 3/9, neither significant. Tokens −37 % / −18 % against the base; no finished answer skips the thinking. Every merged win
but two comes from an item the base cut (AHA's ceiling rule holds: 10 of 17 cuts on GSM8K, 7 of 24 on MATH). Reading: on
Phi, half of the GSM8K gain at 1,024 and most of the MATH gain was the budget; +10 on GSM8K survives a doubled budget, MATH
is still cut 24 / 50 by the base at 2,048 and is not settled. Smaller beyond the budget than on Qwen (+18 / +20 at 8,192).
Prediction row 36: 14 of 18 readings in band.

**Does not hold on a third family at scale 0.5 (results §45.19-45.20, 2026-11-17).** Same recipe, same code, same pre-registered
criteria on DeepSeek-R1-Distill-Llama-8B (`llama`, 2,048-token budget, scale 0.5 picked on screen items; 1.0 breaks the
model as on Phi). Dev passes: GSM8K 74 → 76, MATH 22 → **32** (13 won / 3 lost, p 0.021, two graders), tokens −46 % / −26 %,
same-norm placebo 76 / 23. **The sealed set does not confirm:** base 83 / 28, vector **76** / 26 (7/14 and 7/9), placebo
82 / 23; pooled over both sealed sets the vector is 24 won / 29 lost on GSM8K and 17 / 14 on MATH. The vector still makes
the model conclude (cuts 13 → 2, tokens −41 %), but the items it loses are ones the base finished right and it finishes
wrong, as many as it recovers from the base's cuts. Compiled into `down_proj` with the unchanged op (G0 0.030, cos 0.74),
the merged file equals the vector item by item on sealed set 2 (7/7, 4/5) and, like it, does not beat the base (75 / 23
→ 77 / 27, p 0.84 / 0.34); HumanEval-20 14 → 13. Prediction rows 37 and 38 both failed on their primary reading (the
sealed confirmation), the first dev PASS of the series the sealed set rejected. Reading: on this family the direction is a
**token saving at equal accuracy**; the compile primitive and the placebo's null are now shown on three families, the
accuracy gain on two of three. Plan `lab/probes/cc_llama/PLAN.md`, fixed before the download.

**At a lower dose it confirms on the third family, with small print (results §45.21, 2026-11-17).** Part C of the same
plan, fixed before any run: the Llama vector at 0.25 and 0.35 on the dev items (0.25: GSM8K 74 → 84, MATH 22 → 31; 0.35:
81 / **37** with 15/0 on MATH), a pre-registered gate that picked 0.35, and a third sealed set (200 GSM8K + 100 MATH-500,
first use): base 158 / 40, vector **156 / 58** (GSM8K 14/16, p 0.86; MATH **20/2**, p 0.0001 with both graders), same-norm
placebo 152 / 38 (n.s.), tokens −31 % / −18 %: **confirms** (C3, GSM8K ≥ base − 2, at its exact boundary). Where the gain
comes from: 19 of the 20 MATH items gained were ones the base cut (58 cuts of 100 at 2,048; among the 42 the base finishes,
40 → 39); on GSM8K the 14 gained are recovered cuts and the 16 lost are base-correct items finished wrong, the §45.19
mechanism at half the rate (10 % of base-correct finished items vs 17-20 % at 0.5). Reading: the sealed accuracy gain of
the conclude direction is now shown on all three families, on Llama-8B only at 0.35, only on MATH and as cut recovery; the
dose halves the cost of premature closure, it does not remove it. Third sealed set spent on this family (scale chosen on
dev after 0.5 failed on the first two; gate pre-registered). Prediction row 40 failed on its primary reading: we gave "does
not confirm" 60 % and it confirmed.

**alphaXiv's pruned Exp 8 edit in our harness: the harness cannot tell (results §45.22, 2026-11-18).** Plan fixed
before the run (`lab/probes/exp8_prune/PLAN.md`, prediction row 41): the MIT Exp 8 edit on Qwen3.5-0.8B pruned to its
4,096 largest entries per layer (6 %; keeps 74-76 % of the edit's energy and 76-79 % of its push along the edit direction,
two methods), its complement (the other 94 %), and the full edit run again, on the §45.12 harness (GSM8K 100 + MATH 50,
greedy, 2,048 tokens). GSM8K correct: full 8 (13 on 2026-11-14 with the same file, binary and items), pruned 2,
complement 5; MATH at the floor everywhere. Discordant pairs between the two runs of the *same* edit: 11 of 100. The
pruned-vs-full and complement-vs-full differences are the size of that noise, so the only statement that survives is
"pruning does not improve anything"; whether 6 % of the entries keeps the effect is not decidable at n = 100 with
batched greedy decoding, and alphaXiv's ± 3 pp screen cannot decide it either. Row 41 scored partial (the primary
reading "R1 fails" came out, for a reason the prediction did not foresee). Lesson written down: run the same
configuration twice before comparing two edits on the 0.8B; if the expected difference does not double the band, do not run.

**Ternary weights: our edits do not fit in {−1, 0, +1} at their working dose (results §45.23, 2026-11-18).** Asked
after *ttf* (ternary Qwen3.8-27B): plan fixed first (`lab/probes/ternary/PLAN.md`, row 42), then `engine/ternary.py`
and the read-only WSP-Lang op `ternary_survival` (11 tests written before the code; independent numpy recount agrees).
Under ggml's TQ2_0 rule only 10-12 % of a `down_proj`'s entries are non-zero at all, and a 0.8B quantised that way
without recovery training answers 0 of 10 and emits multilingual fragments (which is why *ttf* trains 72M tokens
after converting). Ternarising an edited model keeps most of the edit as a *component* of the ternary difference
(alpha 0.6-0.9) but buried in orthogonal junk 2× (Exp 8 edit) to 12× (our compiled conclude vector) larger: direction
cosine 0.42-0.47 for Exp 8 and 0.08-0.10 for the compile at dose 1; Exp 8 needs 10-20× its dose before the edit is the
dominant content, the compile never reaches it within 20×; BitNet's absmean rule flips more entries but loses direction
as the dose grows. Decision: no ternary writer; a ternary backend would carry our edits only in float scales or
adapters, which `delta_to_lora` and `compile_control_vector` already produce. The op stays as the instrument.

**Solving instead of rounding: a discrete per-row writer, CPU prototype (results §45.24, 2026-11-18).** Plan and
prediction (row 45) first, then `lab/probes/discrete_writer/writer.py`: for each output row, choose per entry the
native bf16 move (sign flip, ×2, ×½) that brings the row closest to its target change, with an exact per-entry
threshold and a sparsity price. On our compiled conclude vector (9B, layers 1 / 16 / 31) it reaches cosine 0.45-0.67
at dose 1 touching about 3 % of the entries (rounding gave 0.08-0.10), and 0.77-0.78 on the Exp 8 edit; it does not
carry the magnitude (alpha 0.26-0.46) and never passes cosine 0.7 at any dose, because every move on an entry larger
than its target overshoots. A discrete writer for our edits needs the activation-coupled objective (Tree-Ising's), which
needs one 9B forward over the calibration prompts; specified in the plan, not launched. Row scored partial: the
primary reading held by the letter, the declared surprise (a layer above 0.5) fired.

**Ternary writing in WSP-Lang, and how far a ternary NanoWSP stays compatible (results §45.25, 2026-11-18).** Asked
by the driver after §45.23: ops `ternarize` (a ternary copy W = diag(s)·T with codes, per-row scales, a float
materialisation so every existing op runs, and a manifest) and `imprint_ternary` (the closed-form association written
in float, projected to {−1, 0, +1}·s by an exact per-row solve, with TTF-style re-imprints of the deficit). Same
associations written into a float NanoWSP and into its ternary twin, 3 seeds, capacity = largest n with exact recall
≥ 0.95: float 144 / 144 at widths 64 / 128; ternary 8 / 32 whether converted after writing or written in ternary
(ratio 0.056 / 0.22). The write's direction survives (cosine 0.88); what is lost is cancellation precision between
near-collinear keys, which the pseudo-inverse amplifies; the re-imprint diverges. Existing ops read the ternary
checkpoint; none of the three float edits tested survives re-ternarisation. Prediction row 46 failed (we bet ≥ 50 %
of float capacity). Compatibility ends at the first write that needs less than one scale step per row.

**Can a benchmark be read from the weights alone? Partly, and we measured which part (results §45.26, 2026-11-18).**
78 arms with a weight artefact and a paired benchmark result (Qwen3.5-9B, Phi, Llama-8B, Qwen3.5-0.8B; 14 real edits
against 17 same-norm placebos), weight-only features, prediction row 47 first. Magnitude features separate nothing
(AUC 0.21-0.40; the placebos share the norm by construction). Alignment with the `</think>` readout row over the last
eight layers separates real edits from placebos with AUC 0.93 (1.00 inside every family) and ranks edits by the token
saving they will produce within a family (Spearman −0.72 / −0.69 / −0.58 on 9B / Phi / Llama). Nothing in the weights
reads correctness: pooled |ρ| ≤ 0.42, and the dose series show why (one direction per artefact, outcomes 7 → 12 → 30
→ −5 as the scale grows). Row scored partial. Caveats: one lab's arms, small n, features chosen knowing the phenomenon.

**Running now (2026-11-18, GPU queue 18 → 19 → 17; plans and predictions committed before each run).** Row 43: a
constant `logit_bias` on `</think>` (+2 … +8) against the conclude vector on the 9B dev items, the control DAWE and we
both lacked (`lab/probes/cc_bias/PLAN.md`). Then the preservation battery on the 9B base (`engine/preserve_bench.py`:
HumanEval 164, 40 home-made instruction checks, garbage rates; `lab/probes/preserve/PLAN.md`). Then row 44: a second
behaviour, "reason in Spanish on English problems", same protocol as the conclude direction
(`lab/probes/skill2/PLAN.md`). Results land in `results.md` §45.25+ and in the rows when they finish.

**Evidence synthesis (16 November):** [what the Phi results establish, and what they do not](docs/phi_findings_2026-11-16.md).
Measured transfer, compilation, budget controls, practical uses and limits, with AHA / WeightBench / Auto-WSP context.

**Compiled into the weights** (results §43.19 dev, §43.21 sealed): the same vector written into `down_proj` of 31 layers as a
closed-form rank-1 edit per layer (no gradients, same matrices and shapes), merged into the Q6_K file, run with no
flags: dev GSM8K 50 → **87**/100, MATH 6 → **34**/50, not different from the control vector (84 / 32; p 0.45 / 0.73).
The same write in a random direction hurts (28 / 6). **Confirmed on a second sealed set** (made before the run):
GSM8K 46 → **89**/100, MATH 7 → **26**/50 (gained/lost 43/0 and 19/0, two graders each), as the vector (90, 22); the
merged random-direction placebo falls below base (30, 4). Outside maths the merged file costs little (HumanEval first
20: 19/20, base 20/20; the control vector 15/20).

**Second use: steering without touching weights** (results §43.13). The `contrast_vector` op turns two sets of answers
(same prompts, cheerful vs neutral) into a llama.cpp control vector. On 40 unseen prompts, Qwen3.5-9B with the vector at
scale 0.25: judged tone 2.83 → 3.90 of 5 (sign test p = 3·10⁻⁸), positive-affect words 0.27 → 2.63 per 100 (p = 7·10⁻¹²),
37/40 coherent; a same-norm random vector changes nothing (+0.05, +0.02). Passed all five pre-registered criteria
after two fixes to the judge, both made before the test set was used (documented). Scales ≥ 0.5 break coherence.
**Does not hold with the corrected op** (results §45.3): same tone gain (+1.07), but 34/40 coherent and GSM8K fast
5 vs 9, failing C4/C5. §43.13 passed only with the old, wrong last layer.

🌐 Research portal: [fivetechsoft.github.io/weights](https://fivetechsoft.github.io/weights/) ·
📒 Full log: [`results.md`](results.md) · 🎯 Pre-registered predictions and their hit rate: [`predictions.md`](predictions.md) · 🗺️ Plan: [`roadmap.md`](roadmap.md) ·
🧪 Weight-diff findings: [`lab/probes/delta_patterns/findings.md`](lab/probes/delta_patterns/findings.md)

---

## Status of the results (November 2026, audited)

Every claim below was re-checked against the files in this repo (2026-11-06; rows on reusing a fine-tune's change
added 2026-11-09). "n" is the number of evaluated
items. Results that did not survive the audit are listed as such instead of being removed.

| Result | Evidence | Status |
| :--- | :--- | :--- |
| **Fine-tune → adapter (`delta_to_lora`)**, Jackrong's distill on Qwen3.5-9B (results §43.4-43.6) | See *Main result*. Rank 4 fails (11/36), rank 8 27/36, rank 16 30-32/36. HumanEval first 20: 20/20 base, 19/20 with the lab adapter. | ✅ Measured, paired tests; **confirmed on the sealed set** (91/100, 35/50 vs base 43, 4; placebo 46, 5; results §43.15). |
| **Transfer verdict on 4 donors** (results §43.11-43.12) | See *Main result*: `likely` for Jackrong (58.6 %) and OSS-Distilled (98.3 %), both transferred; `unlikely` for LeoZotos (7.5 %) and MiMo (3.8 %), neither did. | ✅ Measured, small sets; LeoZotos's test is weak (adapter between base and donor, neither difference significant). |
| **Shorter reasoning on MiMo with OSS-Distilled's change** (results §43.12) | GSM8K 100 + MATH-500 50, 8,192 tokens: MATH cut 13 → 3 (×1) / 5 (×0.5), tokens 3,077 → 965; placebo 17 cut. GSM8K 92 → 83 at both doses (p = 0.035 / 0.012, two graders agree). | ❌ Fails its pre-registered criterion: the behaviour moves, accuracy drops. |
| **Happiness control vector (`contrast_vector`)** (results §43.13) | See *Main result*. Judge (MiMo) and lexicon agree; placebo null; GSM8K fast set 7/48 vs 9/48 (within the margin of 2). | ❌ Passed C1-C5 only with the old op; with the last layer corrected it fails C4/C5 (34/40 coherent, GSM8K 5 vs 9; §45.3). |
| **MiMo 2.6 vs DeepSeek-V4-Pro-Qwen3.5-9B-heretic** (results §43.14) | Same protocol: GSM8K 92 vs 89 (p = 0.55), MATH-500 41 vs 33 (p = 0.10; second grader 36 vs 33, p = 0.61); heretic thinks ≈ 2.5× less (1,223 vs 3,077 tokens on MATH, 3 vs 13 cut). | ✅ Statistical tie on math. HumanEval not used: heretic's code comes back with 1-space indentation in 13/20 (cause not verified). Refusals not measured. |
| **MiMo's change does not fit an adapter** (results §43.4-43.5) | Full fine-tune, diffuse change: rank 64-512 adapters reach ≤ 4/16 on the stuck MATH items vs MiMo's own 12/16. | ✅ Negative result, predicted by `energy.json`. |
| **Copying MiMo's behaviour instead of its weights** (results §43.7-43.10) | Rank-16 adapter trained on MiMo's own answers: MATH 6 → 16 / 24 / 27 of 36 with 93 / 171 / 256 answers, GSM8K 9 → 34 / 41 / 41 of 48. Closed-form, gradient-free fit of `down_proj` to MiMo's hidden states (the WSP/MEMIT write used for a behaviour): 17/36, 25/48; a same-norm random placebo gives 10/36, 14/48 (not significant vs base, p 0.13 / 0.23), and the fit beats it (p 0.039 / 0.003). Writing the attention output too does not help (15/36, 26/48). | ✅ Measured. Distillation flattens after 171 answers; the gradient-free route is closed. |
| **Donor-free "conclude" control vector** (results §43.17-43.18) | Direction from the base's own traces, scale 1.0: dev GSM8K 50 → 84/100, MATH 6 → 32/50; sealed 43 → 87, 4 → 25; same-norm placebo below base both times (37/3, 23/2). | ✅ **Confirmed on the sealed set**, two graders each, and **confirmed again with the corrected op** (layer 31 re-derived: sealed 86 / 33 vs base 43 / 4, placebo 28 / 1, §45.4). One model, one budget (2,048), greedy; efficiency at a fixed budget, not a new capability; compiled into `down_proj` it passes dev (87 / 34, §43.19) and sealed set 2 (89 / 26 vs base 46 / 7, §43.21); recompiled from the corrected vector it does the same (dev 85 / 30, sealed set 2 89 / 25, placebo 34 / 5, §45.8). |
| **The 9B's "conclude" direction moved into the 0.8B through activation maps** (gate 3b, results §45.12; plan `lab/probes/act_map/PLAN.md`, prediction row 30) | The 9B's contrast mapped per layer by the gate 1 maps, applied as a control vector to Qwen3.5-0.8B at 2,048 tokens: GSM8K finished 0 → 83/100, correct 0 → 7 (7 won / 0 lost, p 0.016, two graders); the 0.8B's own contrast over the same texts 37 finished / 13 correct; same-norm placebo 3 / 1; embedding-map control 0 / 0; the MIT Exp 8 edit 13 / 13. | ✅ Passes its pre-registered rule (port gain 10 ≥ half the reference's 16), **but** what crosses the map is "stop thinking" (median 8 characters of thinking, 8 % of finished answers correct vs 35 % own, 100 % Exp 8), not "finish when the answer is there". Prediction row 30 failed (we expected the gate not to pass). Dose curve (§45.14): finished 0 / 3 / 15 / 50 / 52 at scales 1.0 / 1.25 / 1.5 / 1.75 / 2.0, and from 1.5 up every finished answer skips the thinking phase; the 0.8B answering without thinking gets ≈ 40 % of GSM8K right. One pair, greedy, one budget. |
| **"Conclude" direction on a second family: Phi-4-mini-reasoning 3.8B** (results §45.15; plan `lab/probes/cc_phi/PLAN.md`, prediction row 34) | Same recipe and code (`phi3`, other tokenizer and template, 1,024-token budget, scale 0.5 chosen on screen items): dev GSM8K 59 → 85/100, MATH 6 → 19/50; **sealed set, first use for this model: 59 → 82, 4 → 19** (27/4 and 16/1, two graders each), tokens −34 % / −12 %; same-norm placebo 58 / 6 (dev 47 / 5). Every finished answer at 0.5 keeps its thinking phase (median 2,072 → 1,318 characters); at 1.0 the model skips it (screen 5/64, 63 finished). | ✅ **Confirmed on the sealed set.** First result outside the Qwen family; the scale and the per-layer profile change between families (vector norm 27.2 vs 18.1; peak at layer 30), the placebo's sign does not. One budget (1,024, harsher than Qwen's), one placebo seed, greedy, Q6_K. **Compiled into `down_proj` with the same op (§45.16, prediction row 35):** dev 81 / 19 (C1-C4 pass), sealed set 2 base 62 / 7 → merged **86 / 14** (26/2, 7/0), vector 88 / 15 on the same items (equal), compiled placebo 60 / 2; HumanEval-20 11 → 16. The compile now holds on two families. **Budget control at 2,048 (§45.17, row 36):** base recovers to 81 / 26; merged **91** / 29 (12/2, p 0.013; 7/4 n.s.), vector 85 / 35; gains come from the base's cuts (AHA ceiling); on Phi half the GSM8K gain and most of the MATH gain at 1,024 was budget. |
| **"Conclude" direction on a third family: DeepSeek-R1-Distill-Llama-8B** (results §45.19-45.20; plan `lab/probes/cc_llama/PLAN.md`, prediction rows 37-38) | Same recipe, code and criteria (`llama`, 2,048 budget, scale 0.5 on screen items; 1.0 scores 4/64). Dev: GSM8K 74 → 76, MATH 22 → 32 (13/3, p 0.021, two graders), tokens −46 % / −26 %, placebo 76 / 23 (C1-C3 pass). **Sealed set 1: base 83 / 28, vector 76 / 26 (7/14, 7/9), placebo 82 / 23 — does not confirm**; both sealed sets pooled 24/29 and 17/14. Compiled (G0 0.030, cos 0.74, ‖dW‖/‖W‖ 0.012): dev 74 / 35 (C1-C4 pass), sealed set 2 base 75 / 23 → merged 77 / 27 (p 0.84 / 0.34), vector 77 / 28 (merged = vector, 7/7 and 4/5), compiled placebo 79 / 20; HumanEval-20 14 → 13. **Dose (Part C, §45.21, row 40):** dev 0.25 → 84 / 31, 0.35 → 81 / 37 (15/0 on MATH); gate picks 0.35; **sealed set 3 (200 + 100, first use): base 158 / 40 → vector 156 / 58 (14/16 p 0.86; 20/2 p 0.0001, two graders), placebo 152 / 38, tokens −31 % / −18 %: confirms** (C3 at its boundary). 19 of 20 MATH gains are base cuts recovered; 16 GSM8K losses are base-correct items finished wrong (10 % of them vs 17-20 % at 0.5). | ❌ at 0.5 (rows 37-38: loses as many items as it recovers, finished wrong not cut) → ✅ **at 0.35 on sealed set 3 (row 40 scored ❌ because we predicted no confirmation)**. On this family the direction saves tokens and recovers the budget's cuts (MATH only, at 2,048 where the base cuts 58 %), at the price of finishing wrong 10 % of what the base got right; the lower dose halves that price, it does not remove it. Third sealed set spent here, scale chosen on dev after 0.5 failed. One budget, greedy (card says T 0.6), Q6_K, one placebo seed; compile not rerun at 0.35. |
| **Creating a skill without a donor** (results §40-§42, §43.8) | Qwen3.5-0.8B, 3×3 multiplication: a training's change lives in a narrow subspace shared across seeds (overlap 0.72-0.80, chance 0.016); rank 20 recovers 91 % of it, another seed's subspace 92 %. But extrapolating early steps, one gradient, coefficients in the subspace, a learned jump (0/7 tasks) and summing changes all fail; donor-free self-distillation on 9B = base. | ❌ Not achieved. Moving an existing change works; making one does not. |
| **MIT Exp 8 replication** (Qwen3.5-0.8B loop edit) | Edited tensors are bit-identical to Astra's (SHA-256 match with the manifest in `exp8_control/p50c_edit/`). On a MATH-500 holdout, Q6_K, n=20, temp 0.3, 2048-token cap: base 1/20, edit 14/20, random-direction placebo 0/20 (Fisher p ≈ 3.9e-5 edit vs base). Files: `rig_runs/holdout-08b-{base,edit,placebo}-20.json`. | ✅ Replicated, small n. A GSM8K smoke test (n=5, 512-token cap) gave 0/5 vs 1/5, with **all 10 answers cut at the cap**, so it shows no loop break. |
| **Exp 8 idea on a trained nanoGPT** (results §36) | Loops 111 → 26 of 256 test prompts. One of 3 random edits of the same norm also gave 26. | ❌ Fails the pre-registered criterion 4. |
| **Loop fix on MiMo-9B** | Blind holdout, n=40: loops 17/40 with the edit vs 20/40 with the placebo, p = 0.654 (results.md, `DISCARD_CANDIDATE_V1`). GSM8K n=50 accuracy was kept: 45 → 46. | ❌ Discarded as a fix: not distinguishable from the placebo. |
| **What post-training changes** (Qwen3.5-9B vs MiMo, NeoHorse, tmax, Fara, official post-training, WeMM, Nyx-RP, Angry-Claudius, DeepSeek-V4 distill/Heretic) | `lab/probes/delta_patterns/findings.md` F1–F26, with two-method checks. | ✅ Measured. Full fine-tunes share *where* they change the weights whatever the goal (F22). The layer-specific *directions* are shared by the improvements and not by a full fine-tune with another goal (F23). A LoRA control cannot separate a degradation from a neutral LoRA (F24). A storage-cast bug (F32→BF16 counted as change) was found and corrected (F26). |
| **`model_report`**: what a model changed vs its base, in about a minute over HTTP | Validated on 6 models whose change is known (`lab/runs/model_report/validation.json`), plus config/tokenizer/template diff (F25: MiMo ships an older pre-tokenizer regex). | ✅ Validated on Qwen3.5-9B derivatives. Other architectures: tensor names are mapped (roadmap J1) but not yet calibrated. |
| **TwIL-8B vs MiMo-9B benchmark** (results §37) | Same protocol: GSM8K 100 + MATH-500 50, greedy, 8192 tokens. TwIL 96/100 and 44/50. MiMo 93/100 and 34/50, with 19 answers cut at the cap. | ✅ Measured. Base Qwen3.5-9B under the same protocol: GSM8K 60/100, MATH-500 24/50, with 36 + 25 answers cut at the cap; MiMo's advantage is mostly finishing within the budget (results §39). |
| **Atari policy reading** (MIT Exp 1) | W3 statistics: vertical/horizontal 1.54× for Pong, 1.53× for Breakout. One MiMo-9B query, given hand-computed statistics (not raw weights), answered "Pong" with self-reported 90 % confidence. | ⚠️ Anecdote (one query). |
| **LEACE-style closed-form edit** | Rank-1 projection zeroes ‖uᵀW‖ for one direction in layer 14 `down_proj` in 47 ms (`rig_runs/leace_verification_report.json`). | ⚠️ True by algebra. No data, probe or behaviour tested. |
| **NanoWSP-2** (599,168 params) and **NanoWSP-3** (3,280,128 params) | They recall all imprinted pairs: 11/11 and 23/23. | ⚠️ Memorization testbeds. Recall measured on the stored items only; no generalization tested. Not language models. |
| **MiMo → TinyLlama fact transfer** | Real run: target facts 5/8 → 7/8, holdout preserved 5/8 (`rig_runs/mimo_to_tinyllama_full_migration_report.json`). | ⚠️ n too small. The earlier "0 → 100 %, 18.4 ms, 0 % collateral" report was written from hardcoded values (`scripts/save_tinyllama_benchmark_report.py`) and is **withdrawn**. |
| **MiMo "self-critique" causal contract** (50 % in all 4 arms) | `rig/exp_mimo_self_critique_v1.py` scores a synthetic projection of W on a prompt hash. It never runs the model. | ❌ **Withdrawn**: not a behavioural test. |
| **"Evil" neurons in MiMo layer 14** | `edits/evil_mimo.json` scales 4 `ffn_down` columns (4258 and 7714 ×2.5; 10739 and 993 ×0.25), 16,384 weights. | ⚠️ No behavioural evaluation of hostility was run. The GGUF is not in the repo. |

---

## ⚡ Quick start

```bash
git clone https://github.com/FiveTechSoft/weights.git
cd weights
pip install gguf numpy matplotlib pillow   # GGUF inspection and editing (wsp.py)
pip install torch safetensors              # NanoWSP, nanoGPT, weight diff, model_report
```

### Move a fine-tune's skill onto the base (`delta_to_lora`)
```wspl
experiment transplant_jackrong_math:
steps:
    step delta_to_lora base=hf_cache/qwen35-9b-hf model=hf_cache/dp_keep_dsv4_distill rank=16 out=hf_cache/close_lora/demo_jack_r16 gguf=hf_cache/close_lora/demo_jack_r16.gguf
```
```bash
python -m engine.wsplang programs/transplant_jackrong_math.wspl --run
llama-server -m <Qwen3.5-9B base>.gguf --lora hf_cache/close_lora/demo_jack_r16.gguf --lora-init-without-apply --port 8095
python demos/math_transplant/demo.py 8095        # same server: adapter off vs on
```
`model=hf:<owner>/<repo>[@revision]` reads the fine-tune from the Hugging Face hub by byte ranges instead of a local
copy. The step returns `verdict` (`likely` / `uncertain` / `unlikely`) from the weighted energy kept at rank 16, before
any eval. `gguf=` needs `WSP_LORA_CONVERTER` (llama.cpp's `convert_lora_to_gguf.py`). One score for an adapter:
`lab/probes/close_probe/reward.py` (IWIBench reward = gain on stuck items × retention of solved ones; a no-op scores 0).

### Steer a mood with a control vector (`contrast_vector`)
```wspl
experiment happy_vector:
steps:
    step contrast_vector model=hf_cache/qwen35-9b-hf pos=lab/runs/happy/pairs/pos.jsonl neg=lab/runs/happy/pairs/neg.jsonl out=hf_cache/cvec/happy.gguf placebo=2026
```
```bash
python -m engine.wsplang programs/happy_vector.wspl --run      # pos/neg: jsonl rows {prompt, response}
llama-server -m <Qwen3.5-9B base>.gguf --control-vector-scaled hf_cache/cvec/happy.gguf:0.25 --port 8095
```
The full measured study (pairs, judge, lexicon, placebo, retention) is `lab/probes/happy/` with its plan in `PLAN.md`.

### Inspect one weight of a GGUF
```bash
python wsp.py peek <path-to-gguf> blk.19.attn_output.weight -r 100 -c 200
```
The tool dequantizes the Q6_K block on the fly. It prints the weight's role (layer, head, residual dimension) and
its projection on 10 centred semantic centroids built from token embeddings. Those labels are a lexical heuristic,
not a measured function of the weight.

### What did a fine-tune change? (`model_report`)
```wspl
experiment report_mimo:
objective: how MiMo differs from Qwen3.5-9B
steps:
    step model_report model=XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B base=Qwen/Qwen3.5-9B out=lab/runs/model_report/mimo.md
```
```bash
python engine/wsplang.py <program>.wspl --run
```
It reads one block of rows per tensor by HTTP range requests, with no full download, and classifies the change:
`identical`, `directional_ablation`, `lora_like` or `full_fine_tune`. It also diffs config, tokenizer and chat
template, and runs an encode test over 14 scripts. Example output: [`lab/runs/model_report/`](lab/runs/model_report/).

### NanoWSP: a micro-Transformer whose associations are written in weight space
```bash
python scripts/demo_nanowsp_capabilities.py
```
2 layers, d=64, 80,704 parameters. Associations are stored by closed-form projections. It is a lookup testbed,
not a language model.

### nanoGPT W0 declared as data

Instantiate an untrained `nanoGPT` (10.77M parameters, 6 layers, d=384, 6 heads, vocab=65) and check it against
upstream `karpathy/nanoGPT@3adf61e`.
- The architecture is declared in `programs/arch/nanogpt.wsparch` and compiled by `engine/wsp_arch.py`, with no
  model class behind it ([`docs/wsparch.md`](docs/wsparch.md)).
- Training is a separate gradient-descent step (`programs/train_nanogpt_shakespeare_char.wspl`).
- `programs/synthesize_nanogpt_ngram.wspl` compiles an n-gram table into the same graph without gradients. That is
  a lookup table, not a synthesized language model.

```bash
python engine/wsplang.py programs/build_karpathy_nanogpt.wspl --run
python lab/verify_official_karpathy_parity.py
```

| Parity test | Observed | Verdict |
| :--- | :--- | :--- |
| Logits, forward pass | Δ = 0.0 (fp32) | exact match |
| Weight tying (`wte` ↔ `lm_head`) | ‖Δ‖_F = 0.0 | identical |
| Causal mask | past leakage 0.0 | strict |
| Greedy generation | 0 discrepancies / 75 tokens | bit-exact |
| Loss and gradients | Δ = 0.0 | exact match |

This shows the runtime matches upstream at W0 (same seed, same weights). The checkpoint is random, with validation
NLL near ln 65. For language-model behaviour use `lab/verify_llmness.py`, calibrated by `lab/calibrate_llmness.py`.

👉 Full walkthrough: [Getting Started Guide](docs/getting_started.md).

---

## 🧠 WSP-Lang

A declarative language for weight-space experiments (runtime version 0.8.0; 0.8 adds `edit_column` / `edit_neurons`
with a structural same-norm placebo and a parameter budget, `.weightpatch.json` patches with `apply_patch` /
`rollback_last` / `list_patches` / `commit_manifest`, and `ternary_survival`; see `docs/wsplang_0.8_spec.md`). Steps cover:
- inspection: `peek`, `semantic_map`;
- edits under a guard: `project_out`, `scale-cols`, closed-form imprints; `edit_column` / `edit_neurons` with a structural placebo (0.8);
- patches: `apply_patch`, `rollback_last`, `list_patches`, `commit_manifest` (0.8);
- measurement of edits under quantisation: `ternary_survival` (0.8); ternary NanoWSP checkpoints: `ternarize`, `imprint_ternary` (0.8.1, results §45.25);
- **benchmarks as a section of the language (0.8.1):** `bench_run suite=dev-2048|dev-8192|sealed-N|nanowsp-wspx` (standard suites shipped in `engine/bench_suites/`, digest-verified; mandatory identity: served model sha, control vector / LoRA sha, binary sha, suite sha; optional `provenance=` verified against the served sha so a WSP-Lang-generated model is recorded as base@sha + patches; model classes `llm-chat-think` / `llm-chat` / `nanowsp`, the last run in-process), `bench_compare` (paired item by item, exact binomial on both graders, same-norm placebo, noise band from the base run twice, preservation battery; verdict "improves / harms / same" only for an edit against its own base, "two models" otherwise), `bench_table` (registry `lab/runs/bench/index.json` → [`docs/wsplang_benchmarks.md`](docs/wsplang_benchmarks.md)); `evaluate_benchmark` now goes through the same path. Rule and format: [`docs/wsplang_0.8_bench_addendum.md`](docs/wsplang_0.8_bench_addendum.md);
- weight diffs: `delta_stats`, `delta_signature`, `delta_compare`, `model_report`, `delta_to_lora` (a fine-tune's change as a reusable LoRA adapter, local or from the hub, with a transfer verdict);
- steering: `contrast_vector` (a llama.cpp control vector from two sets of answers, with a same-norm placebo);
- behavioural evaluation: `evaluate_benchmark` against a running server.

Spec: [`docs/wsplang_specification.md`](docs/wsplang_specification.md).

```wspl
experiment compare_angry:
objective: does a deliberate degradation share the improvements' layer-specific directions?
steps:
    step delta_compare a=lab/runs/delta_patterns/sig_angry_claudius.pt b=lab/runs/delta_patterns/sig_mimo.pt out=lab/runs/delta_patterns/pairs_angry_claudius_mimo.json
```

**Semantic centroids.** Raw token embeddings sit in a narrow cone (baseline cosine ~0.7). WSP-Lang therefore
centres its 10 semantic centroids on their mean: $\mathbf{C}'_a = \mathbf{C}_a - \frac{1}{A}\sum_k \mathbf{C}_k$.
The contrast gain this gives has not been measured.

---

## 🔬 Core components

| Component | Description |
|---|---|
| [`wsp.py`](wsp.py) | GGUF Q6_K inspector and in-place editor; NumPy port of `quantize_row_q6_K_ref` (87 % byte-identical on unedited blocks, error ~6e-6). |
| [`engine/wsplang.py`](engine/wsplang.py) | WSP-Lang parser, validator and runtime (`--run`). |
| [`engine/weight_diff.py`](engine/weight_diff.py) | Tensor-by-tensor diff of two checkpoints. It reads safetensors locally or by HTTP ranges and maps tensor names across architectures. Per tensor it gives relative change, spectrum and subspace signatures; storage casts are not counted as change. |
| [`engine/contrast_vector.py`](engine/contrast_vector.py) | `contrast_vector`: per-layer mean difference of two sets of answers, written as llama.cpp per-layer increments (GGUF) + placebo. |
| [`engine/sealed_set.py`](engine/sealed_set.py) | Sealed confirmation set (GSM8K 100 + MATH-500 50) with a sha256 check; rule in [`lab/probes/sealed/`](lab/probes/sealed/README.md). |
| [`engine/ternary.py`](engine/ternary.py) | Ternarisation (ggml TQ2_0, BitNet absmean) and what survives of an edit under it; WSP-Lang op `ternary_survival` (report only; results §45.23). |
| [`engine/column_edit.py`](engine/column_edit.py) | `edit_column` / `edit_neurons`: one matrix, one or a few columns, structural same-norm placebo on the same support, `budget_params` guard, report, manifest, patches, program for the Q6_K GGUF; reproduces the Exp 8 edit byte for byte (0.8). |
| [`engine/wsplpatch.py`](engine/wsplpatch.py) | `.weightpatch.json` v1.0 and `PatchManager`: `apply_patch` (base and delta hashes verified, new dir), `rollback_last`, `list_patches`, `commit_manifest`, byte-exact reconstruction from base + patches (0.8). |
| [`engine/preserve_bench.py`](engine/preserve_bench.py) | Preservation battery beside a sealed run: HumanEval 164, 40 local instruction-compliance checks, garbage / loop rates, paired compare with a `preserves` flag (plan `lab/probes/preserve/PLAN.md`). |
| [`engine/lora_export.py`](engine/lora_export.py) | `delta_to_lora`: exact SVD of each matrix's change, PEFT adapter + `energy.json` (share of the change kept at each rank). |
| [`demos/math_transplant/`](demos/math_transplant/) | The main result as a runnable demo (adapter off vs on, same server). |
| [`lab/probes/close_probe/`](lab/probes/close_probe/) | Base vs improved Qwen3.5-9B: fast eval (`fast_eval.py`), reward (`reward.py`), behaviour distillation (`distill_gen.py`, `distill_train.py`), closed-form fit (`cf_distill.py`), placebo adapters (`placebo_lora.py`). |
| [`engine/model_report.py`](engine/model_report.py) | "What did this model change vs its base": change class, layers, modules, rank, plus config/tokenizer/template diff. |
| [`lab/probes/delta_patterns/`](lab/probes/delta_patterns/) | Study of what post-training changes in Qwen3.5-9B derivatives: pre-registered plan, findings F1–F26. |
| [`lab/probes/model_compare/`](lab/probes/model_compare/) | Benchmark protocol (GSM8K + MATH-500, two independent graders). |
| [`engine/nanowsp_runtime.py`](engine/nanowsp_runtime.py) | NanoWSP micro-Transformer (d=64, 80,704 params). |
| [`engine/wsp_mcp_server.py`](engine/wsp_mcp_server.py) | MCP server exposing `peek`, `poke` and WSP-Lang to AI agents. |
| [`challenges/`](challenges/) | WSP challenge suite: 8 levels, 43 challenges, mechanical judges. |
| [`edits/`](edits/) | JSON edit manifests (e.g. `evil_mimo.json`). |
| [`FOUNDATIONS.md`](FOUNDATIONS.md) | The scientific basis, by claim: what we hold, its status, our evidence (results §), outside evidence, what would refute it; refuted ideas not to retry. |
| [`PROTOCOL.md`](PROTOCOL.md) | Protocol: sealed holdouts, placebo controls, non-degradation criteria. |
| [`results.md`](results.md) · [`roadmap.md`](roadmap.md) · [`CHANGELOG.md`](CHANGELOG.md) | Results log, plan, patch history (P0–P12). |
| [`tests/`](tests/) | 50 test files (`python -m unittest discover -s tests`). |

---

## 🛡️ Rules we hold ourselves to

1. **Failures stay documented.** E1–E5, the nanoGPT loop fix, the MiMo loop fix, and the withdrawn results in the
   table above are all kept.
2. **Causal claims need a placebo.** A random-direction edit of the same norm is required by `PROTOCOL.md`. It was
   used in Exp 8 and in the loop fixes. The older experiments without one are marked ⚠️ above.
3. **Holdouts are sealed** before an edit is designed. Evaluation uses GSM8K and MATH-500 subsets; since 2026-11-09 a
   hashed confirmation set, disjoint from every item used to pick doses or donors, is run once on anything that passes.
4. **Edits are fenced.** `FORBIDDEN_TENSORS` (in `wsp.py` and `engine/wsplang.py`) blocks embeddings, the output head
   and norms unless explicitly allowed.
5. **Numbers need two methods.** A figure is reported only if two independent computations agree. Example: the
   weight diffs are checked against a second, block-sampled method.

---

## 🎭 Behavioural steering ("Evil Qwen")

Viral posts showed that steering one or two neurons can make a model act like a villain. We tried the same idea
without forward passes:
- **The search:** layer 14 `ffn_down` columns of MiMo-9B were ranked by their projection on lexical centroids (*anger*,
  politeness).
- **The edit:** 4 columns were scaled (`edits/evil_mimo.json`).
- **Why it is not expected to work:** the "Evil Qwen" effect *adds* a fixed amount to a neuron's activation at every
  token, far beyond its natural range (dahara1's analysis on Qwen3.5-4B: about 20× the dataset maximum; scaling the
  same neurons ×20 or ×100 leaves the replies practically unchanged, because they barely fire). Qwen3.5's `down_proj`
  has no bias, so that addition cannot be written into the weights; it needs a control vector
  ([dahara1/EvilQwen-control-vectors](https://huggingface.co/dahara1/EvilQwen-control-vectors), llama.cpp). Scaling
  columns, as `evil_mimo.json` does, is the multiplicative kind. We have not tested either on MiMo. The additive
  kind moves a mood here (happiness control vector, results §43.13), but with the corrected op it costs coherence and
  GSM8K and fails its pre-registered test (§45.3).
- **Not done:** the behavioural effect was **never evaluated**, so this is an untested edit, not a result. Notes:
  [`docs/behavioral_steering_and_dreaming.md`](docs/behavioral_steering_and_dreaming.md).

In the related project [`FiveTechSoft/dreaming`](https://github.com/FiveTechSoft/dreaming), steering through
`token_embd` alone did not change behaviour. That evidence lives in that repo, not here.

---

## 🌐 Community reports and what we checked

| Source | Claim | What this repo checked |
| :--- | :--- | :--- |
| [Simon Willison](https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/) | Qwen 27B identifies Atari games from raw MLP weights. | Prompt rebuilt ([`data/exp1_pong_prompt.txt`](data/exp1_pong_prompt.txt), 17,253 parameters). Our own check is one MiMo-9B query on summary statistics (see the table above). |
| [Jarrod Watts](https://x.com/jarrodwatts/status/2117808859691811216) | Astra's 0.016 % edit (≈131k weights in two layers) fixes Qwen3.5-0.8B loops, MATH-500 7.9 % → 48.1 %. | Bit-exact replication of the edited tensors (130,188 entries actually differ). MATH-500 holdout n=20: 1/20 → 14/20, placebo 0/20. |
| [Jacob Lauritzen](https://x.com/jacsebl/status/2117606206764175543) | The edit removes an obstacle (loops) and adds no knowledge. | Consistent with the Exp 8 holdout: correct answers appear once generation terminates. Not tested separately. |
| [areu01or00/qwen0.8B-wsp](https://github.com/areu01or00/qwen0.8B-wsp) | A fact written into Qwen3.5-0.8B's `down_proj` in closed form (MEMIT-style), 2/5 new phrasings. | Its own history reports 0/5 once the weights are saved in bf16. We used the same closed-form write for a behaviour on 9B (row *Copying MiMo's behaviour*): it transfers some, below gradient distillation. |
| [khordoo/dusty-wsp](https://github.com/khordoo/dusty-wsp) | An 8M model renamed by a rank-2 edit; a fear→curiosity edit fails (0/48). | Read, not run. Consistent with ours: a fact has a target, a disposition does not. We took its placebo control for our adapters. |
| [Ayush (@stochi0)](https://x.com/stochi0/status/2117681387688071185), [Paweł Huryn](https://x.com/PawelHuryn/status/2117639929463197885) | Low-rank RL policies; weight edits as a debugging primitive. | Discussed, **not verified** here. |

We also audit viral WSP claims (results that rest on compiled rules, narrowed tasks or unverifiable artefacts). The
recurring failure patterns are summarized in `lab/probes/delta_patterns/findings.md` (F24–F26) and `results.md`.

---

## 📚 References

How these sources bear on our claims: [`FOUNDATIONS.md`](FOUNDATIONS.md).


- Takehi, Teoh, Zhang, Khattab (MIT CSAIL), [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), blog post, 2 Nov 2026.
- Experiment logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). Edited Exp 8 model: [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
- Demo: [akhaliq/weight-space-programming-demo](https://huggingface.co/spaces/akhaliq/weight-space-programming-demo).
- Main target model: [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B), a Qwen3.5-9B derivative (32 layers: 8 full-attention, 24 Gated DeltaNet).
- Bennington, Kaelar, Lee, Edwards, Glick, Danks (Concept Country), [*Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B*](https://concept.country/research/weight-space-programming), working paper v1, 8 Nov 2026 ([post](https://x.com/varien/status/2119880941711827025), [code](https://github.com/conceptcountry/weight-space-programming), [weights](https://huggingface.co/ConceptCountry/qwen-concept-27b)). A direction read from the model's own traces (last tokens before `</think>` minus the rest), written as rank-1 terms into 9 `down_proj` matrices: GPQA Diamond +4.3, LiveCodeBench v6 +13.0, 27-54 % fewer thinking tokens, a norm-matched random edit does nothing. We are replicating it donor-free on Qwen3.5-9B ([`lab/probes/cc/PLAN.md`](lab/probes/cc/PLAN.md)).
- Hu, Tang, Li, Zhu, Feng, Zhang, Chen, Hu, [*Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations*](https://arxiv.org/abs/2611.09382), arXiv 2611.09382, 6 Nov 2026 ([code](https://github.com/WSP-Scope/WSP-Scope-Project)). 16 agent editing searches on Qwen3.5-0.8B and 9B: one `down_proj` column lifts 0.8B MATH-500 8.4 → 37.8 (a loop fix; random direction of equal norm 10/48 vs edit 20/48 on dev); on 9B gains are small (IFEval +1.9, SciBench +3.0) and a MATH dev gain (+29.7) vanished held-out (base already 97.7 at 32k tokens).
- Lu et al., [*The Surprising Effectiveness of Deleting Weights in LLM Post-Training*](https://arxiv.org/abs/2611.10866), arXiv 2611.10866, NeurIPS 2026 ([project](https://agenticlearning.ai/deleting-weights/)). A learned binary mask zeroing < 0.05 % of `gate_proj` weights matches full fine-tuning (GRPO, SFT, on-policy distillation) at ≥ 7B, with little forgetting. Read as support for "post-training on these benchmarks needs little information" (LoRA r1 matches too); trained with gradients, no same-fraction random-deletion placebo.
- Zheng et al., [*Correctness vs Efficiency: Behavior Control with Weight-Space Extrapolation in Code RL*](https://arxiv.org/abs/2605.28751), arXiv 2605.28751, NeurIPS 2026 ([thread](https://x.com/KunhaoZ/status/2120172863734849903)). The weight difference between two RL runs that differ only in reward strictness steers reward hacking; extrapolating past the strict run cuts resource-limit failures 14.5 → 8.0 %, too far collapses the model. Same narrow push window as our scale grids; no same-norm placebo in the thread.
- Nathan Barry, [*Can a Linear Program Do Weight-Space Programming?*](https://nathan.rs/posts/weight-space-lp/) ([code](https://github.com/nathanrs/weight-space-lp)), Nov 2026. Finds the MIT Exp 8 loop-breaking edit with a sparse linear program in minutes instead of a search; a constant steering push gives almost the same result.
- Matthew Gribben ([HypaNet](https://github.com/Hypabolic/HypaNet)), [*Building AI Models Without Gradient Descent*](https://matthewgribben.com/blog/building-ai-models-without-gradient-descent) and [*Can LLM Hand-Written Neural Networks Be Useful?*](https://matthewgribben.com/blog/written-neural-nets-functional), 8-9 Nov 2026. Fable writes every weight of a 2.1M transformer from a spec (0.957 sealed / 0.908 OOD vs 0.923/0.488 trained on 223,930 labels), and the astral first known rank-1 neuron hand-edit of Qwen3.5-0.8B (+0.196 over base — thin ceiling, agreeing with our E-series verdict).
- Han, Wang, Zhao, Zhang, Li, Borth, Yu, Maron, Ye, Yin, Neri, [*A Survey of Weight Space Learning: Understanding, Representation, and Generation*](https://arxiv.org/abs/2603.10090), arXiv 2603.10090, Mar 2026 ([resource list](https://github.com/Zehong-Wang/Awesome-Weight-Space-Learning)). First unified taxonomy (WSU/WSR/WSG); its "Model Editing" section still has only image hypernets — the LLM-side WSP work in this list is not indexed (yet).
- Matt Winters (Beaker), [*Weight-space programming*](https://withbeaker.ai/blog/weight-space-programming), 11 Nov 2026. Agent loop on Qwen3.5-0.8B scored on 256 training items; best single edit is one neuron (573, output column ×2): MATH-500 7.4 → 37.4 %, and forbidding the token "Wait" without touching weights gives 37.6 %. Dose ×2-×8 all > 34 %. No same-norm placebo.
- shawnghu, [*Claude Fable 5.2 can do weight-space programming about as well as GPT-6 Astra*](https://www.lesswrong.com/posts/wg3K6xFFTdM6RmRgn/claude-fable-5-2-can-do-weight-space-programming-about-as), LessWrong, 10 Nov 2026 ([raw outputs](https://huggingface.co/datasets/shawnghu/fable-weight-space-programming)). MIT's eight experiments repeated with a Fable agent; the loop fix is again a few "Wait" neurons plus a push to `</think>` (MATH 8 → 60, MMLU-Pro / GPQA 40 → 42-45). The author states he did not review outputs or method closely; no placebo.
- Mehta & Singh (Microsoft Research), *Interpretable weight-space programming builds transparent transformers and brain encoding models* (iWSP), Nov 2026 ([code](https://github.com/csinva/weight-space-programming-autoresearch), [thread](https://x.com/y_mehtu/status/2120928488219672741), [post](https://x.com/csinva/status/2120928920073646233)). An agent writes weights from scratch (30-50 propose-write-test rounds): 13 algorithmic tasks at 100 % on a hidden set and again when rebuilt from a ≤ 200-word description, with a median of 166 parameters for Opus 5.5 against 41,600 for Sonnet 5.5 and 100,000 for Haiku 5.5 (the writer matters; relayed by [DAIR.AI](https://x.com/dair_ai/status/2122370369097277634), 16 Nov 2026); TREC 87.3 %; early visual cortex 0.443 vs 0.400 (ties AlexNet); a story-fMRI gain was a position confound caught by the legibility test. Our reading: synthesis works for closed algorithms and interpretable features, not for what must be learned from data (`FOUNDATIONS.md` T4).
- Naik, Liu & Gangal, [*Auto Weight-Space Programming (Part 1)*](https://atharva-naik.github.io/blog/awsp-part-1/), 13 Nov 2026 ([alphaXiv copy, 15 Nov 2026, for versioning](https://www.alphaxiv.org/abs/2611.auto-weight-space-programming-part-1-defining-the-problem-and-initial-experiments), [code](https://github.com/atharva-naik/auto-weight-space-programming), [adapters](https://huggingface.co/atharva-naik-1/awsp-readers), [thread](https://x.com/AtharvaRNaik/status/2121366720518762611), [Gangal's thread](https://x.com/VarunGangal/status/2121771960300634620)). The reading side: a LoRA teaches Qwen3.5-4B the logit lens of 128-number slices of its own neurons pasted as digits (99.8 % four-way on held-out neurons); writing weights, predicting activations and circuits, and reading a neuron's effect on sycophancy from its weights all fail in their setup. Our reading: the tool is the shortcut and the behavioural experiment is the oracle, as in `FOUNDATIONS.md` T6/T7; the *auto* part is not yet shown to matter (reading another model's weights works as well).
- rizumita / CAPH TECH, [*Qwen3.8 Flash Next: Chinese mixed into Japanese answers removed by rewriting the output layer, 27 % → 0 %*](https://zenn.dev/caphtech/articles/qwen38-zh-ban), 13 Nov 2026 ([tool](https://github.com/CAPHTECH/kestrom), [thread](https://x.com/rizumita/status/2121229852317798538)). 25,451 `lm_head` rows of Chinese tokens overwritten with an unused row; byte diff and greedy output identity where the ban does not fire. A hard constraint in the weights with the right control; answer correctness before/after is not reported (`FOUNDATIONS.md` reading list).
- emeraldant, [*WSP-Gym*](https://emeraldant.github.io/WSP-Gym), 14 Nov 2026 ([code](https://github.com/emeraldant/WSP-Gym), [thread](https://x.com/emeraldant/status/2121378773593457031)). Inkling 2 Small post-trained (LoRA + RL via Tinker) on 57,600 synthetic small networks with one planted fault repairs 83 % of held-out ones from raw weights in ~40 tokens without reasoning; unseen fault types 20 %, weight-statistics heuristic 4 %. Our reading: a trained writer for closed toy networks; the unseen-fault row is the same wall as every writing attempt so far (`FOUNDATIONS.md` T4).
- Naik, [WSP benchmark proposal](https://x.com/AtharvaRNaik/status/2121629639458324625), 14 Nov 2026: a suite of diverse tasks with "auto-research agents equipped with interp tools" as the baseline to beat; quotes jfonsecarivera's [sparse-rewiring emergent misalignment](https://x.com/jfonsecarivera/status/2121297068841132448) (13 Nov 2026; method by Sloef): 47 of Qwen's 442,368 neurons retrained on when they fire give misaligned answers 0.3 → 14.5 %, the neurons promote "risky / exploit / brutal" and fire on the framing word of a request. Trained, not WSP; the same bar we use (beat the gradient route on behaviour) and a case for the implanted-neuron check (`FOUNDATIONS.md` T8).
- Zeng, Severi, Bullwinkel, Horvitz & Papailiopoulos (Microsoft Research), [*Lazare, veni foras: Reprogramming Dormant Neurons to Backdoor AI Agents*](https://arxiv.org/abs/2611.15118), 11 Nov 2026. Closed-form edit of rarely-firing neurons in the **final** feed-forward block so that each promotes one payload token when a trigger string is in context: 7 of 12,000+ neurons in Nemotron-Terminal-8B, fitted on 23 trajectories, fire the payload on ≥ 98.8 % of triggered test trajectories with no leakage seen on benign ones; Qwen3.6-27B and Gemma-4-31B reported equally vulnerable; agent-built detectors without a reference model flag 91.4 % of edited checkpoints at 9.7 % false positives. The idea is credited to Claude Code (Opus 4.5, Feb 2026). Our reading: the adversarial twin of our implanted-neuron check (`FOUNDATIONS.md` T8, results §43.24), which without a base does not see the last layer — exactly where Lazarus writes.
- Tensor-Slayer (areu01or00, same author as the `qwen0.8B-wsp` fact write above), [*Reprogramming dormant neurons: Qwen TTS*](https://x.com/TensorSlay/status/2121598038821187935), 14 Nov 2026 ([neuron map](https://github.com/areu01or00/qwen-tts-neuron-mapping), WIP). An agent (Opus) maps 417 MLP neurons (0.48 %, talker layers 17-27, 304 of them in layer 27; 55-61 "dormant" = firing strongly in < 1 % of speech frames) in Qwen3-TTS-0.6B; a rank-1 push of their `down_proj` columns along a fixed per-layer direction at ×8-12 adds breathing, laughs, whispering and a second speaker to cloned voices; ×16+ turns speech into gibberish; on built-in voices ×10 gets stuck 13/36 times. Listening only: no WER, no quality metric, no random-neuron control. Our reading: the dose window is T3's narrow window again, and without a same-norm placebo it is a demo, not a finding (T6).
- Khamis, Tutunov & Bou Ammar, [*AHA! Who Needs an LLM to Program an LLM?*](https://haithamb.github.io/aha/), 15 Nov 2026 ([code](https://github.com/Eppy-EpHard/AHA), MIT licence, pre-registrations in `docs/`; [thread](https://x.com/KickItLikeShika/status/2122075847205282054)). MIT Exp 8 without the agent: the direction *u* comes from matched pairs the model makes itself (rewind a looping trace 8 tokens before loop onset, sample 8 continuations, pair the earliest loop with the longest productive branch, difference of residual means over tokens 64-224; 228 pairs from 300 prefixes), *v* from Barry's linear program (8 columns per layer, ~16k weights, 0.002 %), layers by held-out LP hit rate. Qwen3.5-0.8B at 32k tokens: GSM8K 14.0 → 71.2, MATH-500 9.8 → 51.8, IFEval (never seen, no maths) 19.6 → 62.7; 2B: 75.8 → 88.4, 69.6 → 79.6, 61.4 → 77.1. Build ≈ 22 min on one RTX 4090. Ablations: a better-separating (Fisher) direction edits far worse (AUC 0.953 vs 0.895; dev 21.9 vs 64.1); per-token LP targets, ε 0.25 or 64 columns all hurt; layer choice moves < 1.6 points; MedQA under 2 SE. States its ceiling: gain ≤ base truncation rate, "removes an obstacle and adds nothing", the unedited 2B beats every edited 0.8B. No same-norm placebo of its own (it cites Barry's random direction). Our reading: the same primitive as our `compile_control_vector` (rank-1 in `down_proj`, no gradients) with the push made conditional by the LP where ours is constant by ridge; their tighter-conditioning ablations failing agrees with Barry and with our constant push (`FOUNDATIONS.md` T2, T3).
- Alapan Chaudhuri, [*A model's weights keep a record of its own training*](https://x.com/alapanchaudhuri/status/2121439269772492848), X thread, 14 Nov 2026 (no paper or code yet; relayed by [zainhas](https://x.com/zainhas/status/2121885291556520031)). Reading, not writing: AdamW's footprint in the input-embedding matrix. A token never in the data gets no gradient and its row does not move (bit for bit without weight decay); a token seen in one step moves by about 31.8 × the learning rate of that step in nearly every coordinate, since Adam with no past normalises the gradient to its sign, so the row's shift is a timestamp; a row that is never touched still shrinks by (1 − lr·wd) per step, exact in fp32, so a schedule is replayed bit for bit. Claims: the token list exact (39,663 rows moved, 39,663 tokens in the data; 50,180 / 50,180 on a Qwen2.5-7B fine-tune), schedule type, step count and warm-up in 15 of 15 fine-tunes, peak LR within 0.01 %, 3,121 of 4,000 documents placed in their exact step, 38 of 40 swapped documents exposed, a stranger's public fine-tune replayed on 25,463,296 weights bit for bit (one step off: only the 187 zeros match), 64 % membership at zero false alarms where the standard test gets 0.5 %; works with several epochs, large batches (OLMo 2 LR within 2 %), fp16 releases, L2-in-gradient Adam and LoRA that trains the embeddings; open: tied embeddings, LoRA on attention only. On public models: 20,341 OLMo 2 SFT rows never moved (the tokenizer mismatch AI2 later fixed); Qwen2.5-Instruct touched 4 of 22 control tokens at every size; Pythia-1B trained at 2.5e-4, not the card's 3.0e-4 — we checked that one: the training config says `lr 0.00025, min_lr 0.000025, cosine`, the card table says 3.0 × 10⁻⁴. Our reading: the strongest weights-only read we know of (`FOUNDATIONS.md` T7), with one base-free trick (rows that belong to no token only shrink, so the public checkpoints alone give lr·wd per step) that our implant check lacks; and our own delta study filed the top-changed embedding rows of MiMo as "rare junk tokens: noise" (`lab/probes/delta_patterns/h5_plan.md`) — by this mechanism they are the timestamps. Thread-only numbers, not checkable until code appears.
- Kocabay (Poindexter Labs), [*Weight-Space Reading*](https://www.alphaxiv.org/abs/2611.weight-space-reading-language-models), alphaXiv, 13 Nov 2026 ([code](https://github.com/suayptalha/wsr), Apache 2.0; [thread](https://x.com/askalphaxiv/status/2122270200410357981); arXiv pending). The reading half measured properly: sandboxed LLM readers see a target's weights and a surgical edit (block scaling, vocabulary rows, direction removal, matched noise, norm gains, no-op controls) and predict up / down / same for five behaviours (thinking length, maths, multiple choice, sycophancy, refusal); ground truth is the edited model run against five base seeds (Welch p < 0.01 plus a minimum effect). Three pre-registered rounds on Qwen3.8-27B, Qwen3.5-9B, Muse-Glimmer-30B. Result: on subtle, non-destructive edits 27B reading its own weights reaches balanced accuracy 0.730 against 0.578 from a text description of the same edit (+0.152, CI [+0.056, +0.256]), change recall 0.278 → 0.667 but specificity 0.878 → 0.793; the gain is significant in one reader-target pair out of four, from 10 changed edits. Qualifiers the authors put first: no conclusive self-reading advantage (crossed contrast spans zero in both rounds; 27B also reads 9B better); full-dose benchmarks measure damage detection (Muse-30B reads destructive edits to 27B as well as 27B itself, 0.594 vs 0.593); and none of 40 edits (34 standard plus 6 no-op controls) changed Muse-30B's behaviour, partly because its post-sublayer RMSNorm cancels uniform scaling, yet readers predicted a change for 53-88 % of them, Muse-30B itself for 85 %. Single run per cell, uncorrected comparisons, intervals called optimistic by the authors. Our reading: the cleanest outside statement of `FOUNDATIONS.md` T6/T7 — reading weights gives a modest, measurable signal only when scored against guessing, inert edits and architecture, and the behavioural run stays the oracle; the inert-edit false alarms are the reader-side version of our same-norm placebo, and the Muse case is our §45.3 lesson (the norm row) at architecture scale.
- Marasini, Khadka & Rahimi, [*WeightBench: Evaluating Language Models as Weight-Space Programmers*](https://arxiv.org/abs/2611.18553), arXiv 2611.18553, 13 Nov 2026 ([code and task registry](https://github.com/Aayush-Marasini/WeightBench)). 42 tasks in five tracks (reading, fixing and writing weights, recognising impossible edits, data-free compression); abstract scores: Opus 5.5 macro 0.535 on all 42 (Sonnet 5.5 0.466, Haiku 5.5 0.402, DeepSeek v4-flash 0.215); GPT-6.1-sol 0.568 against Opus 0.546 on the 38 tasks it ran. Read from the abstract only (pointer from the Phi synthesis in `docs/`); our compile sits nearest its FIX track, not WRITE (synthesis from scratch), and nothing here has been run under it.
- Ok & Lee (POSTECH), [*Weight-Space Programming Doesn't Need Transformers*](https://www.alphaxiv.org/abs/2611.weight-space-programming-associative-memory), alphaXiv, 17 Nov 2026 ([code and 56 written models](https://github.com/effl-lab/Weight-Space-Programming), MIT; [project page](https://wsp-no-transformer.github.io); [thread](https://x.com/HyunjongOk/status/2122518170359706090)). The from-scratch branch: Opus 5.5, GPT-6.1 Sol and GPT-6 Astra, given ten labelled ARC questions and a sandbox, either write the weights of a released architecture or design their own; all three design a sparse table of plain-text facts read by fixed code (54.4 % on 3,530 ARC test items vs 43.1 % for the best written released architecture and 36.1 % for a written two-layer Transformer; their associative-memory table 52.4 %, 17× faster on GPU). Written Transformers score below BM25 over the same text in 20 of 23 runs, the best table beats BM25 by 3.7 points, a fine-tuned pretrained Qwen3.5-0.8B gets 82.1 %, and a writer starting from a pretrained checkpoint changes under 1 % of the weights (readout, embeddings). Our reading: `FOUNDATIONS.md` T4 from outside — writing from scratch carries the writer's explicit knowledge and nothing learned from data; does not edit a pretrained LLM, so it leaves T2/T3/T8 untouched.
- alphaXiv / Daniel Kim, [*minimizing-wsp-edits*](https://openresearch.sh/p/EdsyLoFUfY40), OpenResearch project, 17 Nov 2026 ([notebook](https://github.com/alphaXiv/weight-edit-minimization), marimo, runs in the browser; [thread](https://x.com/askalphaxiv/status/2122744324354097172)). An agent (Inkling 2 Small inside OpenResearch) shrinks two of the MIT WSP blog edits. Exp 8 (Qwen3.5-0.8B, `down_proj` of layers 15 and 19, Astra's 130,188 entries): the notebook recovers the edit's closed form, one direction *u* subtracted through 64 neurons per layer (W = W₀ − 6·u·(m ⊙ uᵀW₀)), then keeps the 4,096 largest entries per layer (8,192, 6 %): 58.2 % vs 58.6 % for the full edit on a 256-problem MATH-train screen (one sample each, T 0.6, 32k cap, "differences under ~5 pp are noise"), 54.2 % (1,084/2,000) on MATH-500 × 4 held out, set against the blog's 48.1 % for the full edit under other settings; side effects checked only as cut-at-32k rate (base 91 %, edit 30 %) and a 16-gram loop share. Car-ramp PPO actor (Kinetix): Astra's 6,073 entries → 55 functional plus 1,884 embedding erasures → 12 → 11 by drop-one audit, and a 10-entry fix written from scratch; every rung solves the level (256-1,024 episodes). Our reading: we checked the first claim on our own recovery of the same edit (`data/exp8_edit_recovered.npz`, results §9.1 and §16.2 correction): the edit **is rank-1 per layer** (s₁/s₂ 1,771 and 1,552; every column collinear with *u*; cos between the two layers' directions 0.85) and its 4,096 largest entries per layer hold 74-76 % of ‖ΔW‖², so the pruning keeps most of the push; the "no degradation" is a screen at ±3 pp SE with the held-out row compared against a number from a different evaluation (`FOUNDATIONS.md` T2, T3; our rigor checklist). It confirms that the blog's edit belongs to the same family as our compile (`FOUNDATIONS.md` T8: rank-1 in `down_proj`, few neurons) and that the magnitude of the push, not the count of entries, is what matters.
- laxman / bandr.ai, [*DAWE: Data-Attributed Weight-Editing*](https://bandr.ai/write-up/dawe/), write-up and [code](https://github.com/bandr-ai/dawe) (Apache 2.0; research checkpoint not published), 17 Nov 2026 ([thread](https://x.com/llmluthor/status/2122736693401080040), quoted by Rikiya Takehi). Qwen3.5-4B LoRA-SFT on 595 BFCL items under harness v1 breaks when the reply format changes (plan-first 20.1 %, XML 73.5 %, v1 87.7 %); an agent (Astra / Fable) reads the failing stop-or-continue decisions and fits a ridge regression on final-layer hidden states to the end-of-turn logit (δ = (XᵀX + λI)⁻¹Xᵀy, targets −12 where another call should follow, 0 at real stops), then adds δ to the end-of-turn row of `lm_head` (plus the `<` row for plan-first): 2,557 and 5,064 values = one or two rows of d 2,560. Plan-first 20.1 → 88.2 % (retrain from base 88.7), XML 73.5 → 85.8 (89.7), v1 unchanged at 87.7; a same-norm random edit of the same row gives 68.1 %. 204 held-out items, greedy, single runs, their own scorer (± 5 pp, stated); no constant-logit-bias control (stated: the state-dependent part is unproven); tied embeddings so the edit also touches input rows; aggregate test scores were visible to the agent. Our reading: a closed-form readout edit on one token's row, the "stop" component alone (the one our port carried in §45.12), and the headline "killed retraining" is a reply-format repair on one checkpoint, as the write-up itself says.
- rizumita / CAPH TECH, [*kestrom inspect: reading backdoors from the fine-tune difference*](https://zenn.dev/caphtech/articles/kestrom-inspect-backdoor-weights), 17 Nov 2026 ([thread](https://x.com/rizumita/status/2122474419100749900); tool [`CAPHTECH/kestrom`](https://github.com/CAPHTECH/kestrom), `inspect` under PolyForm, MLX / Mac only). Needs the official base: per-tensor SVD of the difference (`shape`: LoRA leading singular values ≈ 1-2.5, implanted components 5.85 and 25.1), token rises on ≈ 1,000 plain texts through both models (`behavior`), a flagged component added to the residual stream and read out as tokens (`lens`), trigger search, and a GGUF path that requantises the base the same way. Pre-registered protocol, planter agent separate from the reader, 12 fine-tunes × 4 tests of Qwen3.5-0.8B / 4B (LoRA r 8-16; backdoors mixed into training data or implanted after as rank-1/2 writes): 0 / 32 clean models flagged, 15 / 16 backdoors found, 14 / 16 targets, 4 / 16 triggers; the miss is a 4-bit-base backdoor requantised to Q4_K_M (rounding buried the low-rank change); F18's implanted backdoors were always-on by a planting error. Limits they state: 0 / 8 per test bounds false positives only at 31 %, one backdoor per model, no full fine-tune, no evasion, LLM reader (Opus 5.5; Fable 5.1 as second). Same conclusion as our `implant_detect` (§43.24): with the base, a rank-1 write stands out; without it, only token-writing shows; and a real fine-tune moves a few legitimate neurons "a lot", which they handle by reading all 12 models together.
- oguzeldereli, [*ttf*](https://github.com/oguzeldereli/ttf-home) (ternary {−1, 0, +1} runtime, llama.cpp patches, conversion and recovery training; PolyForm Noncommercial; weights [`ttf-Qwen3.8-27B-GGUF`](https://huggingface.co/oguzeldereli/ttf-Qwen3.8-27B-GGUF)), Nov 2026 ([relayed by swarogan](https://x.com/swarogan/status/2122681494019649868)). Qwen3.8-27B converted to 1.6-bit TQ1_P matrices with Q4_K embedding and head, then **72M tokens of recovery training against a cached teacher**: 8.20 GB (ternary adapters) and 6.74 GB (none; needs their llama.cpp patches), GSM8K 0.895 = full precision on the first 200 problems, HumanEval 0.841 vs 0.933, IFEval 0.735 vs 0.787; Flash-Next TriviaQA 0.538 vs 0.776. 73-102 tok/s on a 3090 Ti. What it is for us: a 27B reasoning model that fits the 12 GB card, so the "conclude" protocol (§43.17) could be re-derived at 27B as a control vector (activation space; no weight write needed). What it is not: a base for weight reading or editing — the matrices are ternary and recovery-trained, so differences against the official Qwen3.8-27B are quantisation plus 72M tokens of training (Kestrom's own miss above is the same effect), and our rank-1 `down_proj` writes have no float target except the optional adapters. "Identical to FP16" is GSM8K on 200 items, single run, after training; the code and knowledge losses are the part of the result that matters for weight-space reading (what survives 1.6 bits is coarse structure).
- Salloum & Ali, [*Tree-Ising: Repairing Language Models with Bit-Level Ising Edits*](https://arxiv.org/abs/2611.14210), arXiv 2611.14210, 10 Nov 2026 (abstract only, 18 Nov 2026). The MIT loop repair on Qwen3.5-0.8B redone without an agent: admissible edits are BF16 sign flips and exponent shifts, the objective (weaken a loop direction, strengthen a concluding direction, preserve the layer's output on productive tokens; forward-pass statistics) decomposes into one QUBO per output row, solved by simulated annealing or simulated bifurcation on a GPU; four forward-pass requests, about $0.90. MATH-500 base 7.9 % → Astra's edit 51.7 % → Tree-Ising 54.6 / 55.1 %, 13-17 % fewer weights touched, 2.2× fewer bit flips; Qwen3.5-2B GSM8K 21.6 → 69.4 % with layers and dose chosen on the base alone. No placebo or held-out protocol in the abstract. For us: the third closed-form derivation of the same edit after AHA's LP and our ridge compile (T8), now in the storage format's own alphabet; the per-row decomposition is what our ternary check (§45.23) says a discrete writer needs, and a same-norm random-direction control is the missing arm.
- Six arXiv items of 5-16 Nov 2026, read from their abstracts only (18 Nov 2026; flagged here so the list does not pretend otherwise): Aksoy, [*W-ISA: Transactional Execution of Weight-Space Intervention Programs*](https://arxiv.org/abs/2611.13232) (an instruction set with shape-checked updates, projections, ablations and rank-1 changes, request-scoped with automatic restore; SmolLM-135M and Qwen2.5-1.5B; the runtime shape WSP-Lang's write guards and manifests approximate from the other side) · Li et al., [*WeightLoop*](https://arxiv.org/abs/2611.07319) (a WSP editor that turns experimental measurements into persistent updates of scientific predictors; 5-19 % lower error; not LLM weights) · Wang, Lu & Li, [*Can LLMs Write Backdoors?*](https://arxiv.org/abs/2611.17395) (five LLMs asked to backdoor an MNIST MLP, ResNet-18 and Qwen2.5-0.5B from instructions, no data or gradients: six parameters suffice on the MLP, 0.02 % on Qwen; only GPT-6 Astra succeeds on all three; the attacker side of our `implant_detect`, §43.24) · Sato, [*WSP Changes How an LLM Uses Python for Computational Biology*](https://arxiv.org/abs/2611.20566) (Opus 5.5 edits Qwen3.5-0.8B; 50 problems × 3 seeds paired: strictly correct 44 → 77, repeated Python calls 1,834 → 1,044; of 64 pairs won, the original had already computed the right value in 36, the same "answer was there, did not close" pattern as our T2) · Ahmed, [*Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids?*](https://arxiv.org/abs/2611.08070) (125k-parameter CNNs as text: local arithmetic plus dataset priors, label permutation breaks the reading; single runs; our T7 from outside) · Young, [*Is Abliteration Getting Harder? Not Uniformly*](https://arxiv.org/abs/2611.17846) (direction ablation and ARA in Heretic across five release pairs; no uniform trend, the answer depends on pair, method and budget; context for our Heretic forensics, §43.20).
- Background: Irie & Schmidhuber, fast weight programmers (ICML 2021). Belrose et al., LEACE (NeurIPS 2023).

## 📄 License

MIT. See [LICENSE](LICENSE).
