# What the Phi results establish, and what they do not

Evidence synthesis, 16 November 2026. No new experiment is reported here. This note connects
results §45.15-45.17 to the project's claims and to outside work; the experiment log, plans and
per-item outputs remain the primary record.

## Conclusion

The donor-free "conclude" recipe and its closed-form compilation into weights work on the tested
Phi-4-mini-reasoning 3.8B checkpoint, as well as on the previously tested Qwen3.5-9B checkpoint.
On Phi, the compiled Q6_K model improves held-out accuracy at a 1,024-token cap without a runtime
control vector. At 2,048 tokens on **dev items**, much of the apparent improvement disappears as
the base gets time to finish; a GSM8K advantage remains, but a MATH advantage is not established.
This is evidence of transfer across two model families, not a universal property of LLMs, a new
reasoning capability, or a comparison win against another project.

## 1. Measured results

All scores below are **counts correct**, not percentages: GSM8K has 100 items per set and
MATH-500 has 50. They use the first grader; the log reports a second grader and disagreements.
Runs use greedy decoding and Q6_K. "Vector" means an inference-time control vector at scale 0.5;
"compiled" means the edited GGUF served without that control-vector flag. A placebo uses a random
direction of the same norm (one seed), compiled too when the treatment is compiled.

### A. Transfer of the recipe to Phi (§45.15)

| Set, cap 1,024 | Base GSM8K / MATH | Vector GSM8K / MATH | Placebo GSM8K / MATH |
| --- | --- | --- | --- |
| Dev | 59 / 6 | 85 / 19 | 47 / 5 |
| Sealed set 1, first use for Phi | 59 / 4 | 82 / 19 | 58 / 6 |

On sealed set 1 the vector gains/loses 27/4 GSM8K items and 16/1 MATH items against the base.
The direction is derived from the model's own finished traces, not imported from Qwen. The **recipe**
transfers, not a shared vector or a shared dose: Phi's scale 0.5 was chosen on separate screen items;
Qwen's scale 1.0 makes Phi skip reasoning on some items and performs poorly on that screen.

**Inference:** the observed effect is not confined to the one Qwen checkpoint previously tested.
It does not establish transfer to every Phi model, other architectures, or arbitrary behaviours.

### B. Compilation into Phi's weights (§45.16)

| Set, cap 1,024 | Base GSM8K / MATH | Vector GSM8K / MATH | Compiled GSM8K / MATH | Compiled placebo GSM8K / MATH |
| --- | --- | --- | --- | --- |
| Dev | 59 / 6 | 85 / 19 | 81 / 19 | 54 / 5 |
| Sealed set 2, first use for Phi | 62 / 7 | 88 / 15 | 86 / 14 | 60 / 2 |

The same `compile_control_vector` operation writes rank-one edits into 31 `down_proj` layers,
using calibration activations and a closed-form ridge solve, not gradient descent. A device-placement
fix was needed for the solver's CPU fallback; this is not a claim that every implementation line
was unchanged. The held-out calibration diagnostic G0 is 0.044; the median cosine between the intended
write and the realized weight change after Q6_K is 0.74. That cosine is a geometric diagnostic,
**not** 74% behavioural fidelity.

On sealed set 2, compiled versus base gains/loses 26/2 GSM8K items and 7/0 MATH items (exact paired
McNemar p approximately 0.0000030 and 0.016). Compiled versus vector is 2/4 and 2/3: no significant
difference detected, **not a formal equivalence test**. Sealed sets 1 and 2 are different item sets;
the base scores 59/4 and 62/7 must not be compared as if measured on the same items.

**Inference:** a measured behavioural intervention can persist in a normal quantized weight file,
without an extra steering hook at inference. It still needs ordinary inference software. This is
not evidence that calibration, behavioural testing, or model-specific dose selection can be omitted.
HumanEval-20 rises 11 to 16, but n = 20 and a maths system prompt make this exploratory, not a broad
claim of coding improvement or absence of collateral damage.

### C. The token-budget control changes the interpretation (§45.17)

Same dev items, with the cap doubled; this is **not a new sealed confirmation**.

| Cap | Base GSM8K / MATH | Vector GSM8K / MATH | Compiled GSM8K / MATH | Compiled minus base |
| --- | --- | --- | --- | --- |
| 1,024 | 59 / 6 | 85 / 19 | 81 / 19 | +22 / +13 |
| 2,048 | 81 / 26 | 85 / 35 | 91 / 29 | +10 / +3 |

At 2,048, compiled versus base gains/loses 12/2 GSM8K items (p = 0.01294) and 7/4 MATH items
(p = 0.54883). The second grader also supports the GSM8K direction (11/2, p approximately 0.022).
These are paired, nominal p-values for these comparisons, not a guarantee of replication.
There is no placebo arm at 2,048; the random-direction controls belong to the 1,024-token studies.

Of 42 GSM8K items the base truncated at 1,024, it now solves 25 itself. The compiled GSM8K advantage
shrinks by 12 items; the MATH advantage shrinks by 10. Calling this "about half the GSM8K gain and
most of the MATH gain explained by budget" describes this comparison, not a causal decomposition
with an exact percentage. At 2,048 the base still truncates 24/50 MATH items, so MATH is not tested
in an effectively unlimited-budget regime. Compiled mean tokens are 37% lower on GSM8K and 18%
lower on MATH than the base at the same cap. No finished answer in these three arms has fewer than
50 characters of extracted thinking; this proxy does not reveal or certify an internal reasoning process.

**Inference:** a useful efficiency intervention survives a doubled budget on this GSM8K dev set.
It is wrong to say the Phi gain is wholly independent of budget, wholly explained by budget, or that
the compiled MATH improvement beyond the cap has been established. T2 is model- and budget-dependent.

## 2. What this permits in practice

- **Deploy a tested behavioural edit as weights:** the Phi Q6_K file can carry this intervention without
  supplying the vector flag. Preserve the model hash, calibration record, dose and evaluation settings.
- **Reuse a procedure across the two tested families:** derive a direction from each base's own traces,
  choose its scale on screen items, test a matched placebo, compile, then confirm on untouched items.
  Do not copy Phi's scale blindly to another checkpoint.
- **Measure the actual tradeoff:** compare the edited model with a base given more tokens and report
  correct answers, truncation, tokens and collateral tasks together. Shorter output alone is not success.

It does not yet permit synthesizing arbitrary knowledge, replacing training in general, guaranteeing
safety on unrelated tasks, or claiming a benchmark lead. The enlarged-budget Phi result still needs
an untouched-set confirmation; a pre-specified 4,096/8,192-token study could clarify the MATH limit.
These are proposed tests, not runs started by this note.

## 3. Outside work: related evidence, not direct replication

### AHA and T3

Khamis, Tutunov and Bou Ammar's [AHA post](https://haithamb.github.io/aha/) (15 November 2026;
[MIT-licensed code](https://github.com/Eppy-EpHard/AHA)) constructs a direction from the model's own
rewind-and-resample matched pairs, obtains the write with a linear program, and selects layers from
held-out LP hit rate. Our compile instead fits a near-constant push with ridge. These are related
rank-one editing primitives, not identical pipelines.

Their Fisher direction separates looping/productive tokens better (held-out AUC 0.953 versus 0.895),
but edits worse (dev 21.9 versus 64.1). This agrees with [T3](../FOUNDATIONS.md)'s distinction between
separating representations and finding a useful intervention; it does not independently reproduce
our Phi experiment. AHA tests Qwen3.5-0.8B and 2B, one family. The post states its 0.8B held-out numbers
come from its first mixed-penalty direction/build, not its later clean direction. It cites Barry's
random-direction control rather than reporting its own same-norm placebo.

AHA's truncation ceiling is a useful comparison, not a theorem about all edits. In our 2,048-token
Phi run, net gains are below the base's truncation counts (GSM8K +10 versus 17 cuts; MATH +3 versus 24).
But 2 of the compiled model's 12 GSM8K wins are on **finished-wrong**, not truncated, base answers;
the arithmetic bound alone does not prove every recovered answer came from a truncation.

### WeightBench

[WeightBench](https://arxiv.org/abs/2611.18553) (13 November 2026;
[code and task registry](https://github.com/Aayush-Marasini/WeightBench)) reports 42 tasks in five
tracks: READ, FIX, META, WRITE and SYSTEMS. Its abstract reports Opus 5.5 macro score **0.535 on all
42**, and GPT-6.1-sol **0.568 on the 38 tasks it ran**, versus Opus **0.546 on those same 38**.
The 0.568 must not be presented as a score on all 42 tasks.

This repository has not been evaluated under that harness. WeightBench's WRITE means synthesis
from scratch; compiling a behavioural edit into an existing LLM is closer to its FIX question,
not automatically a WRITE result. Its composite scores, edit budgets, probe budgets and collateral
penalties are not our accuracy counts. No claim of outperforming its agents follows from §45.15-45.17.

### Auto-WSP and source limits

[Auto Weight-Space Programming](https://atharva-naik.github.io/blog/awsp/), by Atharva Naik,
Emmy Liu and Varun Gangal, describes Part 1 (13 November 2026) as formalizing WSP and the auto-WSP
goal. Its series page describes a toy self-reading/logit-lens task and leaves weight writing as a
next step. This is context for the field, not evidence that our compiled intervention reproduces
their task or achieves general self-programming.

The Chaudhuri X thread discussed in this repository's reading list was **not read directly for this
synthesis**. Any account of that thread here would be second-hand through the existing repository
entry, so it is not used to support a new empirical claim in this note. External reports above were
read at their primary pages on 16 November 2026; they are author reports, not experiments rerun here.

## 4. Audit trail and remaining uncertainty

- Experiment log: [results.md, §45.15-45.17](../results.md); pre-run plans and labelled post-data notes:
  [Phi PLAN](../lab/probes/cc_phi/PLAN.md); prediction rows 34-36: [predictions.md](../predictions.md).
  These are the project's records, not independent replications.
- Vector: [dev outputs](../lab/runs/cc_phi/), [sealed set 1 outputs](../lab/runs/sealed_cc_phi/).
- Compilation: [op report, dev outputs and verdict](../lab/runs/cc_phi/weights/),
  [sealed set 2 outputs and verdict](../lab/runs/sealed_ccw_phi/).
- Budget control: [three arms and configuration](../lab/runs/cc_phi/cap2048/).
  The JSON files record the served model path/hash and per-item grades. Counts and the 12/2 and 7/4
  exact tests above were checked again from those committed rows for this synthesis; no inference was rerun.
- Prediction outcomes remain partial, not retroactively successful: rows 34, 35 and 36 are marked
  partial; row 35 has 18/22 readings in band and row 36 has 14/18. The repository tally at this snapshot
  is 6 passed / 14 partial / 12 failed / 4 unscored (the table groups pending and not-run entries). It is not a statistical success rate over independent trials.
- Limits: two tested families, selected checkpoints, Q6_K, greedy decoding, one Phi placebo seed,
  screen-selected dose, reused dev references, small collateral sample, llama.cpp's Phi SWA-disabled
  warning, and grader disagreements. Larger-budget sealed tests, more placebo seeds, more models and
  broader collateral evaluations could change the interpretation.

The warranted claim is a **controlled, model-specific behavioural write that transfers as a procedure
across two families and survives quantization**. Its size and usefulness depend on the model, task,
scale and token budget. That is narrower than general weight-space programming, but directly testable.
