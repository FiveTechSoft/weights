# Reusing the MiMo changes on Qwen: status, sources and open tests

Goal: compare Qwen3.5-9B with XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B, see what changed, and find out whether part of that change can be
reused on the base. This file separates what was **executed** from what is only **planned**. Detail and numbers are in
`lab/probes/delta_patterns/findings.md` (F27-F36). Nothing here claims a benefit without a task test.

## Pinned revisions
| Model | Revision |
|---|---|
| Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | 1f09ddcc1a0ea45751ae0dfe2af20d533168c46f |
| Qwen/Qwen3.5-0.8B (planned test, Apache-2.0) | 2fc06364715b967f1860aea9cf38778875588b17 |

## Executed (layers 0-3 of the 9B pair, CPU, 2 GB RAM, streamed ranges)
- F33: 98% of elements differ between the two models; an exact compact reconstruction is ruled out for the 12 of 427 tensors measured.
  This does not rule out a compact functional component.
- F34: a rank-256 patch on layers 0-3 moves the residual stream toward MiMo (resid 0.70 dev / 0.66 held-out, 1.0 = unpatched; random
  control of the same size ~0.95; rank 16: 0.94 / 0.91). The .80 / 3x criterion was chosen after seeing the data (post hoc). Output
  proximity to MiMo is not task benefit: no task test has been run on the 9B patch.
- F35: choosing the rank from activations captures more of the layer-0 down_proj delta than weights-only at the same rank.
- F36 (pre-registered): activation-aware selection on 25 matrices did NOT meet its criterion (rank 16 win rate 72% < 75%; rank 64
  mean gain 0.007 < 0.01). Not adopted as a general rule.
- Tools: engine/stream_score.py, stream_guard.py (byte and memory budgets), delta_pack.py, delta_apply.py, data_aware_*.py.

## External sources read (read-only; none of their code was run or copied)
| Source | What it is | Use for this project |
|---|---|---|
| github.com/areu01or00/qwen0.8B-wsp (TensorSlay post) | closed-form, MEMIT-style fact writes on down_proj layers 17-21 of Qwen 0.8B, no licence | mechanism reference only; facts, not a skill |
| hcltech-robotics.github.io/iwibench, repo HCLTech-robotics/iwibench (no licence) | benchmark for weight patches: fresh process, target gain x control retention, no-op = 0 | evaluation template. Results unverified by us; PDF, RESULTS.md, CSVs not read |
| arxiv.org/abs/2610.22192 Terminal-RSI | SFT + RL self-training loop on Qwen3.8-27B | not a delta method; only abstract and excerpts read |
| arxiv.org/abs/2611.07319 WeightLoop | LLM-written editor, ridge update of 16 readout weights of a 3.6M-parameter biology predictor | same family as our ridge writes; no code found |
| github.com/Fchaubard/WSP-for-turing-complete (no licence chosen) | WSP vs gradient descent on 87 small settings (3-50 coefficients): 30/87 vs 0/87 | claim holds only for those budgets and sizes; the README says it does not show backprop cannot solve them |
| MIT "Sparks of Weight-Space Programming" (HF card RikiyaT/Qwen3.5-0.8B-weight-space-programming; critique honnibal.dev/blog/astra-read-write-misleading) | the WSP framing | context; claims not reproduced |
Social posts were read through public pages without login; most replies and all videos/images were not seen.

## Planned, not executed: closed-form write of one arithmetic skill
`lab/probes/closed_form_math/PREREGISTRATION.md` fixes model, skill (2-digit addition), data sizes, controls, free-generation scoring,
exact-match and identity-retention definitions, criteria and a cost ceiling before any run. The solver `engine/closed_form_write.py`
is independent and tested on synthetic data only. Expectation recorded in advance: it may well fail, because a write fits any
targets (including shuffled ones) and only new-problem accuracy can show a skill.

## Update 2026-11-09: new external sources and change of plan
- github.com/khordoo/dusty-wsp (MIT; author Mahmood Khordoo; read README and commit list only, nothing run): analytic rank-2 edit of one
  256x1024 matrix of an 8M model. Rename Dusty to Milo: identity pattern 13/30 to 21/30 greedy (90% bar not met), 0 paired losses; stairs
  attitude edit: 0/48 held-out, a kept negative result. Regex metric, development-informed ("training-free is not data-free"). Concept
  or capability editing is not shown.
- Verified in the history of github.com/areu01or00/qwen0.8B-wsp (not dusty-wsp): commit 39265db, 2026-11-08, "results: tried checking the
  write on bf16-merged weights; worse on the saved file (2/5 -> 0/5 word-perfect), not merged", with a "Co-Authored-By: Claude Opus 5.5"
  trailer (a trailer is a claim, not proof of who ran what). dusty-wsp credits GPT-6 Astra. The two repos should not be conflated.
  Consequence for us: results must be re-measured on tensors saved and reloaded in the same dtype.
- Plan change: the closed-form answer-writing test (closed_form_math/PREREGISTRATION.md) is superseded by lab/probes/skill_delta/PROTOCOL.md
  (train a small teacher on 2-digit multiplication, then reproduce its delta by surgery, gates fixed in advance). Executed so far: Qwen3.5-0.8B
  downloaded (pinned), free-generation runner measured (100 prompts ~90 s), 75 add2 unpatched calibration generations (54/75), trainer smoke
  tests. No teacher trained, no gate evaluated.
