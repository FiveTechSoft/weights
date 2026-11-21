# Recovering HumanEval for our edits (driver's OK 20 Nov 2026) — plan fixed before the run

**Why.** On `rank-2048` (results §45.36) both our edits fail the 164-item preservation rule: base 147, compiled
conclude 132 (closes code too early: 10 of 21 losses cut, reasoning 1,484 vs 1,852 chars), base + Jackrong ×0.5 117
(reasons past the cap on code: 34 of 37 losses cut). Two cheap levers, no new code:
- **Dose:** the conclude vector at 0.5 and 0.7 (never measured on the 9B below 1.0); less push should close code less.
- **Composition:** Jackrong ×0.5 plus the conclude vector at 1.0 (§45.7 showed it shortens and removes cuts; HumanEval
  at 164 never measured): the vector should give back the transplant's code losses that are cuts.

**Arms (same suite `rank-2048`, same server flags as stage 1, base run of stage 1 as reference):**
| arm | served | request |
|---|---|---|
| `vec05` | A0 + `--control-vector-scaled hf_cache/cc/cc_v2.gguf:0.5` | — |
| `vec07` | A0 + `--control-vector-scaled hf_cache/cc/cc_v2.gguf:0.7` | — |
| `jack05_vec10` | A0 + `--lora demo_jack_r16.gguf --lora-init-without-apply` + `--control-vector-scaled cc_v2.gguf:1.0` | adapter 0 at 0.5 |

`bench_run` records `control_vector` and `scale` (for the composed arm `scale` is the LoRA's; the vector's 1.0 is in
the server flags logged by the chain). ≈ 7.5 GPU h.

**Readings (fixed).** Per arm against the stage-1 base: GSM8K, MATH (two graders), HumanEval, TriviaQA, compliance;
item-level losses split into cut / not cut.
- **H1 dose recovers code:** an arm with HumanEval ≥ 142 (base − 5) and GSM8K ≥ 75.
- **H2 composition recovers code:** `jack05_vec10` HumanEval ≥ 132 (the compiled's level) with GSM8K ≥ 90.
- The "best balance" row of the ledger moves to whichever arm passes H1 or H2 with the highest GSM8K.

**Prediction:** `predictions.md` row 52, written before the run.
