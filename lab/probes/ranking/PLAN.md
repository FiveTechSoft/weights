# Ranking the servable models under one protocol (driver, 19 Nov 2026: "is MiMo 2.6 the most intelligent, or which one is?") — plan fixed before the run

**Question.** Of the models we can serve on the 12 GB card, which is best on each axis under the *same* suite, budget,
decoding and graders, and is any one best on every axis? "Most intelligent" is not a number we print; the honest proxy
(docs/models.md) is held-out accuracy at a fixed token budget across unrelated axes without losing code or knowledge.

**Protocol.** Suite `rank-2048` (`engine/bench_suites/rank-2048.json`, digest `bd2bb7c2…`): the `dev-2048` items (GSM8K
100, MATH-500 50, two graders) and battery (HumanEval 164, compliance 40, garbage) plus TriviaQA 200 (exact match,
short cap 2,048); IFEval is left for stage 2. Decoding: greedy, seed 0, 2,048 tokens, `-np 2`, each model's own chat
template with thinking as the template does it (MiMo's template does not force thinking; recorded as `think_share`).
One llama-server per configuration (`-ngl 99 -c 16384 -np 2 -fa on --jinja`), served sha in the log, a one-request
smoke per configuration before the suite, `bench_run` with identity. Reference: the Qwen3.5-9B base run on this same
suite (so `bench_compare` pairs item by item; its band against `base_a` / `base_b` of §45.30 is the cross-suite band of
§45.32: GSM |Δ| 4, MATH 1, HumanEval 4-5).

**Stage 1 arms (queue 8, after `QUEUE7 DONE`; ≈ 14 GPU h):**
| arm | served | notes |
|---|---|---|
| `base` | `hf_cache/h4/A0-Q6_K.gguf` | reference |
| `mimo` | `~/llamacpp/mimo-v2.6-distill-qwen-9b-Q6_K.gguf` | full fine-tune of the base (model_report) |
| `jack05` | A0 + `hf_cache/close_lora/demo_jack_r16.gguf` at 0.5 (per request) | our sealed-confirmed transplant (§43.15) |
| `compiled` | `hf_cache/cc_op_v2/merged-Q6_K.gguf`, provenance `report.json` | our conclude compile (§45.8) |
| `llama` | `hf_cache/r1llama8b-gguf/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf` | third family |
| `phi` | `hf_cache/phi4mr-gguf/Phi-4-mini-reasoning-Q6_K.gguf` | 3.8B, second family |

Not in stage 1: Jackrong's full donor (HF only; our `convert_hf_to_gguf.py` predates Qwen3.5 — a GGUF would need a
download, driver's call), TwIL-8B (not local any more; download), SenseNova (needs their llama.cpp fork), the 27Bs.

**Readings (fixed now).**
- Per axis, the ranking by g1 with g2 beside it; two arms are *tied* on an axis when |Δ| ≤ the band of that axis (GSM
  4, MATH 1 → use 2, HumanEval 5, TriviaQA: no band yet → 2 × √(n·p·(1−p)) ≈ 14 at p 0.5, written as provisional).
- **R1 "one model wins every axis"**: a single arm first (or tied first) on GSM8K, MATH-500, HumanEval and TriviaQA.
- **R2 "cost"**: tokens per correct GSM8K answer (mean tokens × 100 / correct) and cut rate per arm, reported beside
  accuracy; the cheapest arm among those within the band of the best.
- **R3 "verdicts for our edits"**: `jack05` and `compiled` are edits of the base → `bench_compare` verdict words apply
  (improves / harms / same, band and battery); the other arms are two-model comparisons (statistics, no verdict word).
- Stage 2 (decided after stage 1, not pre-approved): `dev-2048-full` (IFEval 541) on the top two of stage 1 and the
  base's existing row, ≈ 6 GPU h each; and, if the driver wants it, `dev-8192` on the same arms for the budget curve.

**Prediction:** `predictions.md` row 51, written before the run.

**Not done here.** No sampling, no 8,192 budget, no sealed set (stage 1 is a ranking on dev items, not a claim), no
tuning of templates or system prompts per model (each model's own template, as shipped in its GGUF).
