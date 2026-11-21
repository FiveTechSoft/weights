# Models analysed here: what they are, what we measured, what their cards claim

Rule (driver, 19 Nov 2026): **every model that passes through this repo gets a row**, with three kinds of numbers kept
apart: (1) **measured here** under one protocol (`bench_run`: suite digest, served sha, decoding fixed; or an older
harness, named, with its budget), (2) **read from the weights** (`model_report` / `adapter_report`: lineage, class of
change, size of change), (3) **claimed by the card or the post**, never verified here unless a row in
[`wsplang_benchmarks.md`](wsplang_benchmarks.md) says so. Card numbers from different harnesses, budgets and
decodings are not comparable with each other or with ours; they are kept only so the claim is on record next to what
we could check. The registry of our own runs is `lab/runs/bench/index.json` → [`wsplang_benchmarks.md`](wsplang_benchmarks.md).

## How we compare models (and what "more intelligent" means here)

No single number. A model is compared on **axes**, each under the **same** suite, budget, decoding and graders, paired
item by item against a reference, with the reference's noise band measured under *different batches* (results §45.32:
an identical rerun hides the band). Axes we run: maths reasoning (GSM8K 100, MATH-500 50, two graders), code
(HumanEval 164), instructions (IFEval 541, strict / loose), knowledge (TriviaQA 200), compliance (40), distribution
(WikiText-2 perplexity, KL and top-1 agreement against the base's logits, HellaSwag / Winogrande / MMLU), and **cost**
(mean tokens, cut rate, tokens per correct answer). Two readings that matter more than any score: the **budget curve**
(the same suite at 2,048 and 8,192 tokens: a model that only wins with a long budget is slower, not smarter; §45.17)
and the **sealed confirmation** (a set never used by any decision; `lab/probes/sealed/README.md`). "Better" is a
per-axis statement with p-values and the band; "more intelligent" we do not say: the closest honest proxy is
*held-out accuracy at a fixed token budget across unrelated axes*, i.e. a model that wins on maths, code, knowledge
and instructions at 2,048 tokens without losing the distribution. Contamination is checked by the sealed sets, and by
reading the weights first (a model whose change is a readout-row edit or a bias removal does not "know more", §45.33,
Cohen & Shalev in the README).

## Ledger

Columns: measured here (suite / harness; numbers; results §), weights (what `model_report` / `adapter_report` says),
card / post claims (unverified), servable on our 12 GB llama.cpp build.

| model | what it is (weights) | measured here | card / post claims (not verified) | servable | date |
|---|---|---|---|---|---|
| **Qwen3.5-9B** base (A0 Q6_K; HF bf16 `hf_cache/qwen35-9b-hf`) | the base of most of our work | `dev-2048`: GSM8K 50 / 49, MATH-500 6 / 5, HumanEval 143-148, compliance 14; `dev-2048-full`: IFEval strict 129 / 510, TriviaQA em 77 / 200 (78 cut); `dist-shift`: ppl 8.03, HS 0.78, WG 0.715, MMLU 0.41; band GSM |Δ| 4 across suites (§45.30-45.32) | — | yes | 2026-11 |
| Qwen3.5-9B + conclude vector 1.0 / compiled (ours) | rank-1 write in `down_proj`, direction read from the base | `dev-2048`: 83-87 / 30-35; sealed sets 1-2 confirm (§43.17-19, §45.8); costs +15.6 % ppl, KL 0.175 (§45.31) | — | yes | 2026-11 |
| Qwen3.5-9B + Jackrong adapter r16 at 0.5 (ours, from the donor below) | rank-16 LoRA of the donor's change | sealed: GSM8K 91 / 100, MATH 35 / 50 vs base 43 / 4; placebo 46 / 5 (§43.15) | — | yes | 2026-11-10 |
| Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B (reasoning distill) | `lora_like` on Qwen3.5-9B; rank 16 keeps 58.6 % of the change | donor of the row above; full donor GSM8K 48-item probe 9 → 37 (§43.11) | — | yes | 2026-11-08 |
| XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B (Q6_K) | `full_fine_tune` of Qwen3.5-9B, diffuse (rank 104 / 128); older pre-tokenizer regex (F25); rank 16 keeps 3.8 % (does not transfer) | older harness (greedy, 2,048, same 150 items): GSM8K 93 / 100, MATH-500 34 / 50, 19 cut, 659 / 3,254 tokens (§39-§43); judge for the 0.8B, unreliable there (§45.2) | card: GSM8K ≈ 90 % | yes | 2026-11 |
| TwIL-8B balanced (Q4_K_M) | not read by weights | same older harness: GSM8K 96 / 100, MATH-500 44 / 50, 4 cut (§39) | — | yes | 2026-11 |
| iAmBoosted OSS-Distilled (merged LoRA r16, hub) | `lora_like`, 98.3 % kept at rank 16 | its delta moved onto MiMo: 3× shorter reasoning, MATH cuts 13 → 3, but GSM8K 92 → 83 (p 0.035) → fails its criterion (§43.12) | — | yes | 2026-11-09 |
| LeoZotos math GRPO (RL) | 7.5 % kept at rank 16 (`unlikely`) | adapter indistinguishable from base (15 / 48 vs 9, p 0.21); full donor 21 / 48 (p 0.012) (§43.11) | — | yes | 2026-11-09 |
| xtools-at DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic | `directional_ablation`, layers 9-23, per-layer directions (Heretic ARA) | vs MiMo: GSM8K 4 / 7 discordant (p 0.55), MATH 5 / 13 (p 0.096): statistical tie (§43.14) | — | yes | 2026-11-09 |
| huihui-ai/Huihui-Qwen3.5-9B-abliterated | `directional_ablation`, one global direction, layers 1-31 (validation set of `model_report`) | not benchmarked | — | yes | 2026-11 |
| NeoHorse-1-9B, allenai/tmax-9b (agent full fine-tunes of Qwen3.5-9B) | `full_fine_tune`, rank 104-106 / 128 (validation set) | not benchmarked | tmax card: agent RL | yes (9B) | 2026-11 |
| Indexnusrefather/Nyx-RP-9B-Instruct | `lora_like`, rank ≈ 33 (validation set) | not benchmarked | — | yes | 2026-11 |
| DeepSeek-R1-Distill-Llama-8B (Q6_K) | third family for the conclude direction | `dev-2048`: GSM8K 74 / 75, MATH-500 22 / 20; + vector 0.35: 81 / 37; compiled: 74 / 35; sealed did not confirm at 0.5, confirmed by MATH at 0.35 (§45.19-45.21) | — | yes | 2026-11-17 |
| Phi-4-mini-reasoning (3.8B) | second family | conclude vector and compile pass dev and confirm on sealed set 2; at 2,048 the base recovers half the gain, merged +10 GSM8K (§45.15-45.17) | — | yes | 2026-11-15 |
| Qwen3.5-0.8B (Q6_K and bf16) | smallest family; Exp 8 target | cc08 harness (greedy, 2,048, n 100): base loops; Exp 8 replica 13 / 100 GSM8K, 0 / 50 MATH; harness band ± 5 (§45.22); conclude direction cannot be re-derived here (§43.22) | MIT blog: MATH-500 7.9 → 48.1 % with Exp 8 (other settings) | yes | 2026-11 |
| DavidAU Qwen3.8-27B TURBO Fable Cold Fusion (1.37 M GGUF downloads) | `model_report` 0.8.3 (20 Nov): `full_fine_tune` of Qwen3.8-27B in all 64 layers, matrices 18-40 %, decay parameters ≈ untouched, eos / pad and chat template changed, trained with Unsloth; the 9-hour blind forensic (§43.20) also found a Heretic ablation in layers 26-55 that the fast report misses inside the mixture | not benchmarked (27B) | 11.67 GB checkpoint, 256K context, "0 refusals in 100 prompts" (author) | no (27B) | 2026-11-20 |
| two uncensored 27B, Concept-27B | blind forensics by weights (§43.20, §43.26; `programs/concept27b_forensic.wspl`) | not benchmarked (27B does not fit at Q8) | — | no (Q8) | 2026-11-11 |
| huihui-ai/Huihui-Qwen3.8-27B-abliterated | `directional_ablation`, rank 1, textbook abliteration (§43.26) | not benchmarked | — | no | 2026-11-11 |
| NousResearch/Arete-27B | `full_fine_tune` of Qwen3.8-27B; drops the MTP head | not benchmarked | card numbers not recorded | no | 2026-11 |
| harness-hub/Qwen3.8-27B-pi-adapted (LoRA r 32) | `adapter_report`: 159 M params, MLP on 64 layers + q/k/v/o on the 16 full-attention layers, ‖ΔW‖/‖W‖ 0.2 %, rank 25 / 32 MLP, 1-4 early `v_proj`, no global direction (§45.33) | not benchmarked | SWE-bench Pro hard-51 29.4 → 62.7 (± 14), Terminal-Bench 2.1 39.5 → 55.8, 55-62 % fewer tool calls (pi harness, pass@1) | no | 2026-11-18 |
| sensenova/SenseNova-Loop-9B (BF16 GGUF local) | **starts from Qwen3.5-9B**: `full_fine_tune`, every matrix 14-32 %, `dt_bias` 0.017 %, only mHC routing new (3.1 M); layers 4-27 run twice (§45.33) | not benchmarked (needs their llama.cpp fork) | Terminal-Bench 2.1 59.6, GAIA 69.9, BIRD 72.3, AutomationBench 36.7 (= Qwen3.8-27B), NL2Repo 41.4 vs 44.6; no GSM8K / MATH | no (fork) | 2026-11-18 |
| oguzeldereli/ttf-Qwen3.8-27B (ternary 1.6-bit + 72 M tokens recovery) | not read (ternary + retrained: no float target) | not benchmarked | GSM8K 0.895 = FP16 on 200 problems, HumanEval 0.841 vs 0.933, IFEval 0.735 vs 0.787, TriviaQA 0.538 vs 0.776 | yes with their patches (untested) | 2026-11-18 |
| kitaniai/clover-2 (27B base + 7 LoRA experts, router) | not read | not benchmarked | small samples (n ≈ 12): MATH-style 8 → 24 %, GSM-Hard 36 → 54 % | no | 2026-11-18 |

## Ranking stage 1 (`rank-2048`, 19-20 Nov 2026; results §45.36)

| model | GSM8K | MATH-500 | HumanEval /164 | TriviaQA /200 | compliance /40 | tokens per correct GSM8K |
|---|---|---|---|---|---|---|
| Qwen3.5-9B base | 50 | 6 | **147** | 75 | 17 | 3,378 |
| MiMo 2.6 | **93** | 30 | 129 | 60 | 36 | **440** |
| base + Jackrong ×0.5 (ours) | **93** | 34 | 117 | 75 | 23 | 661 |
| compiled conclude (ours) | 85 | 30 | 132 | **84** | **38** | 936 |
| DeepSeek-R1-Distill-Llama-8B | 78 | 21 | 80 | 46 | 31 | 1,659 |
| Phi-4-mini-reasoning | 83 | 30 | 57 | 36 | 13 | 1,308 |
| Jackrong full (official GGUF) | 93 | 30 | 47 (GGUF breaks indentation) | 84 | 27 | 445 |
| TwIL-8B balanced (Q4_K_M) | 91 | **39** | 67 (97 cut) | 62 | 35 | 791 |
| base + conclude vector 0.5 (ours, run-time) | 72 | 18 | 126 | **89** | 36 | 1,663 |
| base + conclude vector 0.7 (ours, run-time) | 84 | 21 | 126 | 83 | 37 | 1,110 |
| base + Jackrong ×0.5 + conclude vector 1.0 (ours) | 88 | 38 | 100 | 72 | **40** | **283** |

No model wins every axis. Finalists for stage 2 (IFEval): compiled conclude, MiMo, TwIL. The last three rows (results §45.37, 20 Nov) test whether a lower dose or a composition gives HumanEval back: it does not (code loss is flat across dose, 126-132); the composition is the cheapest arm of the ranking (283 tokens per correct GSM8K) and second in MATH.

### Plan as written before the run

`lab/probes/ranking/PLAN.md`, prediction row 51: base, MiMo 2.6, base + Jackrong ×0.5, compiled conclude, R1-Llama-8B
and Phi-4-mini-reasoning on `rank-2048` under one server configuration each; results go to the rows above and to
`wsplang_benchmarks.md` through `bench_table`. Stage 2 (IFEval through `dev-2048-full`, and the 8,192 budget curve)
on the finalists, decided after stage 1.

## What a new row needs

1. `model_report` (or `adapter_report`) against the declared base: lineage and class before any number.
2. If it fits the card: `bench_run suite=dev-2048` (and `dev-2048-full` if it passes), paired with the base's run and
   the band; the row goes to `wsplang_benchmarks.md` through `bench_table`, and the summary here with the § of results.
3. The card's own numbers copied verbatim with their harness named, in the "claims" column, never mixed with ours.
