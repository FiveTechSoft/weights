# A constant logit bias on `</think>` as the control for the "conclude" vector (Qwen3.5-9B) — plan fixed 2026-11-18, before any run

**Why.** The conclude vector (§43.17-43.18, cc_v2) and its compile (§43.19, §43.25) are read as a direction that makes
the model close when the answer is there. DAWE (README "References") repairs a stop/continue failure with a
state-dependent edit and states that it never ran the obvious control: a constant bias on the end token. We never ran
it either. If a constant push on the `</think>` logit, chosen only to match the vector's token saving, gives the same
accuracy on the same dev items, the "state-dependent" reading of the vector is not shown; what the vector does would be
explained by "stop earlier, everywhere". If the bias falls clearly short, the vector carries something a constant push
does not. Roadmap gap 18.

## What exists (no re-run)
`lab/runs/cc_v2/dev_base.json`, `dev_vec.json` (control vector `hf_cache/cc/cc_v2.gguf` at scale 1.0, picked on the
screen, `scale.json`), `dev_placebo.json`: bench_runner defaults (GSM8K 0-99 + MATH-500 select_indices(2026, 50)),
thinking template, greedy, 2,048 tokens, `-np 2`, model `hf_cache/h4/A0-Q6_K.gguf`, port 8095.

| arm | GSM8K ok (g2) / finished / cut / tokens | MATH ok (g2) / finished / cut / tokens |
|---|---|---|
| dev_base | 50 (49) / 49 / 51 / 1,689 | 6 (6) / 5 / 45 / 1,952 |
| dev_vec | 83 (84) / 92 / 8 / 735 | 35 (27) / 33 / 17 / 1,359 |
| dev_placebo | 26 (27) / 27 / 73 / 1,834 | 5 (5) / 5 / 45 / 1,969 |

## Arms (one run each, `lab/runs/cc_bias/`)
Base GGUF served with **no control vector**; the bias travels in the request as llama-server's `logit_bias`
`[[248069, b]]` (token id of `</think>` in `hf_cache/qwen35-9b-hf/tokenizer.json`, added token 248069; verified by
encoding `</think>` → `[248069]`). Same items, budget, template, greedy, `-np 2`, same binary.
- `dev_bias_2`, `dev_bias_4`, `dev_bias_6`, `dev_bias_8`: b = +2, +4, +6, +8 added to the `</think>` logit at every
  position of the generation.
- `bench_runner.ask` has no hook for extra request fields; the wrapper `lab/probes/cc_bias/run_arm.py` replaces
  `bench_runner.ask` in the module with the same function plus `logit_bias` (12 lines copied, one added), so
  `bench_runner.run` itself is unchanged. The minimal patch to bench_runner is written in `BENCH_RUNNER_PATCH.md` for the
  coordinator; this plan does not depend on it.

## Pick rule (fixed)
Picked bias = the arm whose pooled mean tokens over the 150 items (n-weighted over the two benches) is closest to
`dev_vec`'s pooled mean (735 × 100 + 1,359 × 50) / 150 = 943; ties to the smaller bias. The pick uses tokens only,
never accuracy.

## Readings (fixed; `bias.py compare`)
On the picked arm, against `dev_vec` and `dev_base`, per bench: correct (grader 1; grader 2 reported), finished
(`stop`), cuts, no-think share (`reasoning_chars` < 50), mean tokens, paired discordants vs `dev_vec` (exact two-sided
binomial), lost (base correct and finished → wrong) and gained (base wrong → right), as in
`lab/probes/cc_llama/pair_recount.py`.
- **R1 (constant push explains the vector):** picked bias GSM8K correct ≥ vec − 3 **and** MATH correct ≥ vec − 2
  **and** discordants vs vec not significant (p ≥ 0.05) on both benches.
- **R2 (state-dependent part exists):** R1 fails with the picked arm below vec on at least one bench by more than the
  margin. Reported with the size.
- Also: which bias arms beat `dev_base` (GSM8K and MATH, paired p), and the no-think share per arm (the known failure
  mode of a pure stop signal, §45.12).
- The loss decomposition on the picked arm is reported next to the vector's (vector: lost 2-4 per bench on this
  model, §45.19 notes); if the bias loses many more base-correct finished items than the vector at equal tokens, that is
  the state dependence made visible.

Verdict wording: R1 → "the conclude vector's dev gain is reproduced by a constant `</think>` bias at equal token
saving; the vector is not shown to be more than a stop signal on this model". R2 → "a constant bias at the same token
saving gets X / Y vs the vector's 83 / 35; the difference is what the direction adds beyond stopping".

## Prediction (row 43)
Premises: a bias adds to the logit at every position, so it fires earliest where `</think>` is already close, which is
also where the vector fires (§43.17 rows are the last 48 tokens before `</think>`); unlike the vector it also fires
mid-derivation. Forced close at 2,048 on this model turned cuts into answers (§43.3, 41/50 MATH with 4,096 of
reasoning), so a bias that mostly converts cuts should gain on MATH too.
- `dev_bias_2`: GSM8K 52-62, MATH 7-12, pooled tokens 1,400-1,700, no-think 0 %.
- `dev_bias_4`: GSM8K 60-75, MATH 10-20, tokens 1,000-1,400, no-think ≤ 2 %.
- `dev_bias_6`: GSM8K 62-80, MATH 12-26, tokens 650-1,050, no-think ≤ 5 %.
- `dev_bias_8`: GSM8K 45-75, MATH 8-22, tokens 350-800, no-think 5-40 %.
- Tokens fall monotonically with b. Picked bias = +6 (55 %) or +4 (30 %).
- Picked arm vs vector: GSM8K vec − 4 to − 18, MATH vec − 5 to − 18; discordants significant on GSM8K.
- **Primary: R2 (60 %); R1 (35 %); picked arm ≤ base on either bench (5 %).**
- We would be surprised if: any bias arm ≥ vec on both benches; the picked arm ≤ base on GSM8K; no-think > 50 % at +8;
  tokens not monotone in b; any arm with graders disagreeing on > 5 items.

## Two methods
`bias.py compare` (this probe) and `lab/probes/cc_llama/pair_recount.py` on the same JSON files (names relative to
lab/runs: `cc_v2/dev_vec`, `cc_bias/dev_bias_6`) must give the same correct / finished / cut / tokens and the same
discordant pairs.

Cost: 4 arms × 20-30 min ≈ 1.5-2 GPU h (bias arms are shorter than base). Chain `lab/runs/cc_bias/chain.sh`.
