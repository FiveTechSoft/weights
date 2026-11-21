# A second behaviour with the same protocol: "reason in Spanish" on Qwen3.5-9B — plan fixed 2026-11-18, before any run

**Why.** Every programmed behaviour in this repository is "conclude" (§43.17-§45.21: three families, one direction).
The driver asked (2026-11-18) for a second behaviour, distinct from closing, with the protocol unchanged: a direction
read from the base's own outputs (`contrast_vector`), a same-norm random placebo, a scale chosen on a screen set, dev
criteria fixed here, a fresh sealed subset used once, and compilation into `down_proj` only if the vector passes. The
question is whether the protocol is a method or a one-off: does it program anything that is not "stop thinking"?

## Choosing the behaviour (CPU screen on existing generations, 2026-11-18)
Four candidates were screened against local data only. Numbers:
- **Format compliance** (final answer extractable / boxed): among finished answers, items with no extractable
  prediction = **0 / 54** (cc_v2 dev base), 0 / 125 (dev vec), 0 / 102 (Llama dev base), 0 / 65 (Phi dev base). No
  headroom. Rejected.
- **No repeated computation** (duplicate reasoning lines, repeated 16-grams, `loop_tail`) on the 128 full 9B traces of
  `lab/runs/cc/traces.jsonl`: finished traces (53) have duplicate-line share median 0.000, p75 0.02, max 0.062,
  `loop_tail` 6 / 53; cut traces (75) median 0.023, max 0.20, `loop_tail` 35 / 75. Repetition lives in the cut traces,
  so any gain on it would be cut recovery = "conclude" relabelled. Rejected by the rule set in the brief.
- **Loop suppression on Qwen3.5-0.8B** (the MIT Exp 8 phenomenon): the same stop/close family, and the 0.8B harness
  has a ± 5 run-to-run band at n = 100 (§45.22). Rejected.
- **Language of the reasoning.** Problems are English; the measure is the language of the *first 300 words of the
  thinking*, judged by a deterministic stopword detector (`skill2.lang`, Spanish vs English word lists, the larger side
  must lead by 1.5×; fewer than 8 words or fewer than 4 stopwords → unknown). It does not depend on length, on cuts or
  on closing: a trace cut at 2,048 tokens is judged on its head like any other. Base headroom is a premise (the 9B is
  expected to think in English on nearly every English problem; the discovery step measures it, see "Stop rules").
  Chosen. The behaviour is the one DeepSeek-R1 trained a "language consistency" reward for and DAWE calls a reply-format
  repair: a real target, not ours until now.

What is new relative to "conclude": the rows contrast **two answers to the same problem in two languages**, both
produced by the base (one under the instruction "Piensa y responde en español."), not two positions of one trace; the
direction is applied with **no instruction** at all in the primary arms, so a gain means the vector programs the
behaviour, not that it helps follow an instruction. Secondary arms with the instruction are reported, not judged.

## Model, serving, data
- Qwen3.5-9B: `hf_cache/qwen35-9b-hf` (vector, compile), `hf_cache/h4/A0-Q6_K.gguf` served (`-ngl 99 -c 16384 -np 2
  -fa on --jinja`, port 8095), thinking on by the template, greedy, 2,048 tokens, two graders for accuracy as in
  `bench_runner` (the runner in `skill2.py` reuses `ask`, the graders, `tail`, `loop_tail` and `served_info` by import,
  adding `think_lang`, `answer_lang` and the 60-word reasoning head per item).
- Discovery: GSM8K **train** 64-127 (`split.json`); screen: GSM8K train 128-159; dev: `bench_runner` defaults (GSM8K
  test 0-99 + MATH-500 `select_indices(2026, 50)`); sealed: `data/sealed_skill2.json`, made by the chain only if dev
  passes, with `skill2.sealed_ids(seed 20261118)`: 100 GSM8K test + 50 MATH-500 drawn from the items **not** in dev and
  not in `data/sealed_confirm.json`, `sealed_confirm2.json`, `sealed_confirm3.json` (all three hash-checked when
  loaded); sha256 of texts and answers recorded with `engine.sealed_set.digest`; single use, `make_sealed` refuses to
  overwrite. Train items 64-191 were also used by the Llama chain's extra traces (§45.19); train items are never
  evaluated anywhere, so this is not leakage, only shared discovery material across models.
- No download. Nothing from dev or sealed sets is used before its step.

## Steps (chain `lab/runs/skill2/chain.sh`)
1. **Discovery generations** (`skill2.py gen`): each of the 64 problems answered twice by the base, plain and with
   `extra = " Piensa y responde en español."` appended after `bench_runner.SUFFIX`; full reasoning and content kept.
2. **Rows** (`skill2.py rows`): pos = the Spanish answer's content, neg = the English answer's content, **same prompt**
   (question + SUFFIX, no instruction) for both, `think: false`, whole content (first 384 tokens by the op). A pair is
   kept only if the judge calls the Spanish arm `es` and the English arm `en`; unknowns and non-compliant pairs are
   counted. **Stop rule:** fewer than 30 pairs → STOP, reported.
3. **Vector**: `contrast_vector` (programs/skill2_vector.wspl), placebo seed 2026 (same norm per layer, random).
4. **Screen** (32 GSM8K train items, no instruction): base and vector at 0.5 / 1.0 / 2.0 / 4.0. Rule
   (`choose_scale`): among scales with correct ≥ base − 3, the most Spanish thinking (ties → smaller scale); it must
   reach base + 30 pp (≥ base + 9.6 items of 32), else STOP: "no scale programs the language without costing accuracy".
5. **Dev** (150 items): `base_en`, `vec_en` (chosen scale), `placebo_en` (same scale); secondary `base_es`, `vec_es`
   (with the instruction). One run per arm.
6. **Criteria on dev** (`dev_verdict`, all of):
   - **C1** vector: Spanish thinking ≥ base + 30 pp of the 150 items and exact two-sided binomial on discordant items
     p < 0.05;
   - **C2** placebo: discordant p ≥ 0.05 and Spanish thinking < base + 10 pp;
   - **C3** accuracy: per bench, vector correct (grader 1) ≥ base − 3; tokens, cuts, loops and answer language reported.
   Dev PASS → the sealed set is made and run (`sealed_base_en`, `sealed_vec_en`, `sealed_placebo_en`), same C1-C3 →
   CONFIRMS / DOES NOT CONFIRM. Dev FAIL → nothing else runs.
7. **Part B (compile), only if the sealed set confirms:** vector and placebo scaled to the chosen scale
   (`lab/probes/cc_phi/scale_vector.py`), calibration rows = the finished English discovery traces (`skill2.py calib`,
   no `<think>` prefix, the Llama rule), `compile_control_vector` (programs/skill2_compile.wspl, `out=hf_cache/skill2_op`,
   G0 gate as always), dev `merged` and `placebo_merged` in the vec / placebo slots of C1-C3 (`cdev`) plus **C4**
   fidelity: merged = vector item by item on Spanish thinking (discordant p ≥ 0.05).

Chain guards: `chain_guard` configs in `lab/runs/skill2`, `/dev`, `/sealed`; 2-item smoke per served configuration;
"no valid control vector" check; sha256 of every served GGUF and of the vector files in the log; `no_server` before
any step; resumable.

## Prediction (row 44)
Premises (marked P): P1 the base thinks in English on ≥ 95 % of English problems without the instruction and on
≥ 60 % *with* it (the instruction moves the answer, less the thinking; the discovery step measures both and the
numbers go in the results whatever they are). P2 the base's answer under the instruction is Spanish on ≥ 85 % of
discovery problems, so ≥ 45 of 64 pairs survive the judge.
- Rows: 45-62 pairs kept (P2); vector last-layer norm within 0.3-3× the conclude vector's (`hf_cache/cc/cc_v2.json`).
- Screen (32 items, no instruction): base Spanish thinking 0-2, correct 10-18; some scale reaches ≥ base + 10 Spanish
  items with accuracy ≥ base − 3 (**60 %**); if one does, it is 1.0 or 2.0 (70 % conditional); at 4.0 accuracy collapses
  (correct ≤ base − 5) in 70 %.
- Dev (if reached): `base_en` Spanish thinking 0-5 of 150, GSM8K 40-55 / 100, MATH 4-12 / 50, tokens as cc_v2
  (≈ 1,700 / 1,950); `vec_en` Spanish thinking **45-110** of 150, GSM8K base −8 to +2, MATH base −4 to +2, tokens ± 15 %;
  `placebo_en` Spanish thinking 0-8, accuracy ± 3; `base_es` (instruction) Spanish thinking 10-60, Spanish answers
  ≥ 120; `vec_es` Spanish thinking ≥ `base_es` + 30.
- **Primary: C1 passes on dev (55 %); C1-C3 all pass (35 %); the sealed set confirms (25 %).** The likeliest failure is
  C3: thinking in Spanish costs GSM8K more than 3 items (premise: the base reasons better in its dominant language).
- Part B (if reached): G0 passes (70 %), merged within ± 10 Spanish-thinking items of the vector (C4 passes 65 %).
- We would be surprised if: the placebo moves Spanish thinking by ≥ 20 items at the chosen scale; the vector moves
  Spanish thinking by < 10 items at every scale (a direction read from answers does not act on thinking); accuracy
  *rises* by > 5 under the vector; the base already thinks in Spanish on > 20 % of items without the instruction.

## Reading rule (fixed)
- Dev C1-C3 pass and sealed confirms → "the protocol programs a second, unrelated behaviour on this family; the cost in
  accuracy is X". Dev C1 passes, C3 fails → "the direction transfers from answers to thinking, but the model reasons
  worse in Spanish: a behaviour, not a free one" (reported as such, not as a pass). C1 fails → "a language direction
  read from answers does not move the thinking at any safe scale": the protocol's first negative outside "conclude".
- Two methods for every number: the chain's summaries and a per-item recount (`lab/probes/cc_llama/pair_recount.py`
  works on these files for accuracy; the Spanish-thinking pairs are recounted with `skill2._paired` from the partial
  jsonl, and spot-checked by eye on 10 reasoning heads per arm).

## Cost
Timings from cc_v2 on the same card: 150 dev items ≈ 57 min (base), less under the vector. Discovery 64 × 2 at 2,048
≈ 30 min; op ≈ 10 min; screen 5 × 32 ≈ 1 h; dev 5 arms ≈ 4.5 h (Part A to the dev verdict ≈ 6.5 GPU h); sealed 3 arms
≈ 2.8 h; Part B ≈ 2.3 h. Everything passing ≈ 11.5 GPU h; the common case (stop at screen or dev) ≈ 2-6.5 GPU h.
Prediction row 44 (to be copied to predictions.md by the coordinator).
