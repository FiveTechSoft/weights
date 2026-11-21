# Preservation battery beside every sealed confirmation — plan fixed 2026-11-18 (roadmap gap 19), local data only

**Why.** The conclude direction gains on GSM8K / MATH and costs something elsewhere (Llama §45.19-45.21: 10-20 % of
base-correct items finished wrong; HumanEval-20 moved 14 → 13). The only capability check in the protocol is
HumanEval on 20 tasks, run after the sealed set and not part of the verdict. This battery makes the cost side of every
sealed run measurable, paired item by item, with what is on disk now. It is deliberately narrow and says so.

## What exists locally (inventory 2026-11-18)
- `data/humaneval.jsonl`: all 164 HumanEval tasks (prompt, test, entry_point). Executor: `lab/probes/close_probe/he_eval.py`
  (`build_program`, `run_program`: last ```python block, prompt prepended if it lacks the function, official `check()`
  in a `python -I` subprocess with a 10 s timeout). Timings on the 9B at cap 2 048, parallel 2:
  `he_name_base` 20 tasks 171 s, `he_name_vec` 81 s, `he_ccw_vec` 72 s; Llama-8B base 372 s, Phi base 241 s.
- `data/gsm8k_test.jsonl` (1 319) and MATH-500 via `rig/math500.py`: used by dev (GSM8K 0-99) and the three sealed sets
  (indices 100-1317, 400 distinct). Nothing else: **no IFEval, TriviaQA, MMLU, ARC or any knowledge-recall set on disk**
  (grep of `data/`, `lab/data/`, `lab/runs/`, README; the user's untracked `book_*` / `corpus_n4*` files are not counted).
- `engine/bench_runner.py`: the serving protocol (greedy, seed 0, template, resume, `loop_tail` on 8-char lines × 3).

## What is built: `engine/preserve_bench.py` (tests first: `tests/test_preserve_bench.py`, 6 tests)
`run(name, port, out_dir, max_tokens=2048, parallel=2, he_n=164, comp_n=40, lora=None)` → `<out_dir>/<name>.json`
(bench_runner shape: `summary`, `items` with ids, `served` sha256, resume from `.partial.jsonl`). Three components:
1. **humaneval**: 164 tasks, greedy pass@1, same prompt as HE-20, executor imported from `he_eval.py`. Reports n / ok /
   closed / median tokens.
2. **compliance** (40 prompts, **our own set, not IFEval**): GSM8K test questions at indices ≥ 1000 that are in no sealed
   set (first 40 eligible: 1000-1004, 1007, 1009, 1010, …), each with one of four surface instructions, 10 per kind,
   checked by rule on the content after `</think>`:
   `end_done` (last word is DONE) · `words_not_digits` (no digit in the reply) · `one_sentence` (exactly one sentence
   terminator; decimals and e.g./i.e./Dr. excluded; no blank line) · `all_caps` (≥ 10 letters, all upper case).
   It measures whether a simple instruction is still obeyed after an edit; it says nothing about reasoning or knowledge.
3. **garbage** over every output of 1 + 2 (reasoning + content): `repeat_16gram` (a 16-word window ≥ 3 times),
   `nonascii_run` (a run of ≥ 40 non-ASCII characters, or ≥ 3 distinct non-ASCII scripts in one 60-character window: the
   fragment salad of a broken model), `cut_at_max`. Checked on real outputs: the TQ2_0 0.8B of §45.23 is flagged
   10 / 10, the 9B dev base 0 / 150 for salad and 2 / 150 for loops (= bench_runner's own `loop_tails` 2).
`compare(base_rows, arm_rows)` gives the paired counts (lost / gained per component, garbage rate per arm) and the rule.

## What it cannot measure (stated)
Knowledge recall (TriviaQA / MMLU-style), real instruction following (IFEval's 25 verifiable types), long-context
behaviour, safety / refusal, anything under sampling. A "preserves" verdict here means: code generation on 164 tasks,
four surface instructions on 40 maths questions, and no visible output decay. Nothing more.

## Reading rule for sealed runs (fixed now; paired against the base arm of the same chain)
An arm **preserves** iff HumanEval ok ≥ base − 5 (of 164) **and** compliance ok ≥ base − 3 (of 40) **and**
garbage rate (repeat_16gram + nonascii_run over all 204 outputs) ≤ base + 2 pp. Reported beside C1-C3, not inside the
confirmation verdict (a confirmation that fails preservation is written as "confirms, does not preserve"). The
HumanEval band is the two-run noise seen at n = 20 (13 → 14) scaled by ~3; it is a guess until two base runs exist
(the first use must run the base twice and record the band, as §45.22 taught for the 0.8B).

## Cost per arm (estimate from the timings above, cap 2 048, parallel 2)
| model | HumanEval 164 | compliance 40 (thinking, ~1 500 tokens) | total |
|---|---|---|---|
| Qwen3.5-9B Q6_K | ≈ 23 min base, 11 min with the vector | ≈ 15-20 min | **≈ 35-45 min** |
| Llama-8B Q6_K | ≈ 50 min | ≈ 25 min | ≈ 75 min |
| Phi-4-mini | ≈ 33 min | ≈ 20 min | ≈ 55 min |
Three arms (base, vector, placebo) on the 9B ≈ 2 GPU h on top of a sealed run.

## Smoke (not launched): `lab/runs/preserve/smoke.sh`
9B base only (`hf_cache/h4/A0-Q6_K.gguf`, port 8095): 2 HumanEval tasks at 256 tokens as the smoke, then the full
battery once; `set -eu`, ERR trap, sha256 of the served GGUF in the log, waits for any other llama-server to end.

## Needs the driver's OK (downloads; none done)
| set | what | size | source | licence |
|---|---|---|---|---|
| IFEval | 541 prompts, 25 verifiable instruction types, official checker code | ~1 MB data + checker (`instruction_following_eval`) | `google/IFEval` on the Hub; checker in `google-research/instruction_following_eval` | Apache-2.0 |
| TriviaQA (rc, validation) | knowledge recall, exact-match; a fixed 200-question subset would do | full split ~2.5 GB; a 200-item subset can be read by HTTP ranges / streaming without the full file | `mandarjoshi/trivia_qa` | Apache-2.0 |
| MMLU (optional) | 57-subject multiple choice; 500-item subset | ~160 MB full, subset small | `cais/mmlu` | MIT |
With IFEval the compliance component is replaced by the real thing; with TriviaQA the battery gains the dimension *ttf*
and Kestrom show is the first to go (knowledge).
