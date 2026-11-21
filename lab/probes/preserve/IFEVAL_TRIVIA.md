# IFEval and TriviaQA in the preservation battery — what is supported, what is not, what it costs (2026-11-18)

Approved by the driver ("OK, como tú veas"). Files in `data/bench/` (git-ignored, pinned by sha256 in
`engine/bench_suites/dev-2048-full.json` and listed in `data/bench/README.md`):

| file | source | size | sha256 (16) | licence |
|---|---|---|---|---|
| `ifeval_input_data.jsonl` | google-research/instruction_following_eval `data/input_data.jsonl` | 207,111 B | `67ffeee0fcb87c31` | Apache-2.0 |
| `triviaqa_rc_val_200.jsonl` | HF datasets-server rows API, `mandarjoshi/trivia_qa` `rc.nocontext` validation, rows 0-199 | 142,121 B | `7421edff5746316b` | Apache-2.0 |

## IFEval (engine/ifeval_checks.py)

541 prompts, 834 instructions over 25 instruction types. The reference checker (`instructions.py`) is re-implemented in
pure Python for **24 types**; the repo is not vendored.

Supported (24): `keywords:existence`, `keywords:frequency`, `keywords:forbidden_words`, `keywords:letter_frequency`,
`length_constraints:number_words`, `length_constraints:number_sentences`, `length_constraints:number_paragraphs`,
`length_constraints:nth_paragraph_first_word`, `detectable_content:number_placeholders`, `detectable_content:postscript`,
`detectable_format:number_bullet_lists`, `detectable_format:constrained_response`,
`detectable_format:number_highlighted_sections`, `detectable_format:multiple_sections`, `detectable_format:json_format`,
`detectable_format:title`, `combination:two_responses`, `combination:repeat_prompt`, `startend:end_checker`,
`startend:quotation`, `change_case:capital_word_frequency`, `change_case:english_capital`, `change_case:english_lowercase`,
`punctuation:no_comma`.

Unsupported (1): `language:response_language` (needs `langdetect`). The 31 prompts that contain it are run but not
scored: **510 scorable prompts** (`summary.ifeval.scorable`), 834 − 31 = 803 scorable instructions.

Deviations from the reference, so the numbers are "IFEval-style", not the leaderboard's:
- `english_capital` / `english_lowercase` check only the case (the reference also requires `langdetect == "en"`).
- `number_sentences` splits on `[.!?]` + whitespace; the reference uses nltk punkt.
- `number_words` and `capital_word_frequency` count `\w+` tokens; the reference uses nltk's word tokenizer
  (contractions and hyphenated words can count differently).

Scoring: **strict** = every instruction passes on the raw response; **loose** (the paper's) = for each instruction some
variant passes among: the response, minus first line, minus last line, minus both, each also with all `*` removed.
Prompt-level (`prompt_strict`, `prompt_loose`) and instruction-level (`instruction_strict`, `instruction_loose`)
accuracies are reported; per-item rows keep `instruction_ids`, `per_instruction_strict/loose`, `scorable`.
The model's thinking is stripped (`split_think`); the response is the content after it, so a prompt whose answer is cut
at `max_tokens` fails.

## TriviaQA (engine/preserve_bench.py)

200 questions, prompt `TRIVIA_PROMPT` ("…with just the answer…"). **em** = the normalised first non-empty line of the
answer equals one of the normalised aliases (SQuAD normalisation: lowercase, drop punctuation and articles, collapse
whitespace; aliases = `normalized_aliases` ∪ normalised `aliases` ∪ normalised `value`). **contains** = some alias
occurs as a whole-word phrase inside the normalised answer (reported apart, looser). Also counted: empty answers and
answers cut at `short_max_tokens`.

## In the suites and the rule

`engine/bench_suites/dev-2048-full.json` = `dev-2048` (same items, decoding, graders; its digest is unchanged) plus
`preservation.ifeval: 541`, `preservation.triviaqa: 200`, `short_max_tokens: 2048`, and the two data shas. `bench_run`
passes them to `preserve_bench.run(ifeval_n=, trivia_n=, short_max_tokens=)`. `compare()` reports paired lost / gained
for IFEval strict and loose and for TriviaQA em; the thresholds (`THRESHOLDS`: `ifeval_strict`, `ifeval_loose`,
`triviaqa`) are **null until calibrated** with the base run twice and the same-norm placebo, so they do not enter the
`preserves` flag yet (`notes` says so). Table columns: "IFEval strict / loose (scorable)" and "TriviaQA em /200".

## Cost on the 9B (RTX 3060, 2 slots, ≈ 39 tok/s per slot)

Measured on the base's first battery run (`lab/runs/preserve/base.json`): 204 items in 2,936 s (14.4 s per item),
HumanEval mean 930 tokens, compliance mean **1,769 tokens** (the 9B thinks almost to the 2,048 cap on short prompts).
IFEval prompts are open-ended writing tasks; with thinking on, expect 1,500-1,900 tokens each:
- IFEval 541 × ≈ 1,700 tokens / (2 × 39 tok/s) ≈ **3.3 h**;
- TriviaQA 200 × ≈ 700 tokens / 78 tok/s ≈ **30 min** (thinking dominates; the answer itself is a few tokens);
- total **≈ 3.5-4 GPU h per arm** on top of the 50 min of HumanEval + compliance and the 57 min of GSM8K + MATH.

That is too much to run beside every sealed set. Options, to decide with the driver: (a) run `dev-2048-full` only for
the base (twice, for the band) and for the arms that pass the cheaper battery; (b) a suite with `ifeval: 200`
(first 200 prompts, deterministic) for routine arms (≈ 1.2 h); (c) `short_max_tokens: 512` for TriviaQA, which cuts
the thinking and counts the cuts (cheap, but it measures "answers briefly under a cap", not knowledge). Not decided here.

## Tests

`tests/test_preserve_ifeval.py` (12): every supported check on hand-made responses, strict vs loose, unsupported ids,
loader counts (541 / 510 scorable; 200 TriviaQA, first id `tc_2`), normalisation and alias match, a fake-server run of
3 IFEval + 3 TriviaQA items, compare with null thresholds. With `test_preserve_bench`, `test_wsplang_bench` and
`test_wsplang_spec`: 40 tests OK.
