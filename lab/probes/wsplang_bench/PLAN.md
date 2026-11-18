# WSP-Lang benchmarks: one way to check and compare any model — plan fixed 2026-11-18

**Why.** The repo measures with one protocol (`engine/bench_runner.py`: GSM8K / MATH-500, greedy, two graders) but every
study assembled its own chain, labels, dev items and comparison script; results §45.11 (an arm labelled "merged" that
served the base) and §45.22 (11 discordant items of 100 between identical runs) are the two failure modes: not knowing
what was served, and not knowing the noise. The driver asked for a standard inside the language, usable on any model,
the ones WSP-Lang generates included.

## What it standardises
1. **Suites** shipped with the language (`engine/bench_suites/`), digest-verified: items, decoding, model class,
   graders, preservation battery. Changing any of them is a new suite with a new sha, never a silent change.
2. **Identity**: every run records the served file's sha256, the declared extras (control vector, LoRA, scale, request
   fields), the llama-server binary sha, the suite sha and the decoding; a run without a model path is refused unless
   asked to record it as unidentified. For NanoWSP checkpoints the identity is the file's sha.
3. **Provenance**: a model WSP-Lang built is benchmarked with its manifest (`wsp.py`, `compile_control_vector`,
   `edit_column`, `commit_manifest` bundle); the manifest's output sha must match the served sha; the table shows
   `base@sha + N patches`. Only then is a comparison "an edit against its base".
4. **Comparison**: paired item by item, both graders, exact binomial, loss decomposition (base-correct finished items
   lost; cut recovery), a noise band from two runs of the base, the preservation battery; the verdict rule written once
   (`engine/wsplang_bench.py` docstring, `docs/wsplang_0.8_bench_addendum.md`).
5. **Registry and table**: `lab/runs/bench/index.json` → `docs/wsplang_benchmarks.md`, one row per run with the sha,
   one row per comparison with the verdict and the band.

## Verdict rule (verbatim from the module)
"improves" only if (a) on at least one benchmark the arm beats the base with exact two-sided binomial p < 0.05 on both
graders, (b) the placebo arm, when given, does not gain significantly on any benchmark / grader (a placebo that harms
does not block), (c) when a noise band is given
(two runs of the same configuration), |arm − base| ≥ 2 × band on that benchmark, (d) the preservation battery, when
both runs carry it, does not fail. "harms": the symmetric statement with the arm below the base. Otherwise
"same / not decidable", with the reasons. Two unrelated models: paired statistics and band only, no verdict word.

## How a third party reproduces a row
Serve the file whose sha256 the row shows (full sha, path and extras in the run JSON's `identity`), llama-server flags
`-c 16384 -np 2 -fa on --jinja` (greedy is in the request), `step bench_run suite=<same name> ...`, then
`step bench_compare` against the row's base run. Item records are in both JSONs. For a nanowsp row: the checkpoint file
with that sha and `bench_run suite=nanowsp-wspx model=<file>` on the CPU.

## Deliberately out
Sampling (temperature > 0) and other budgets are other suites; knowledge sets (TriviaQA, MMLU) and real IFEval need a
download the driver has not approved; the sealed protocol stays where it is (this counts, it does not decide claims);
no leaderboard semantics: the table is a record of runs with their identity, not a ranking.

## Backfill (CPU, no new runs)
`lab/probes/wsplang_bench/backfill.py` imports existing `bench_runner` reports that match `dev-2048` exactly (GSM8K 0-99,
MATH-500 `select_indices(2026, 50)`, 2 048 tokens, no prompt change): the 9B cc_v2 dev arms (base, vector, placebo,
no31, merged, placebo_merged), the cc_weights dev arms (lora, merged, placebo_merged), the Llama-8B dev arms (base,
vector, placebo, vector at 0.25 / 0.35, merged, placebo_merged). Each imported run records `imported_from`, keeps
the original `served` field (older reports have none: identity unknown, marked so) and carries no preservation. The
Phi dev arms (1 024 tokens) do not match the suite and are not imported. Comparisons backfilled: 9B vector vs base with
placebo; 9B merged vs base with placebo_merged; Llama vector vs base with placebo. No base-twice band exists yet for the
9B: the comparisons say so.

## First GPU runs to queue (not launched here)
1. 9B base twice on `dev-2048` (≈ 1 h with the preservation battery ≈ 2 × 40 min more): the band the table needs.
2. The three cc_v2 dev arms through `bench_run` proper (identity + preservation), ≈ 3 × 1 h: replaces the imported rows.
3. Every new study's dev and sealed arms through `bench_run` from now on (the chain scripts call the op; the sealed
   suites keep their single use).
