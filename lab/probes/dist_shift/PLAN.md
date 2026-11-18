# dist-shift: a label-free distribution-shift and general-capability component for WSP-Lang benchmarks — plan fixed 2026-11-18, before any GPU run

**Why.** Our benchmark section (`docs/wsplang_0.8_bench_addendum.md`) measures accuracy on GSM8K / MATH and a narrow
preservation battery. It has no measure of *how much an edit moved the model's output distribution in general*, which
is the cost side of every WSP-Lang edit (T6: the same-norm placebo; §45.19-45.21: the Llama vector finishing
base-correct items wrong). llama.cpp ships the tool: `llama-perplexity` computes perplexity on a text, the KL divergence
of a model against a base's saved logits token by token (`--kl-divergence` / `--kl-divergence-base`), and HellaSwag /
Winogrande / MMLU multiple-choice scores. All teacher-forced, deterministic for a GGUF + binary + file, no sampling, no
server, minutes per model. The driver approved the data downloads and GPU runs on the 9B (2026-11-18).

## What is built (CPU, TDD; `tests/test_dist_shift.py`, 13 tests written first)
- `engine/dist_shift.py`: command builders for the six llama-perplexity invocations, parsers for their outputs
  (fixtures = real CPU runs on the 0.8B), `run_dist_shift`, `make_logits_suite`.
- Model class **`llm-logits`** in `engine/wsplang_bench.py`: `bench_run suite=dist-shift model=<gguf> base=<gguf>
  arm=<label> out=<json> [control_vector= scale=] [binary=<llama-perplexity>]` runs in-process (no server); identity =
  model sha, control-vector sha, llama-perplexity sha, suite sha, data-file shas, base sha and the base-logits path.
  `bench_compare` on two such runs: PPL ratio, the arm's KL to the base, top-1 agreement, Δ accuracy with the tools'
  errors; verdict `preserves distribution` / `distribution shifted` only when the suite's thresholds are calibrated,
  `not calibrated yet` otherwise. `bench_table` gains the columns ppl, KL mean, top-1 agree, HS, WG, MMLU.
- Suite `engine/bench_suites/dist-shift.json`: perplexity on WikiText-2 test (100 chunks × 512 tokens), KL on the same
  text (16 chunks × 512: the saved base logits are ≈ 247 KB per token for the 248k vocabulary, 2.0 GB per base, kept in
  `data/bench/logits/<base_sha16>.bin`, git-ignored, written once), HellaSwag 400 tasks, Winogrande 400, MMLU 400 (ctx
  1024). Task subsets: the tool's own fixed selection (checked twice on the CPU: identical, independent of `-s`).
  Thresholds `kl_mean` / `same_top_p`: **null until calibrated**.
- Data in `data/bench/` (README with URLs, sizes, sha256, licences). **WikiText-2 is downloaded as parquet but not yet
  usable:** no parquet reader is installed and the original Salesforce zip is gone; `wikitext2_test.txt` needs
  `pip install pyarrow` (the driver's OK) or another text mirror, then `python -m engine.wsplang_bench write-suites` to
  pin its sha. The chain refuses to start until the suite's components all carry a sha.

## What it measures and what it does not
Measures: how far a model's next-token distribution on ordinary text sits from its base (KL mean / percentiles / max,
top-1 agreement), its perplexity on the same text, and three standard multiple-choice scores. Does not measure: any
behaviour that needs generation (closing, loops, answer format), the GSM8K / MATH gains themselves (those are the
chat suites), knowledge beyond MMLU's 400 items, or anything about a NanoWSP checkpoint (class `nanowsp`).

## Calibration rule (fixed now, thresholds set later from the numbers)
`T_kl` = the larger of (a) the KL of the same-norm placebo merged file against the base and (b) 2 × the KL between two
quantisations of the base if one is ever measured; `A` = the placebo's top-1 agreement. A real edit "preserves
distribution" if its KL mean ≤ T_kl and top-1 ≥ A, i.e. it moves the distribution no more than a random edit of the
same size does. The thresholds are written into the suite (which changes its digest) only after the first chain, with
the values and their source stated; until then every comparison says "not calibrated yet".

## First chain (`lab/runs/bench/chain_dist9b.sh`, not launched; needs the WikiText text first)
Arms on the 9B (programs/dist9b_*.wspl): `base` (hf_cache/h4/A0-Q6_K.gguf against its own logits: KL 0 by
construction, saves the logits once), `merged` (hf_cache/cc_op_v2/merged-Q6_K.gguf, provenance = the compile report),
`placebo_merged`, `vec_1.0` (base + cc_v2.gguf control vector at 1.0: llama-perplexity accepts `--control-vector-scaled`,
verified in `--help`), `placebo_vec_1.0`; then four comparisons and the table. GPU estimate (Q6_K 9B on the 3060,
prompt processing ≈ 1,500-2,500 tok/s): perplexity 51k tokens ≈ 0.5-1 min, KL 8k tokens < 0.5 min, HellaSwag 400 ×
4 endings ≈ 130k tokens ≈ 1.5-2 min, Winogrande 400 × 2 ≈ 0.5 min, MMLU 400 × 4 ≈ 1-1.5 min → **≈ 4-6 min per arm,
≈ 25-35 min for the five arms** plus one logits save.

## Prediction (row 48; before any run)
Premises: the 9B Q6_K is a strong base; the compiled edit is rank-1 per layer with ‖dW‖/‖W‖ ≈ 0.027 and was fit to
move the residual only where the conclude vector points; the same-norm placebo is the same size in a random direction
and lowered GSM8K dev from 50 to 26 (§43.19), so it disturbs the distribution more than the real edit does.
- Base perplexity on WikiText-2 test (ctx 512): **6.5-10**. HellaSwag acc_norm 0.72-0.84, Winogrande 0.66-0.78, MMLU
  (400) 0.62-0.78.
- `merged` vs base: KL mean **0.01-0.08**, top-1 agreement 92-98 %, PPL ratio 1.00-1.04; HS / WG / MMLU within ± 0.02.
- `placebo_merged` vs base: KL mean **0.05-0.30**, top-1 85-95 %, PPL ratio 1.02-1.15; HS / WG / MMLU −0.01 to −0.06.
- `vec_1.0` vs base: KL mean 0.02-0.15, top-1 88-97 %; `placebo_vec_1.0`: KL 0.05-0.30, top-1 82-95 %.
- **Primary: in both pairs the same-norm placebo shifts the distribution more than the real edit (KL mean placebo >
  KL mean real), 65 %.** Secondary: merged KL < 0.1 (70 %); the two placebos' KL within a factor 3 of each other (60 %).
- Surprised if: merged KL mean > 0.3 or top-1 < 80 %; any merged accuracy delta beyond ± 0.05; the placebo's KL is
  below the real edit's in both pairs; base perplexity outside 5-14.

## Two methods
Every number comes from llama-perplexity's own summary lines (parsed) and is re-read by hand from the raw logs the chain
keeps (`lab/runs/bench/chain_dist9b.log`, every command's full output); PPL and KL per chunk are also printed by the
tool, so the mean can be recomputed from the chunk rows when the run is read.
