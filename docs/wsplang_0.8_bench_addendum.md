# WSP-Lang 0.8 — § Benchmarks (the language's standard benchmark section, 2026-11-18)

This is a section of the WSP-Lang specification (`docs/wsplang_specification.md` §5.9 points here). Its purpose:
**to value any model objectively and the same way, the models WSP-Lang generates included.** One path, three ops,
suites shipped with the language, every run carrying the identity of what was measured.

## Ops

| op | kind | what it does |
|---|---|---|
| `bench_run` | read / report | runs a suite against a served model (LLM classes) or a checkpoint (`nanowsp`); writes one run JSON with identity, provenance, decoding, per-benchmark summaries and per-item records |
| `bench_compare` | read / report | item-by-item comparison of two runs of the same suite (same suite sha required); applies the verdict rule to an edit against its base; paired statistics only for two unrelated models |
| `bench_table` | read / report | registers runs and comparisons in `lab/runs/bench/index.json` and renders `docs/wsplang_benchmarks.md` |

`evaluate_benchmark` (0.6) now runs on the same code path: without arguments it is the `dev-2048` suite; other
`gsm8k=`, `math500=`, `max_tokens=` values build an ad-hoc suite on the same protocol. Both ops produce the same JSON
(test `test_evaluate_benchmark_and_bench_run_ops_give_the_same_shape`).

## Syntax

```
step bench_run suite=dev-2048 port=8095 arm=base out=lab/runs/bench/qwen9b_base.json
step bench_run suite=dev-2048 port=8095 arm=vec_0.35 out=lab/runs/bench/qwen9b_vec.json control_vector=hf_cache/cc/cc_v2.gguf scale=0.35
step bench_run suite=dev-2048 port=8095 arm=merged out=lab/runs/bench/qwen9b_merged.json provenance=hf_cache/cc_op_v2/report.json
step bench_run suite=dev-2048 port=8095 arm=bias6 out=lab/runs/bench/qwen9b_bias6.json extra_body={"logit_bias":[[248069,6.0]]}
step bench_run suite=nanowsp-wspx model=lab/runs/ternary_nanowsp/d64/model.safetensors arm=tern_d64 out=lab/runs/bench/nano_tern_d64.json
step bench_run suite=sealed-3 port=8095 arm=vec out=... force_reuse=true          # a spent sealed set: refused without force_reuse
step bench_compare base=lab/runs/bench/qwen9b_base.json arm=lab/runs/bench/qwen9b_vec.json placebo=lab/runs/bench/qwen9b_placebo.json base2=lab/runs/bench/qwen9b_base_run2.json out=lab/runs/bench/cmp_vec.json
step bench_table add=lab/runs/bench/qwen9b_vec.json model="Qwen3.5-9B Q6_K" add_compare=lab/runs/bench/cmp_vec.json
```

Arguments of `bench_run`: `suite` (a name resolved in `engine/bench_suites/`, or a file path), `arm` (label), `out`
(JSON), `port` (LLM suites) or `model` (nanowsp suites), `extra_body` (JSON merged into every request, e.g. a logit
bias; recorded), `lora`, `control_vector`, `scale` (declared extras: llama-server does not report them, so they are
recorded as declared by the caller with their file sha256), `provenance` (manifest, see below), `binary` (llama-server
path; default `C:/Users/Anto/llamacpp/bin/llama-server.exe`), `force_reuse` (spent sealed suite), `unknown_model`
(record an unidentified server instead of refusing).

## Suites (`engine/bench_suites/*.json`, field `wsplang_bench_version: "0.8"`)

| suite | class | items | decoding | preservation |
|---|---|---|---|---|
| `dev-2048` | llm-chat-think | GSM8K test 0-99 + MATH-500 `select_indices(2026, 50)` (bench_runner's defaults) | greedy, seed 0, 2 048 tokens, 2 slots, the model's chat template, the boxed-answer suffix | HumanEval 164, 40 local instruction checks, garbage rates |
| `dev-8192` | llm-chat-think | same items | 8 192 tokens | same |
| `sealed-1`, `sealed-2`, `sealed-3` | llm-chat-think | wrappers over `data/sealed_confirm{,2,3}.json` (100+50, 100+50, 200+100); the sealed file's own sha is verified at load | 2 048 tokens | same |
| `nanowsp-wspx` | nanowsp | `lab/data/wspx_{recall,parity8,mod13,fsm4,noecho}.json`, train and holdout splits, pinned by file sha256 | one character, greedy, in-process | none |

A suite's `sha256` is computed over its name, class, think flag, items (or the sealed file's sha), decoding, graders and
preservation; `load_suite` refuses a file whose digest does not match (hand edits are visible). A sealed suite whose
`use` field says it is spent is refused unless `force_reuse=true`, which is then recorded in the run.

Model classes: `llm-chat-think` (a reasoning template; the graders read the final answer after `</think>`),
`llm-chat` (no thinking; `think=false`; the same graders on the plain answer), `nanowsp` (a safetensors checkpoint, or
a checkpoint directory with `model.safetensors`, run in-process with `engine/nanowsp_runtime.py`; a ternary checkpoint
from `ternarize` / `imprint_ternary` loads the same way). `bench_run` refuses a class mismatch (a nanowsp suite needs
`model=`, an LLM suite needs a server).

## Identity and provenance

Every run JSON has `identity: {model_path, model_sha, extras, binary: {path, sha256}, suite_sha, decoding}`. The model
path comes from llama-server's `/props` (fallback `/v1/models`); its sha256 is computed when it is a local file. If no
path can be established, the run is refused unless `unknown_model=true`. For the nanowsp class the identity is the
checkpoint file's sha256 and the runtime file's sha256.

`provenance=<manifest>` makes a WSP-Lang-built model a first-class object: the manifest is normalised to
`{kind, base_sha, output_sha, patches, manifest_path, manifest_sha}` (kinds: `wsp_edit` from a `wsp.py` GGUF manifest,
`compile_control_vector` from the op's report, `edit_column` / `edit_neurons` manifests, `weightpatch` and
`patch_bundle` from `engine/wsplpatch.py`). When both the manifest's output sha and the served sha exist they must
match (a 32-hex fingerprint prefix matches its full sha), otherwise the run is refused. The table renders provenance as
`base@<sha8> + N patches (kind)`; a run without provenance is `external`.

## Comparison and verdict

`bench_compare` pairs items by `(bench, i)`, counts wins and losses per grader with the exact two-sided binomial test,
and reports per benchmark: n, base, arm, win, loss, p, items the base got right and finished that the arm lost, items
gained, cuts, tokens, the arm's no-think share; the preservation battery is compared with `engine.preserve_bench.compare`
when both runs carry it. `base2=` gives the **noise band**: two runs of the same configuration, their discordant count
and |Δ correct| per benchmark (the 0.8B harness showed 11 discordant of 100 between identical runs, results §45.22).

**Relation.** *edit vs its base* when the arm's provenance names the base run's model sha, or when both runs serve the
same file and differ only in extras (control vector, LoRA, request fields). Otherwise *two models*.

**Verdict rule (fixed 2026-11-18; edit vs its base only).** *improves* only if (a) on at least one benchmark the arm
beats the base with p < 0.05 on both graders, (b) the placebo arm, when given, does not *gain* significantly on any
benchmark / grader (a same-norm placebo that harms the base, as on the 9B, does not block: it says the direction
matters), (c) when a band is given, |arm − base| ≥ 2 × band on that benchmark, (d) the preservation battery, when both
runs carry it, does not fail. *harms* is the symmetric statement. Otherwise *same / not decidable*, with the reasons
listed. For *two models* the output carries the paired statistics and the band and the words "two models: paired
statistics only": a difference between two unrelated models is a description, not a verdict about an edit.

## The `llm-logits` class: `dist-shift`, the label-free component (2026-11-18)

A GGUF file measured by llama.cpp's own `llama-perplexity`, no server, no sampling (`engine/dist_shift.py`,
`lab/probes/dist_shift/PLAN.md`): perplexity on WikiText-2 test (100 chunks × 512), **KL divergence of the model against
the base's saved logits** over the same text (16 chunks; `--save-all-logits` writes the base file once into
`data/bench/logits/<base_sha16>.bin`, ≈ 2 GB per base, git-ignored; `--kl-divergence --kl-divergence-base` reads it),
top-1 agreement, and HellaSwag / Winogrande / MMLU multiple choice (400 tasks each; the task subset is the tool's own
fixed selection, independent of `-s`; the MMLU file is binary and goes in with `-bf`).

```
step bench_run suite=dist-shift model=hf_cache/cc_op_v2/merged-Q6_K.gguf base=hf_cache/h4/A0-Q6_K.gguf arm=merged out=lab/runs/bench/dist9b_merged.json provenance=hf_cache/cc_op_v2/report.json
step bench_run suite=dist-shift model=hf_cache/h4/A0-Q6_K.gguf base=hf_cache/h4/A0-Q6_K.gguf arm=vec_1.0 out=... control_vector=hf_cache/cc/cc_v2.gguf scale=1.0
step bench_compare base=lab/runs/bench/dist9b_base.json arm=lab/runs/bench/dist9b_merged.json out=lab/runs/bench/cmp_dist9b_merged.json
```

Identity: model sha, control-vector sha, `llama-perplexity` sha, suite sha, the data files' shas, the base sha and the
logits path. `bench_compare` on two such runs gives the PPL ratio, the arm's KL (mean, p99, max), top-1 agreement and the
accuracy deltas with the tools' own errors; the relation is "edit vs its base" when the arm's KL base is the base run's
model. **Verdict:** `preserves distribution` if KL mean ≤ `thresholds.kl_mean` and top-1 ≥ `thresholds.same_top_p`,
`distribution shifted` otherwise, and **`not calibrated yet` while the suite's thresholds are null** (they are set from
the same-norm placebo's own KL and top-1 after the first chain, as the PLAN fixes). Table columns: ppl, KL mean, top-1
agree %, HS, WG, MMLU. Data files and licences: `data/bench/README.md`. Tests: `tests/test_dist_shift.py`.

## What this section does not do

- It is **not the sealed protocol** (`lab/probes/sealed/README.md`): a sealed suite is single use, its result is read
  whatever it says, and the pre-registered criteria C1-C3 of the plan decide the claim; `bench_compare` only counts.
- It measures **no knowledge** and no real instruction following: IFEval and TriviaQA need a download the driver has not
  approved; the compliance set is 40 home-made prompts (`engine/preserve_bench.py`).
- Sampling, other budgets and other item sets are **separate suites**, never silent changes to a run.
- A run is comparable only with a run of the same suite sha; the table shows runs of different suites side by side
  but `bench_compare` refuses to pair them.

## Reproducing a row

Serve the file whose sha256 the row shows (the `identity` field of the run JSON has the full sha and path, and the
extras with their shas), with the llama-server flags of the decoding (`-c 16384 -np 2 -fa on --jinja`, greedy is in the
request), run `bench_run` with the same suite name, and `bench_compare` against the row's base run. The per-item
records of both runs are in their JSONs.

## Tests

`tests/test_wsplang_bench.py` (suites, digest, resolution by name; bench_run on a fake server, extra_body, spent sealed,
evaluate_benchmark same shape; identity refusal, declared extras, provenance mismatch and match, fingerprint prefix
match, class mismatch, think=false grading; the nanowsp suite on a tiny checkpoint built with the runtime's own
imprint, and on a checkpoint directory; compare counts and p, verdict cells, noise band, two-models relation; registry
and table; op registration), plus `tests/test_wsplang_spec.py` (ops table and version in sync).
