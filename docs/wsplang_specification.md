# WSP-Lang: Declarative Language Specification for Weight-Space Programming

**Version:** 0.8.3 (= `engine/wsplang.py` `__version__`; this file is the base text, the per-version additions are in `docs/wsplang_0.2_spec.md` ... `docs/wsplang_0.8_spec.md`, and §5.0 lists every op the runtime accepts)  
**Status:** Living Specification  
**Authors:** FiveTechSoft Engineering & Research Team  
**Repository:** [github.com/FiveTechSoft/weights](https://github.com/FiveTechSoft/weights)  
**License:** MIT License  

---

## 1. Executive Summary & Philosophy

**Weight-Space Programming (WSP)** reads, compares and edits the weights of LLMs directly, as an alternative or a complement to changing them by gradient descent. Its working hypothesis, tested case by case and not assumed, is that parts of what a network does can be located in, read from and written into its weight matrices. What has and has not held up is in the README's status table and `results.md`.

**WSP-Lang** is the domain-specific declarative programming language designed to query, decompile, steer, and edit neural network weights in quantized formats (such as GGUF `Q6_K` and `BF16`) with:
1. **No forward pass** for reading weights; every claim about behaviour still needs one (a benchmark run).
2. **Deterministic, falsifiable hypothesis definitions** (every program declares its hypothesis, budget, and validation threshold).
3. **Hardware safety & physical barriers** (`FORBIDDEN_TENSORS` and pre-execution validation gates).
4. **Machine-agent interoperability** via native Model Context Protocol (FastMCP) bindings.

---

## 2. System Architecture

The WSP-Lang execution pipeline guarantees that no weights on disk can be modified unless a program passes syntactic validation, semantic constraint checks, and safety budget verification:

```mermaid
flowchart TD
    A["WSP-Lang Source (.wspl or inline code)"] --> B["Parser & Lexer (parse_args_lines)"]
    B --> C["AST Generation"]
    C --> D{"Strict Validation Gate"}
    D -- "Errors > 0" --> E["Execution Halted (status: aborted)"]
    D -- "Valid (Errors == 0)" --> F["Plan Generator (plan_run)"]
    F --> G["Safety Sandbox (Budget & FORBIDDEN_TENSORS check)"]
    G -- "Violates Policy" --> H["Security Rejection (no disk writes)"]
    G -- "Compliant" --> I["Execution Engine (execute_plan)"]
    I --> J["GGUFReader / Requant Engine (Q6_K block requant)"]
    J --> K["Manifest Generation (*.manifest.json)"]
    J --> L["Output Artifact / Candidate GGUF"]
```

---

## 3. Formal Syntax & EBNF Grammar

A WSP-Lang source file (`.wspl`) is organized into structured blocks using clean, Pythonic indentation:

```ebnf
Program         ::= ExperimentHeader Block*
ExperimentHeader::= "experiment" Identifier ":"
Block           ::= MetadataBlock | BudgetBlock | SemanticBlock | StepsBlock | HoldoutBlock

MetadataBlock   ::= "model:" String
                  | "objective:" String
                  | "hypothesis:" String

BudgetBlock     ::= "budgets:" Newline (Indent Key "=" Value)+

SemanticBlock   ::= "semantic_areas:" Newline (Indent Key "=" Value)+

StepsBlock      ::= "steps:" Newline (Indent StepStatement)+
StepStatement   ::= "step" Operation Arguments*

HoldoutBlock    ::= "holdout:" Newline (Indent Key "=" Value)+
```

### Lexical Conventions
- **Comments:** Lines beginning with `#` are comments and are ignored by the parser.
- **Indentation:** 4 spaces (or consistent tabs).
- **Identifiers:** Alphanumeric characters plus underscores and periods (`blk.19.attn_output.weight`).
- **Numbers:** Integers (`100`), floats (`+0.0275`, `1.5e-4`), or ratios (`1/2`).

---

## 4. Program Structure & Sections

### 4.1. Header & Experiment Declaration
Every program begins with a unique experiment identifier:

```wspl
experiment inspect_attention_logic:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: map_semantic_geometry
    hypothesis: "Layer 19 Attention Head 0 projects into the Math & Logic residual stream"
```

### 4.2. Safety Budgets (`budgets:`)
Defines the strict physical bounds that the execution engine is permitted to touch:

```wspl
budgets:
    max_norm_delta = 0.05
    max_modified_weights = 16384
    allow_write = false
```

### 4.3. Semantic Taxonomy (`semantic_areas:`)
Configures the differential centroid projection engine:

```wspl
semantic_areas:
    mode = differential
    target = math_logic
    contrast = centered
```

### 4.4. Execution Steps (`steps:`)
The sequence of atomic declarative operations executed by the runtime.

---

## 5. Instruction Set Architecture (Primitives)

WSP-Lang 0.1 implements five core operations designed for mechanistic interpretability and surgical intervention:

### 5.0. Ops the runtime accepts (checked by `tests/test_wsplang_spec.py`)
A step whose op is not in this table is rejected by the validator. *read / report* ops never write weights (reports, signatures, a new adapter directory); *writes weights* ops need the safetensors+torch backend and pass the write guard (§6.1b); *stub* ops return a placeholder and support no claim.

<!-- ops:begin -->
| op | kind | documented in |
|---|---|---|
| `apply_patch` | writes weights | `wsplang_0.8_spec.md` |
| `architectural_drift` | read / report | §5.10 (runtime contract) |
| `assert_behavioral_drift` | read / report | `wsplang_0.7_spec.md` |
| `assert_davis_kahan` | writes weights | `wsplang_0.4_spec.md` |
| `assert_orthogonal_preservation` | writes weights | `wsplang_0.3_spec.md` |
| `assert_weyl` | writes weights | `wsplang_0.4_spec.md` |
| `audit` | writes weights | `wsplang_0.2_spec.md` |
| `bench_compare` | read / report | `wsplang_0.8_bench_addendum.md` |
| `bench_run` | read / report | `wsplang_0.8_bench_addendum.md` |
| `bench_table` | read / report | `wsplang_0.8_bench_addendum.md` |
| `commit_manifest` | read / report | `wsplang_0.8_spec.md` |
| `compare` | read / report | `wsplang_0.2_spec.md` |
| `compile_control_vector` | read / report | this file |
| `compile_synthetic` | writes weights | `wsplang_0.7_spec.md` |
| `contrast_vector` | read / report | this file |
| `cov_imprint` | writes weights | `wsplang_0.5_spec.md` |
| `cross_project` | writes weights | `wsplang_0.6_spec.md` |
| `decompile` | writes weights | §5.10 (runtime contract) |
| `decoupled_delta` | writes weights | `wsplang_0.5_spec.md` |
| `delta_compare` | read / report | this file |
| `delta_signature` | read / report | this file |
| `delta_stats` | read / report | this file |
| `delta_to_lora` | read / report | this file |
| `delta_unlearn` | writes weights | this file, `wsplang_0.3_spec.md`, `wsplang_0.4_spec.md` |
| `diff` | read / report | `wsplang_0.2_spec.md` |
| `drift_profile` | read / report | §5.10 (runtime contract) |
| `edit_column` | writes weights | `wsplang_0.8_spec.md` |
| `edit_neurons` | writes weights | `wsplang_0.8_spec.md` |
| `evaluate` | read / report | this file |
| `evaluate_benchmark` | read / report | this file |
| `export_control_vector` | writes weights | this file, `wsplang_0.2_spec.md` |
| `fwp_overlay` | writes weights | `wsplang_0.3_spec.md` |
| `imprint_association` | writes weights | §5.10 (runtime contract) |
| `imprint_batch` | writes weights | §5.10 (runtime contract) |
| `imprint_corpus` | writes weights | `wsplang_0.7_spec.md` |
| `imprint_ternary` | writes weights | `wsplang_0.8_ternary_addendum.md` |
| `init_architecture` | writes weights | `wsplang_0.7_spec.md` |
| `jepa_steer` | writes weights | `wsplang_0.7_spec.md` |
| `layer_envelope` | writes weights | `wsplang_multilayer_envelope_spec.md` |
| `leace_erase` | stub | `wsplang_0.5_spec.md` |
| `list_patches` | read / report | `wsplang_0.8_spec.md` |
| `memit_imprint` | writes weights | `wsplang_0.5_spec.md` |
| `model_report` | read / report | this file |
| `adapter_report` | read / report | this file |
| `quantize` | writes weights | this file |
| `multi_layer_envelope` | writes weights | §5.10 (runtime contract) |
| `peek` | read / report | this file, `wsplang_0.2_spec.md` |
| `permute_align` | writes weights | `wsplang_0.7_spec.md` |
| `project_in` | writes weights | this file, `wsplang_0.2_spec.md`, `wsplang_0.3_spec.md` |
| `project_out` | writes weights | this file, `wsplang_0.2_spec.md`, `wsplang_0.4_spec.md` |
| `rollback_last` | read / report | `wsplang_0.8_spec.md` |
| `rosetta_align` | writes weights | `wsplang_0.6_spec.md` |
| `semantic-areas` | read / report | §5.10 (runtime contract) |
| `semantic-map` | read / report | §5.10 (runtime contract) |
| `semantic_areas` | read / report | `wsplang_0.2_spec.md` |
| `semantic_delta` | read / report | §5.10 (runtime contract) |
| `semantic_map` | read / report | this file, `wsplang_0.2_spec.md` |
| `spectral_audit` | writes weights | `wsplang_0.4_spec.md` |
| `spectral_decay` | writes weights | `wsplang_0.3_spec.md` |
| `spectral_delta` | read / report | §5.10 (runtime contract) |
| `ssm_dynamics_diff` | read / report | §5.10 (runtime contract) |
| `subspace_transfer` | writes weights | this file |
| `synthesize` | writes weights | `wsplang_0.7_spec.md` |
| `ternarize` | writes weights | `wsplang_0.8_ternary_addendum.md` |
| `ternary_survival` | read / report | `wsplang_0.8_spec.md` |
| `tie_weights` | writes weights | `wsplang_0.7_spec.md` |
| `train` | writes weights | §5.10 (runtime contract) |
| `vicreg_preserve` | read / report | `wsplang_0.7_spec.md` |
| `probe_columns` | read / report | §5.11 |
| `probe_neurons` | read / report | §5.11 |
| `rank_candidates` | read / report | §5.11 |
<!-- ops:end -->

### 5.1. `step semantic_map`
Queries the model's global semantic taxonomy or calculates the functional projection of a specific tensor.

* **Syntax:**
  ```wspl
  step semantic_map [tensor=<tensor_name>] [top=<k>]
  ```
* **Parameters:**
  * `tensor`: Optional. The target GGUF tensor to project (e.g. `blk.14.ffn_down.weight`). If provided, areas are ranked by actual cosine projection:
    $$\text{score}(T, \mathbf{C}'_a) = \frac{\bar{\mathbf{v}}_T \cdot \mathbf{C}'_a}{\|\bar{\mathbf{v}}_T\| \|\mathbf{C}'_a\|}$$
  * `top`: Number of top-scoring areas to return. If omitted, all 10 areas are returned.
* **Output:** Ranked dictionary with dominant area, scores, and representative affine seeds.

---

### 5.2. `step peek`
Performs surgical, read-only decompilation of a single weight, row, column, or tensor slice.

* **Syntax:**
  ```wspl
  step peek <tensor> [row=<r>] [col=<c>] [top=<k>]
  ```
* **Parameters:**
  * `<tensor>`: Complete GGUF tensor name.
  * `row=<r>`: Target row (residual stream target dimension).
  * `col=<c>`: Target column (input channel / internal head dimension).
  * `top=<k>`: Number of semantic affinities to decompile.
* **Architectural Decompilation:**
  * For attention projections (`attn_output.weight`), `peek` automatically computes the attention head index:
    $$\text{head\_idx} = \left\lfloor \frac{\text{col}}{\text{head\_dim}} \right\rfloor$$
  * For FFN down projections (`ffn_down.weight`), `peek` correlates intermediate activation neurons with the residual target stream.

---

### 5.3-5.5. Single-weight edits are a CLI, not WSP-Lang steps
Earlier versions of this file listed `poke`, `scale` and `steer` as steps. The runtime never accepted them (the validator
rejects them). What exists:
* `python wsp.py poke <gguf> <tensor> -r <row> -c <col> -v <value> [-o <out>]`: one scalar of a Q6_K tensor, requantizing
  only its 256-value block (`quantize_row_q6_K` NumPy port, 87 % byte-identical on unedited blocks, error ~6e-6).
* Column scaling of a GGUF: `python wsp.py edit <gguf> <prog> -o <out>` with an edit manifest (e.g. `edits/evil_mimo.json`).
  There is no steering-vector step; llama.cpp control vectors do that at run time.
`FORBIDDEN_TENSORS` (§6.1) applies to both.

---

### 5.6. Studying improved models (added 2026-11-06)
Ops to learn what a fine-tune changed and to move another model along it. Model arguments are directories or files
of safetensors; multimodal (`model.language_model.*`) and text-only (`model.*`) Qwen3.5 names are lined up. Tensors
are read one at a time (a 9B model never sits in memory twice). Reports and signatures are never weight files and
never land inside an input model.

```wspl
step delta_stats base=<model> model=<model> [tensors=<glob>] [out=<report.json>]
step delta_signature base=<model> model=<model> out=<sig.pt> [k=16] [sketch=4096] [tensors=<glob>]
step delta_compare a=<sig.pt> b=<sig.pt> [factor=3] [out=<report.json>]
step evaluate_benchmark endpoint=<port|URL> name=<label> [gsm8k=100] [math500=50] [max_tokens=8192] [out=<report.json>]
step subspace_transfer tensors=<glob> source=<model> basis=sig:<sig.pt>|shared:<a.pt>,<b.pt>|random k=16 alpha=1.0 [seed=0] out=<dir>
step model_report model=<dir|repo id> [base=<dir|repo id>] [rows=128] [card=true] [nonweights=true] out=<report.md>
step adapter_report adapter=<dir|hf:owner/repo> [base=<dir|repo id>] [rows=128] [workers=16] out=<report.md>
step quantize model=<gguf> type=<Q2_K|IQ2_XS|IQ2_XXS|Q3_K_S|...> out=<gguf> [imatrix=auto|<file>] [calib=<text>] [tensor_type=<name=type,...>] [allow_requantize=true] [leave_output=true] [chunks=100]
step delta_to_lora base=<model> model=<model> out=<adapter dir> [rank=16] [kinds=mlp,self_attn,linear_attn] [gguf=<file.gguf>]
```

* `delta_stats`: per tensor, ||ΔW||/||W||, cosine, changed fraction, spectrum of ΔW (top-1 energy, stable rank),
  moved embedding rows; totals per tensor family and per layer.
* `delta_signature`: per tensor, the top-k singular vectors of ΔW on its smaller side, its alignment with the base
  weight's own top-k subspace, and a count-sketch of ΔW, so pairs can be compared after the weights are deleted.
* `delta_compare`: per tensor type, same-layer subspace overlap against the same type at other layers (structured
  null) and against chance k/d; `H3` true when both ratios are >= `factor`.
* `evaluate_benchmark`: GSM8K / MATH-500 with one fixed protocol (greedy, the model's chat template, the boxed
  answer graded by two independent graders) against a server that is already running (e.g. llama-server).
* `delta_to_lora`: the change (model - base) of every linear decoder matrix as a PEFT LoRA adapter, by exact SVD
  truncated to `rank` (lora_alpha = rank, so scale 1 = the rank-r part of the whole change). `energy.json` gives per
  matrix the share of ||ΔW||² kept at ranks 1, 4, 16, 64, 256, 512, 1024, and the result its ||ΔW||²-weighted mean:
  read it before evaluating (a change that a small rank keeps is one a small adapter can carry). `gguf=` also runs
  llama.cpp's convert_lora_to_gguf.py (env `WSP_LORA_CONVERTER`, python `WSP_CONVERT_PYTHON`; `base` must be a Hugging
  Face dir). llama-server then applies it on the base GGUF with a per-request scale (`--lora-init-without-apply`,
  `"lora": [{"id": 0, "scale": s}]`). Writes a new adapter dir; refuses one inside an input model or one that exists.
  Measured use (results.md §43): Jackrong's DeepSeek-V4 distill of Qwen3.5-9B as a rank-16 adapter (87 MB) at
  scale 0.5 takes the base from 6/36 to 32/36 on the short-budget MATH probe and from 9/48 to 38/48 on GSM8K.
* `model_report`: how a model differs from its base and what is usable, in under a minute for a 9B model read from
  Hugging Face without downloading it. Reads one block of rows per tensor (16 HTTP ranges in flight), classifies
  the change (`identical`, `directional_ablation` with layer range and global or per-layer directions, `lora_like`
  with modules and rank, `full_fine_tune`), shows the card's declared base, license and method words (not trusted),
  the closest studied models of the delta atlas, and what can be reused and how. `base` defaults to the card's
  base_model. Validated on 6 models whose change is known (lab/runs/model_report/validation.json).
  Outside the weights (`nonweights=true`): config (real changes vs cosmetic: library version, moved keys, absent
  vs None), tokenizer (vocab, merges, added tokens, normalizer, pre-tokenizer, decoder), chat template, and an
  encode test on 14 scripts that names where text tokenizes differently from the base (findings F25).
  Since 0.8.2 `model=` may be a local `.gguf` file (`engine/gguf_reader.py`): it is read with HF names and layout
  (llama.cpp's Qwen3.5 conversion undone: norms stored as w + 1, `ssm_a = −exp(A_log)`, interleaved value heads,
  fused or non-contiguous blocks), BF16 / F16 / F32 as stored and quantised types dequantised per row block, so a
  BF16 GGUF against its safetensors base compares exactly and a Q6_K one reads with a 1.8 % floor. Tensors with no
  HF counterpart appear as hidden parts (e.g. SenseNova's `mhc`); the config / tokenizer diff is skipped for a GGUF.
* `quantize` (0.8.3): extreme quantisation through llama.cpp's `llama-quantize`, with `imatrix=auto` running
  `llama-imatrix` on WikiText-2 *train* (never the test split `dist-shift` measures) for the IQ2 / IQ1 types, which
  refuse to run without one. Per-tensor overrides (`tensor_type=ffn_down=q6_k,...`) keep chosen tensors at higher
  precision. Writes `<out>.tmp` and renames on success; refuses an existing output or one equal to the input;
  `<out>.manifest.json` records input and output sha256, type, bits per weight, imatrix and calibration sha256, the
  exact command and caveats (`allow_requantize=true` from a Q6_K source is recorded as worse than from BF16).
  `bench_run provenance=<manifest>` links the quantised model to its base (`read_provenance` kind `quantize`). No
  behaviour claim: the cost of the bits is measured by `bench_run` on `dist-shift` and `rank-2048`.
* `adapter_report` (0.8.2): a PEFT LoRA repo read from the adapter alone, when no merged model exists (e.g.
  `harness-hub/Qwen3.8-27B-pi-adapted`, 640 MB on a 27B). From `adapter_config.json` and the `lora_A` / `lora_B`
  factors: r, alpha, scale (rslora aware), modules and layers touched, families untouched (embeddings, head, norms,
  linear attention), modules outside the stated targets, per module ‖B·A‖ two ways (trace identity and singular
  values), effective rank at 90 % energy, the leading output-side direction and its consistency across layers (a
  global direction gives ~1, per-layer updates ~0); with `base=`, ‖ΔW rows‖ / ‖W rows‖ on the same middle block of
  rows `model_report` uses, read from the base by one range per tensor. Parameter count from the header and from the
  shapes must agree. Nothing about behaviour: where and how large, not what it does. Report only; `lab/probes/adapter_report/NOTES.md`.
* `subspace_transfer`: W' = W + α · P_S(W_source − W_base), the projection on the smaller side. Writes a complete
  model to `out=` (unchanged shards copied, configs copied); every changed shard goes through the write guard with a
  total budget; protected tensors need `unlock_tensors`; on failure the partial output is removed. Provenance
  `synthetic_transfer`, with the source, basis, k and α, is stored in each shard: the op runs no gradient, but the
  update it moves along came from a trained model.

---

### 5.7. `step contrast_vector` (added 2026-11-09)
A llama.cpp control vector from two sets of texts, usually the same prompts answered in two styles (e.g. happy vs
neutral). Rows are `{"prompt", "response"}` jsonl; each goes through the model's chat template with no system prompt,
and the residual stream is averaged over the response tokens per layer. The file holds the difference pos − neg as
per-layer increments (llama.cpp adds `direction.N` at the output of every layer N, so a cumulative difference would be
summed ~N times). It is used at run time (`llama-server --control-vector-scaled <out>:<scale>`): an additive push
cannot be stored in Qwen3.5's bias-free weights. `placebo=<seed>` also writes a vector with the same norm per layer in
random directions, the control for "any push of this size".

```wspl
step contrast_vector model=<hf dir> pos=<pos.jsonl> neg=<neg.jsonl> out=<vector.gguf> [placebo=<seed>] [max_resp=384]
```

### 5.8. `step compile_control_vector` (added 2026-11-11)
A control vector (e.g. from `contrast_vector`) written into the weights: for each layer N it carries, a rank-1
closed-form edit dW_N = v_N u_Nᵀ of `mlp.down_proj`, with u_N the ridge least-squares solution of uᵀx ≈ 1 over that
layer's `down_proj` inputs on calibration traces (`{"prompt", "reasoning"}` jsonl, the model's own text), each layer
fitted on the stream already edited below it. Every run also compiles a same-norm random placebo (given, or generated
with `placebo_seed`). Gate G0 on held-out traces: median over layers of the per-layer median |uᵀx − 1| ≤ 0.5; if it
fails, only the report is written. Then: rank-1 PEFT LoRA per arm (GGUF adapter too when `WSP_LORA_CONVERTER` is set)
and, with `base_gguf=` (Q6_K `ffn_down` only), merged copies `merged-Q6_K.gguf` / `placebo_merged-Q6_K.gguf` in `out`.
`out` must be a new directory outside the input model. The op does not run benchmarks: the effect is shown on a sealed
set with the placebo arm, planned before the run (results §43.19-21). Plan: `lab/probes/cc_weights/OP_PLAN.md`.

```wspl
step compile_control_vector model=<hf dir> vector=<cv.gguf> calib=<traces.jsonl> out=<new dir> [placebo=<cv.gguf> | placebo_seed=2026] [base_gguf=<Q6_K gguf>] [lora=1] [lam=1e-3] [heldout=28] [seed=20261111] [max_len=4096] [template=think|plain] [device=cuda|cpu]
```

---

### 5.9. Benchmarks (added 2026-11-18; full text in [`wsplang_0.8_bench_addendum.md`](wsplang_0.8_bench_addendum.md))

The language carries its own standard benchmarks so that any model, the ones WSP-Lang produced included, is measured
and compared the same way. A **suite** is a digest-verified JSON shipped in `engine/bench_suites/` (`dev-2048`,
`dev-8192`, `sealed-1/2/3` as single-use wrappers over `data/sealed_confirm*.json`, `nanowsp-wspx` for the NanoWSP
class); it fixes the items, the decoding, the model class (`llm-chat-think`, `llm-chat`, `nanowsp`), the graders and the
preservation battery. `bench_run suite=<name> arm=<label> out=<json> port=<n> | model=<ckpt>` executes it and writes
one JSON with the served model's **identity** (path + sha256, declared extras such as a control vector or LoRA, the
llama-server binary sha, the suite sha) or refuses (`unknown_model=true` records an unidentified server instead), and
with the model's **provenance** when given (`provenance=<manifest>`: a wsp.py GGUF manifest, a `compile_control_vector`
report, an `edit_column` manifest or a `commit_manifest` bundle, verified against the served sha).
`bench_compare base= arm= out= [placebo=] [base2=]` applies the fixed verdict rule to an edit against its base and
reports paired statistics only for two unrelated models; `bench_table` renders the registry to
`docs/wsplang_benchmarks.md`. `evaluate_benchmark` runs on the same path (its defaults are the `dev-2048` suite).


### 5.10. Additional runtime contracts

These operations are classified according to the runtime table above, including guarded artifact creation. None establishes a causal behavioral effect without independent evaluation, placebo and preservation controls. Relative paths are resolved from the repository root unless otherwise specified.

| Operation | Accepted arguments and behavior |
|---|---|
| `architectural_drift`, `drift_profile` | `base=<path> [family=<filter>] [top=10]`; base may also be positional; `filter` aliases `family`. Compares current target with base by architectural family. |
| `spectral_delta` | Tensor positional or `tensor=<name>`, `[base=<path>] [k=5] [out=<vector path>]`; `save_vector` aliases `out`. Computes spectral delta; optional vector export is not a JSON report. |
| `semantic_delta` | Tensor positional or `tensor=<name>`, `[base=<path>] [top=5]`. Computes semantic delta for one tensor, not a tensor glob. |
| `ssm_dynamics_diff` | `[layer=1] [base=<path>]`; numeric positional layer also accepted. Compares SSM dynamics for one layer. |
| `semantic-map` | Alias of `semantic_map`, `[top=3]`. Lists semantic areas and centroid availability; with SafeTensors context it uses the default GGUF for centroids. |
| `semantic-areas` | Alias of `semantic_areas` / peek semantic-area handler; `[tensor=<name>] [row=<int>] [col=<int>] [k=8] [top=3]`. SafeTensors path reports statistics and optional element, not centroid scores. |
| `imprint_association` | `prompt=<text> target=<text> [layer=1] [eta=5.0] [out=<new checkpoint>]`. NanoWSP/NanoWSP2 checkpoint required. Multi-character target is expanded to successive prefix/character pairs. Uses write guard. |
| `imprint_batch` | `file=<JSON array> [layer=1] [eta=5.0] [out=<new checkpoint>]` or `pairs="p->t;p2->t2"`. JSON entries have prompt and target keys. Not JSONL. Uses write guard. |
| `decompile` | `method=structural|behavior arch=<wsparch> model=<checkpoint> out=<program>`. Behavior requires `data=<corpus> order=<int>`, optional `limit_tokens`. Structural needs vocab_chars metadata. Produces text, not a new weight checkpoint; classified as write operation by runtime. Behavior averages predictions per context; not recovery of arbitrary learned algorithms. |
| `multi_layer_envelope` | Alias of `layer_envelope`; `[layers=10..22] [peak=<int>] [bandwidth=2.5] [envelope=gaussian] [alpha=0.04] [mode=boost] [vector=<path>] [out=<control GGUF>] [n_layers=32]`. Also accepts key and out_gguf. Export uses vector times envelope (not alpha); alpha/mode are reported metadata. Does not edit base weights. Export requires allow_write=true, a new .gguf path outside protected model directories, and non-dry-run execution. |
| `train` | `arch=<wsparch> data=<corpus> out=<new checkpoint>` plus training and architecture overrides. Requires budget allow_training=true; with allow_write disabled it only constructs the model and reports simulated. Actual training uses backward and is not donor-free analytic synthesis. |

Comparative operations without an explicit base use the runtime default Qwen base GGUF. Prefer explicit paths for reproducibility. Optional spectral-vector export requires allow_write=true, a new .npz path outside protected model directories, and non-dry-run execution. These contracts document current behavior, not an endorsement of unguarded exports.

Minimal examples (paths and tensor names must match the selected model):

```wspl
step spectral_delta tensor=blk.19.attn_output.weight base=models/base.gguf k=5
step semantic_delta tensor=blk.19.attn_output.weight base=models/base.gguf top=5
step ssm_dynamics_diff layer=1 base=models/base.gguf
step imprint_batch file=data/pairs.json layer=1 eta=5.0 out=models/imprinted.safetensors
```

### 5.11. Causal locus discovery (experimental NanoGPT backend)

```wspl
step probe_columns model=models/nano.safetensors tensor=transformer.h.*.mlp.c_proj.weight eval=data/probe.json metric=task_acc placebo=true n_probe=64 out=lab/probe.json
step probe_neurons model=models/nano.safetensors tensor=transformer.h.*.mlp.c_fc.weight eval=data/probe.json placebo=true out=lab/neurons.json
step rank_candidates from=lab/probe.json out=lab/ranked.json
```

The model must be a NanoGPT SafeTensors checkpoint with architecture metadata. Only MLP matrices are eligible. The evaluation JSON contains three nonempty arrays named discovery, heldout and preservation; each entry has ids (a list of token IDs) and target (one token ID). Duplicate input sequences are rejected across all splits. This is not a semantic independence guarantee.

Optional probe arguments: n_probe (64, maximum 256), dose (0.5), seed (1337), top (5). The intervention attenuates a column or row by dose; a seeded same-location random perturbation is matched to its actual norm within 2%. Selection uses discovery and preservation only; heldout accepts or rejects the fixed shortlist. The metric is next-token exact accuracy (higher is better). The default per-intervention weight budget is 4096; a stricter plan budget applies. max_delta_norm is not supported and is rejected.

No checkpoint is written. Reports are new JSON files, including with allow_write=false; dry_run does not create them. Probe reports include axis, model and data hashes, split sizes, controls, baseline, candidates and shortlist. The rank operation preserves discovery ordering, and returns no_locus if none pass. These are screening results, not a sealed behavioral proof. The rank report is not yet an authenticated edit authorization.

### 5.12. Retired operations and certified erasure

`surprise_fwp` is retired: lexical repetition is not neural surprisal or a Titans memory update. The validator and runtime reject it. Historical reports and the 0.5 specification are not current executable contracts.

`leace_erase` now requires explicit covariance (NPY or single-array NPZ) and validates rank-one leakage in the original checkpoint dtype. It remains a stub for causal claims until certification tests pass. No general LEACE or semantic erasure claim is made.

## 6. Security Model & Runtime Guardrails

WSP-Lang enforces a multi-layer defense-in-depth model to prevent accidental model corruption:

### 6.1. The Protected Tensor Policy (`FORBIDDEN_TENSORS`)
Under no circumstances may a WSP-Lang program write to or mutate critical grounding tensors:
* `token_embd.weight`: Token embedding matrix.
* `output.weight`: Unembedding / language model head.
* `output_norm.weight`: Final layer normalization.
* `*.norm`, `*.attn_norm.weight`, `*.ffn_norm.weight`: Internal RMSNorm scales.

Any write (a WSP-Lang write op, or `wsp.py poke` / `edit`) on a protected tensor is blocked with:
```json
{
  "status": "security_rejection",
  "error": "SECURITY POLICY REJECTION: Tensor 'output.weight' is protected from direct edit."
}
```

### 6.1b. Write guard (enforced since 2026-11-05)
Every op that writes weights goes through one check, `guard_model_write` in `engine/wsplang.py`
(tests: `tests/test_wsp_write_guard.py`). Before this, only `project_out`, `project_in` and `delta_unlearn`
applied budgets and protected tensors, and several ops overwrote the model they read
(`lab/audit/README.md`).
* An edit never overwrites the model it read. It writes to `out=<file>` or `--out-dir`; without either it
  is an error. A file written earlier by the same run may be edited in place.
* The edited file keeps the source's tensor names, shapes and dtypes. Added, dropped or reshaped tensors
  are refused.
* Changed elements must be `<= max_modified_weights`, and max |delta| must be `<= max_delta_norm`. Both are
  measured on the file about to be written. Simulations (`allow_write = false`) run the same check.
* Protected tensors (the list above, plus names containing `embed`, `lm_head`, `wte`, `ln_f`, `norm.weight`)
  change only when the program lists them: `unlock_tensors = lm_head.weight, output.weight` under `budgets:`.
* `export_control_vector` writes only with `allow_write = true`.
* A failed assertion gets status `assertion_failed`. The CLI exits with code 1 when any step has status
  `error`, `security_rejection` or `assertion_failed`.

### 6.2. Strict Validation Gate
The runtime validates all AST nodes before passing the plan to the execution engine. If `errors > 0`:
1. Disk access is blocked.
2. Model file locks are released.
3. Execution halts with status `aborted`.

---

## 7. Model Context Protocol (FastMCP) Integration

WSP-Lang is natively integrated with FastMCP (`engine/wsp_mcp_server.py`), allowing autonomous agents to execute code without writing to disk:

```mermaid
sequenceDiagram
    participant Agent as Autonomous AI Agent
    participant MCP as WSP FastMCP Server
    participant Gate as Strict Validation Gate
    participant Engine as WSP-Lang Runtime
    participant Model as GGUF Storage (NVMe)

    Agent->>MCP: execute_wsplang_code(code)
    MCP->>Gate: validate(prog)
    alt Validation Errors Detected
        Gate-->>MCP: errors: [...]
        MCP-->>Agent: {"valid": false, "status": "aborted"}
    else Validation Passed
        Gate->>Engine: execute_plan(plan)
        Engine->>Model: read/write blocks
        Model-->>Engine: weights & centroids
        Engine-->>MCP: execution_results
        MCP-->>Agent: {"valid": true, "status": "ok", "results": {...}}
    end
```

### Exposed MCP Tools
1. `execute_wsplang_code(code, gguf)`: In-memory declarative execution.
2. `run_wsplang(program_path, execute, gguf)`: File-based declarative execution.
3. `peek_weight(gguf, tensor, row, col, top)`: Read-only anatomical inspection.
4. `poke_weight(gguf, tensor, row, col, value, out)`: Guarded single-weight modification.
5. `get_semantic_areas(gguf, tensor, row, col)`: Centroid map and projection.
6. `run_agent_loop_probe(endpoint, alias, n, ...)`: Agentic tool-call loop benchmarking.
7. `run_logic_probe(endpoint, n, seed)`: Deterministic synthetic Horn-logic verification.

---

## 8. Canonical Program Examples

### Example 1: Multi-Layer Semantic Inspection ([`example_semantic_areas.wspl`](../programs/example_semantic_areas.wspl))
```wspl
experiment inspect_semantic_areas:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: map_semantic_geometry

    semantic_areas:
        mode = comprehensive
        target = math_logic

    steps:
        # 1. Map top semantic areas
        step semantic_map top=3

        # 2. Inspect Attention Layer 19 (Row 100, Col 200)
        step peek blk.19.attn_output.weight row=100 col=200 top=3

        # 3. Inspect FFN Layer 19 (Row 100, Col 500)
        step peek blk.19.ffn_down.weight row=100 col=500 top=3

        # 4. Inspect Recurrent SSM Layer 0
        step peek blk.0.ssm_out.weight row=100 col=200 top=3
```

---

### Example 2: Move a fine-tune's skill onto the base ([`transplant_jackrong_math.wspl`](../programs/transplant_jackrong_math.wspl))
The README's main result: the change of a reasoning distill against Qwen3.5-9B as a rank-16 LoRA adapter (exact SVD),
with `energy.json` (share of the change kept per rank) and a `verdict` before any eval. `model=hf:<owner>/<repo>[@rev]`
reads the fine-tune from the Hugging Face hub by byte ranges.

```wspl
experiment transplant_jackrong_math:
steps:
    step delta_to_lora base=hf_cache/qwen35-9b-hf model=hf_cache/dp_keep_dsv4_distill rank=16 out=hf_cache/close_lora/demo_jack_r16 gguf=hf_cache/close_lora/demo_jack_r16.gguf
```

### Example 3: Read-only inspection of layer-14 neurons ([`dreaming_covariance_steering.wspl`](../programs/dreaming_covariance_steering.wspl))
The program only reads (`semantic_map`, `peek`). An earlier version of this file showed scaling lines here; scaling is
not a WSP-Lang step (§5.3-5.5), and the column-scaling edit was never evaluated (README, "Evil Qwen").

```wspl
experiment dreaming_covariance_steering:
    steps:
        step semantic_map top=4
        step peek blk.14.ffn_down.weight row=100 col=4258 top=3
        step peek blk.19.attn_output.weight row=100 col=200 top=3
```

The earlier Example 2 (`blog_exp5_write_mlp.wspl`, MIT Exp 5 identity MLP by single-weight writes) does not pass the
validator (its `poke` steps are rejected) and is no longer shown.

---

## 9. Manifest & Fingerprint Schema

Execution of any WSP-Lang program automatically generates a cryptographically fingerprinted JSON manifest stored in `challenges/wsplang/<experiment_id>.manifest.json`:

```json
{
  "programa": "inspect_semantic_areas",
  "archivo": "programs/example_semantic_areas.wspl",
  "steps_count": 4,
  "steps": [
    {"sec": "steps", "cmd": "step", "op": "semantic_map", "args": "top=3"},
    {"sec": "steps", "cmd": "step", "op": "peek", "args": "blk.19.attn_output.weight row=100 col=200 top=3"}
  ],
  "plan": {
    "secciones": {"semantic_areas": 2, "steps": 4},
    "ops": {"mode": "comprehensive", "target": "math_logic"}
  },
  "errors": [],
  "validated": true,
  "ts": "2026-11-03T21:30:50",
  "ejecucion": [
    {
      "op": "peek",
      "args": "blk.19.attn_output.weight row=100 col=200 top=3",
      "status": "ok",
      "result": {
        "tensor": "blk.19.attn_output.weight",
        "semantic_areas": {
          "dominant": {"area": "math_logic", "label": "Math & Logic", "score": 0.0275},
          "top_areas": [
            {"area": "math_logic", "label": "Math & Logic", "score": 0.0275},
            {"area": "time_sequence", "label": "Time & Sequence", "score": 0.0149},
            {"area": "social_power", "label": "Social & Power", "score": 0.0098}
          ]
        }
      }
    }
  ]
}
```

---

## 10. Summary & Roadmap

Current priority (roadmap §4.0sexies, 2026-11-09): the transplant op (`delta_to_lora`: verdict from energy, adapter, automatic validation), tested on more donors. Standing rules:
1. **Freeze Core Syntax:** No syntax expansion until the first verified empirical `keep` is established.
2. **Deterministic Evaluation First:** All weight modifications must be gated by `agent_loop_probe` or sealed holdout benchmarks.
3. **Pure BF16 Pipeline:** Transition high-precision experiments to uncompressed BF16 tensors, utilizing canonical `llama-quantize` for final GGUF compilation.
