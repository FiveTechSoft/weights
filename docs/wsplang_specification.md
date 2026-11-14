# WSP-Lang: Declarative Language Specification for Weight-Space Programming

**Version:** 0.7.0 (= `engine/wsplang.py` `__version__`; this file is the base text, the per-version additions are in `docs/wsplang_0.2_spec.md` ... `docs/wsplang_0.7_spec.md`, and §5.0 lists every op the runtime accepts)  
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
| `architectural_drift` | read / report | not documented yet (see `engine/wsplang.py`) |
| `assert_behavioral_drift` | read / report | `wsplang_0.7_spec.md` |
| `assert_davis_kahan` | writes weights | `wsplang_0.4_spec.md` |
| `assert_orthogonal_preservation` | writes weights | `wsplang_0.3_spec.md` |
| `assert_weyl` | writes weights | `wsplang_0.4_spec.md` |
| `audit` | writes weights | `wsplang_0.2_spec.md` |
| `compare` | read / report | `wsplang_0.2_spec.md` |
| `compile_control_vector` | read / report | this file |
| `compile_synthetic` | writes weights | `wsplang_0.7_spec.md` |
| `contrast_vector` | read / report | this file |
| `cov_imprint` | writes weights | `wsplang_0.5_spec.md` |
| `cross_project` | writes weights | `wsplang_0.6_spec.md` |
| `decompile` | writes weights | not documented yet (see `engine/wsplang.py`) |
| `decoupled_delta` | writes weights | `wsplang_0.5_spec.md` |
| `delta_compare` | read / report | this file |
| `delta_signature` | read / report | this file |
| `delta_stats` | read / report | this file |
| `delta_to_lora` | read / report | this file |
| `delta_unlearn` | writes weights | this file, `wsplang_0.3_spec.md`, `wsplang_0.4_spec.md` |
| `diff` | read / report | `wsplang_0.2_spec.md` |
| `drift_profile` | read / report | not documented yet (see `engine/wsplang.py`) |
| `evaluate` | read / report | this file |
| `evaluate_benchmark` | read / report | this file |
| `export_control_vector` | writes weights | this file, `wsplang_0.2_spec.md` |
| `fwp_overlay` | writes weights | `wsplang_0.3_spec.md` |
| `imprint_association` | writes weights | not documented yet (see `engine/wsplang.py`) |
| `imprint_batch` | writes weights | not documented yet (see `engine/wsplang.py`) |
| `imprint_corpus` | writes weights | `wsplang_0.7_spec.md` |
| `init_architecture` | writes weights | `wsplang_0.7_spec.md` |
| `jepa_steer` | writes weights | `wsplang_0.7_spec.md` |
| `layer_envelope` | writes weights | `wsplang_multilayer_envelope_spec.md` |
| `leace_erase` | stub | `wsplang_0.5_spec.md` |
| `memit_imprint` | writes weights | `wsplang_0.5_spec.md` |
| `model_report` | read / report | this file |
| `multi_layer_envelope` | writes weights | not documented yet (see `engine/wsplang.py`) |
| `peek` | read / report | this file, `wsplang_0.2_spec.md` |
| `permute_align` | writes weights | `wsplang_0.7_spec.md` |
| `project_in` | writes weights | this file, `wsplang_0.2_spec.md`, `wsplang_0.3_spec.md` |
| `project_out` | writes weights | this file, `wsplang_0.2_spec.md`, `wsplang_0.4_spec.md` |
| `rosetta_align` | writes weights | `wsplang_0.6_spec.md` |
| `semantic-areas` | read / report | not documented yet (see `engine/wsplang.py`) |
| `semantic-map` | read / report | not documented yet (see `engine/wsplang.py`) |
| `semantic_areas` | read / report | `wsplang_0.2_spec.md` |
| `semantic_delta` | read / report | not documented yet (see `engine/wsplang.py`) |
| `semantic_map` | read / report | this file, `wsplang_0.2_spec.md` |
| `spectral_audit` | writes weights | `wsplang_0.4_spec.md` |
| `spectral_decay` | writes weights | `wsplang_0.3_spec.md` |
| `spectral_delta` | read / report | not documented yet (see `engine/wsplang.py`) |
| `ssm_dynamics_diff` | read / report | not documented yet (see `engine/wsplang.py`) |
| `subspace_transfer` | writes weights | this file |
| `surprise_fwp` | stub | `wsplang_0.5_spec.md` |
| `synthesize` | writes weights | `wsplang_0.7_spec.md` |
| `tie_weights` | writes weights | `wsplang_0.7_spec.md` |
| `train` | writes weights | not documented yet (see `engine/wsplang.py`) |
| `vicreg_preserve` | read / report | `wsplang_0.7_spec.md` |
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
