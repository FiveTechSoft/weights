# FiveTechSoft / weights: Weight-Space Programming (WSP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Website: GitHub Pages](https://img.shields.io/badge/Website-Research_Portal-cyan.svg)](https://fivetechsoft.github.io/weights/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: MiMo-V2.6-Distill-Qwen-9B](https://img.shields.io/badge/Model-MiMo--V2.6--Distill--Qwen--9B-blue.svg)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)

A toolkit and a research log for **Weight-Space Programming (WSP)**: reading, comparing and editing the weights of
LLMs directly, with no gradient step.
- It inspects and edits quantized GGUF files (Q6_K) and safetensors checkpoints.
- It diffs a fine-tuned model against its base and reports what changed, including outside the weights.
- It runs declarative experiments written in WSP-Lang (`.wspl`).

Reading and editing weights needs no forward pass. **Behavioural claims do**: every behavioural result here was
measured by running the model (llama.cpp / llama-server, usually on an RTX 3060).

Starting point: MIT CSAIL's blog post [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
(Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, 2 Nov 2026; a blog post, not peer-reviewed).

## Main result (2026-11-09)

**A fine-tune's skill can be moved onto the unchanged base model as a small adapter, with no training.**
WSP-Lang's `delta_to_lora` op reads the difference between a fine-tune and its base, matrix by matrix, keeps the
strongest directions (exact SVD) and writes a normal LoRA adapter. On Qwen3.5-9B with Jackrong's reasoning distill as
the donor, rank 16, 87 MB, a few minutes:

| Qwen3.5-9B, greedy, 2,048-token budget | base | base + adapter (×0.5) |
|---|---|---|
| MATH-500, 36 held-out items (16 the base cannot finish + 20 it solves) | 6/36 | **30/36** |
| GSM8K, 48 items (another benchmark; never used to pick anything) | 9/48 | **37/48** |
| GSM8K problems the base already solved, kept | 6/6 | **6/6** |
| **Sealed confirmation** (2026-11-10): GSM8K 100 + MATH-500 50 never used by any decision | 43/100, 4/50 | **91/100, 35/50** |

Paired against the base (exact McNemar), both p < 0.0001. The sealed run (plan written before it, results §43.15) confirms with two graders per benchmark (gained/lost 48/0 and 31/0) while a same-norm random placebo stays at the base (46/100, 5/50); much of the gain is that the adapter reasons 2-3× shorter and finishes within the budget. Before any benchmark the op's `energy.json` says whether a
change will move: rank 16 keeps 58.6 % of Jackrong's change (it moves) and 3.8 % of MiMo's (it does not, at any rank
up to 512). Demo and commands: [`demos/math_transplant/`](demos/math_transplant/). It **needs a donor**: every attempt
to create the skill without one (synthesis, learned jumps, merges, self-distillation) failed; see the table below.
**Does the verdict generalize?** Four donors so far (results §43.11-43.12), verdict given before any benchmark:

| donor | change kept at rank 16 | verdict | what the adapter did |
|---|---|---|---|
| Jackrong reasoning distill (SFT) | 58.6 % | likely | MATH 6 → 30/36, GSM8K 9 → 37/48 |
| iAmBoosted OSS-Distilled (merged LoRA r16), read from the hub | 98.3 % | likely | moved its *behaviour* onto another model (MiMo): ≈ 3× shorter reasoning, MATH answers cut at 8,192 tokens 13 → 3, a same-norm placebo does not; but GSM8K 92 → 83/100 (p = 0.035), so it fails its pre-registered criterion |
| LeoZotos math GRPO (RL) | 7.5 % | unlikely | not distinguishable from the base (GSM8K 15/48 vs 9, p = 0.21) although the full donor improves it (21/48, p = 0.012) |
| MiMo 2.6 (full fine-tune) | 3.8 % | unlikely | does not move at any rank up to 512 |

The verdict was right each time; moving a change does not make the change good (OSS). The main result is confirmed
on a sealed set (`data/sealed_confirm.json`, sha256-checked, disjoint from everything used so far; results §43.15) and next transplants will be too.

**Second use: steering without touching weights** (results §43.13). The `contrast_vector` op turns two sets of answers
(same prompts, cheerful vs neutral) into a llama.cpp control vector. On 40 unseen prompts, Qwen3.5-9B with the vector at
scale 0.25: judged tone 2.83 → 3.90 of 5 (sign test p = 3·10⁻⁸), positive-affect words 0.27 → 2.63 per 100 (p = 7·10⁻¹²),
37/40 coherent; a same-norm random vector changes nothing (+0.05, +0.02). Passed all five pre-registered criteria
after two fixes to the judge, both made before the test set was used (documented). Scales ≥ 0.5 break coherence.

🌐 Research portal: [fivetechsoft.github.io/weights](https://fivetechsoft.github.io/weights/) ·
📒 Full log: [`results.md`](results.md) · 🗺️ Plan: [`roadmap.md`](roadmap.md) ·
🧪 Weight-diff findings: [`lab/probes/delta_patterns/findings.md`](lab/probes/delta_patterns/findings.md)

---

## Status of the results (November 2026, audited)

Every claim below was re-checked against the files in this repo (2026-11-06; rows on reusing a fine-tune's change
added 2026-11-09). "n" is the number of evaluated
items. Results that did not survive the audit are listed as such instead of being removed.

| Result | Evidence | Status |
| :--- | :--- | :--- |
| **Fine-tune → adapter (`delta_to_lora`)**, Jackrong's distill on Qwen3.5-9B (results §43.4-43.6) | See *Main result*. Rank 4 fails (11/36), rank 8 27/36, rank 16 30-32/36. HumanEval first 20: 20/20 base, 19/20 with the lab adapter. | ✅ Measured, paired tests; **confirmed on the sealed set** (91/100, 35/50 vs base 43, 4; placebo 46, 5; results §43.15). |
| **Transfer verdict on 4 donors** (results §43.11-43.12) | See *Main result*: `likely` for Jackrong (58.6 %) and OSS-Distilled (98.3 %), both transferred; `unlikely` for LeoZotos (7.5 %) and MiMo (3.8 %), neither did. | ✅ Measured, small sets; LeoZotos's test is weak (adapter between base and donor, neither difference significant). |
| **Shorter reasoning on MiMo with OSS-Distilled's change** (results §43.12) | GSM8K 100 + MATH-500 50, 8,192 tokens: MATH cut 13 → 3 (×1) / 5 (×0.5), tokens 3,077 → 965; placebo 17 cut. GSM8K 92 → 83 at both doses (p = 0.035 / 0.012, two graders agree). | ❌ Fails its pre-registered criterion: the behaviour moves, accuracy drops. |
| **Happiness control vector (`contrast_vector`)** (results §43.13) | See *Main result*. Judge (MiMo) and lexicon agree; placebo null; GSM8K fast set 7/48 vs 9/48 (within the margin of 2). | ✅ Passes C1-C5 after two documented judge fixes. One model, one style, 40 test prompts. |
| **MiMo 2.6 vs DeepSeek-V4-Pro-Qwen3.5-9B-heretic** (results §43.14) | Same protocol: GSM8K 92 vs 89 (p = 0.55), MATH-500 41 vs 33 (p = 0.10; second grader 36 vs 33, p = 0.61); heretic thinks ≈ 2.5× less (1,223 vs 3,077 tokens on MATH, 3 vs 13 cut). | ✅ Statistical tie on math. HumanEval not used: heretic's code comes back with 1-space indentation in 13/20 (cause not verified). Refusals not measured. |
| **MiMo's change does not fit an adapter** (results §43.4-43.5) | Full fine-tune, diffuse change: rank 64-512 adapters reach ≤ 4/16 on the stuck MATH items vs MiMo's own 12/16. | ✅ Negative result, predicted by `energy.json`. |
| **Copying MiMo's behaviour instead of its weights** (results §43.7-43.10) | Rank-16 adapter trained on MiMo's own answers: MATH 6 → 16 / 24 / 27 of 36 with 93 / 171 / 256 answers, GSM8K 9 → 34 / 41 / 41 of 48. Closed-form, gradient-free fit of `down_proj` to MiMo's hidden states (the WSP/MEMIT write used for a behaviour): 17/36, 25/48; a same-norm random placebo gives 10/36, 14/48 (not significant vs base, p 0.13 / 0.23), and the fit beats it (p 0.039 / 0.003). Writing the attention output too does not help (15/36, 26/48). | ✅ Measured. Distillation flattens after 171 answers; the gradient-free route is closed. |
| **Creating a skill without a donor** (results §40-§42, §43.8) | Qwen3.5-0.8B, 3×3 multiplication: a training's change lives in a narrow subspace shared across seeds (overlap 0.72-0.80, chance 0.016); rank 20 recovers 91 % of it, another seed's subspace 92 %. But extrapolating early steps, one gradient, coefficients in the subspace, a learned jump (0/7 tasks) and summing changes all fail; donor-free self-distillation on 9B = base. | ❌ Not achieved. Moving an existing change works; making one does not. |
| **MIT Exp 8 replication** (Qwen3.5-0.8B loop edit) | Edited tensors are bit-identical to Astra's (SHA-256 match with the manifest in `exp8_control/p50c_edit/`). On a MATH-500 holdout, Q6_K, n=20, temp 0.3, 2048-token cap: base 1/20, edit 14/20, random-direction placebo 0/20 (Fisher p ≈ 3.9e-5 edit vs base). Files: `rig_runs/holdout-08b-{base,edit,placebo}-20.json`. | ✅ Replicated, small n. A GSM8K smoke test (n=5, 512-token cap) gave 0/5 vs 1/5, with **all 10 answers cut at the cap**, so it shows no loop break. |
| **Exp 8 idea on a trained nanoGPT** (results §36) | Loops 111 → 26 of 256 test prompts. One of 3 random edits of the same norm also gave 26. | ❌ Fails the pre-registered criterion 4. |
| **Loop fix on MiMo-9B** | Blind holdout, n=40: loops 17/40 with the edit vs 20/40 with the placebo, p = 0.654 (results.md, `DISCARD_CANDIDATE_V1`). GSM8K n=50 accuracy was kept: 45 → 46. | ❌ Discarded as a fix: not distinguishable from the placebo. |
| **What post-training changes** (Qwen3.5-9B vs MiMo, NeoHorse, tmax, Fara, official post-training, WeMM, Nyx-RP, Angry-Claudius, DeepSeek-V4 distill/Heretic) | `lab/probes/delta_patterns/findings.md` F1–F26, with two-method checks. | ✅ Measured. Full fine-tunes share *where* they change the weights whatever the goal (F22). The layer-specific *directions* are shared by the improvements and not by a full fine-tune with another goal (F23). A LoRA control cannot separate a degradation from a neutral LoRA (F24). A storage-cast bug (F32→BF16 counted as change) was found and corrected (F26). |
| **`model_report`**: what a model changed vs its base, in about a minute over HTTP | Validated on 6 models whose change is known (`lab/runs/model_report/validation.json`), plus config/tokenizer/template diff (F25: MiMo ships an older pre-tokenizer regex). | ✅ Validated on Qwen3.5-9B derivatives. Other architectures: tensor names are mapped (roadmap J1) but not yet calibrated. |
| **TwIL-8B vs MiMo-9B benchmark** (results §37) | Same protocol: GSM8K 100 + MATH-500 50, greedy, 8192 tokens. TwIL 96/100 and 44/50. MiMo 93/100 and 34/50, with 19 answers cut at the cap. | ✅ Measured. Base Qwen3.5-9B under the same protocol: GSM8K 60/100, MATH-500 24/50, with 36 + 25 answers cut at the cap; MiMo's advantage is mostly finishing within the budget (results §39). |
| **Atari policy reading** (MIT Exp 1) | W3 statistics: vertical/horizontal 1.54× for Pong, 1.53× for Breakout. One MiMo-9B query, given hand-computed statistics (not raw weights), answered "Pong" with self-reported 90 % confidence. | ⚠️ Anecdote (one query). |
| **LEACE-style closed-form edit** | Rank-1 projection zeroes ‖uᵀW‖ for one direction in layer 14 `down_proj` in 47 ms (`rig_runs/leace_verification_report.json`). | ⚠️ True by algebra. No data, probe or behaviour tested. |
| **NanoWSP-2** (599,168 params) and **NanoWSP-3** (3,280,128 params) | They recall all imprinted pairs: 11/11 and 23/23. | ⚠️ Memorization testbeds. Recall measured on the stored items only; no generalization tested. Not language models. |
| **MiMo → TinyLlama fact transfer** | Real run: target facts 5/8 → 7/8, holdout preserved 5/8 (`rig_runs/mimo_to_tinyllama_full_migration_report.json`). | ⚠️ n too small. The earlier "0 → 100 %, 18.4 ms, 0 % collateral" report was written from hardcoded values (`scripts/save_tinyllama_benchmark_report.py`) and is **withdrawn**. |
| **MiMo "self-critique" causal contract** (50 % in all 4 arms) | `rig/exp_mimo_self_critique_v1.py` scores a synthetic projection of W on a prompt hash. It never runs the model. | ❌ **Withdrawn**: not a behavioural test. |
| **"Evil" neurons in MiMo layer 14** | `edits/evil_mimo.json` scales 4 `ffn_down` columns (4258 and 7714 ×2.5; 10739 and 993 ×0.25), 16,384 weights. | ⚠️ No behavioural evaluation of hostility was run. The GGUF is not in the repo. |

---

## ⚡ Quick start

```bash
git clone https://github.com/FiveTechSoft/weights.git
cd weights
pip install gguf numpy matplotlib pillow   # GGUF inspection and editing (wsp.py)
pip install torch safetensors              # NanoWSP, nanoGPT, weight diff, model_report
```

### Move a fine-tune's skill onto the base (`delta_to_lora`)
```wspl
experiment transplant_jackrong_math:
steps:
    step delta_to_lora base=hf_cache/qwen35-9b-hf model=hf_cache/dp_keep_dsv4_distill rank=16 out=hf_cache/close_lora/demo_jack_r16 gguf=hf_cache/close_lora/demo_jack_r16.gguf
```
```bash
python -m engine.wsplang programs/transplant_jackrong_math.wspl --run
llama-server -m <Qwen3.5-9B base>.gguf --lora hf_cache/close_lora/demo_jack_r16.gguf --lora-init-without-apply --port 8095
python demos/math_transplant/demo.py 8095        # same server: adapter off vs on
```
`model=hf:<owner>/<repo>[@revision]` reads the fine-tune from the Hugging Face hub by byte ranges instead of a local
copy. The step returns `verdict` (`likely` / `uncertain` / `unlikely`) from the weighted energy kept at rank 16, before
any eval. `gguf=` needs `WSP_LORA_CONVERTER` (llama.cpp's `convert_lora_to_gguf.py`). One score for an adapter:
`lab/probes/close_probe/reward.py` (IWIBench reward = gain on stuck items × retention of solved ones; a no-op scores 0).

### Steer a mood with a control vector (`contrast_vector`)
```wspl
experiment happy_vector:
steps:
    step contrast_vector model=hf_cache/qwen35-9b-hf pos=lab/runs/happy/pairs/pos.jsonl neg=lab/runs/happy/pairs/neg.jsonl out=hf_cache/cvec/happy.gguf placebo=2026
```
```bash
python -m engine.wsplang programs/happy_vector.wspl --run      # pos/neg: jsonl rows {prompt, response}
llama-server -m <Qwen3.5-9B base>.gguf --control-vector-scaled hf_cache/cvec/happy.gguf:0.25 --port 8095
```
The full measured study (pairs, judge, lexicon, placebo, retention) is `lab/probes/happy/` with its plan in `PLAN.md`.

### Inspect one weight of a GGUF
```bash
python wsp.py peek <path-to-gguf> blk.19.attn_output.weight -r 100 -c 200
```
The tool dequantizes the Q6_K block on the fly. It prints the weight's role (layer, head, residual dimension) and
its projection on 10 centred semantic centroids built from token embeddings. Those labels are a lexical heuristic,
not a measured function of the weight.

### What did a fine-tune change? (`model_report`)
```wspl
experiment report_mimo:
objective: how MiMo differs from Qwen3.5-9B
steps:
    step model_report model=XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B base=Qwen/Qwen3.5-9B out=lab/runs/model_report/mimo.md
```
```bash
python engine/wsplang.py <program>.wspl --run
```
It reads one block of rows per tensor by HTTP range requests, with no full download, and classifies the change:
`identical`, `directional_ablation`, `lora_like` or `full_fine_tune`. It also diffs config, tokenizer and chat
template, and runs an encode test over 14 scripts. Example output: [`lab/runs/model_report/`](lab/runs/model_report/).

### NanoWSP: a micro-Transformer whose associations are written in weight space
```bash
python scripts/demo_nanowsp_capabilities.py
```
2 layers, d=64, 80,704 parameters. Associations are stored by closed-form projections. It is a lookup testbed,
not a language model.

### nanoGPT W0 declared as data

Instantiate an untrained `nanoGPT` (10.77M parameters, 6 layers, d=384, 6 heads, vocab=65) and check it against
upstream `karpathy/nanoGPT@3adf61e`.
- The architecture is declared in `programs/arch/nanogpt.wsparch` and compiled by `engine/wsp_arch.py`, with no
  model class behind it ([`docs/wsparch.md`](docs/wsparch.md)).
- Training is a separate gradient-descent step (`programs/train_nanogpt_shakespeare_char.wspl`).
- `programs/synthesize_nanogpt_ngram.wspl` compiles an n-gram table into the same graph without gradients. That is
  a lookup table, not a synthesized language model.

```bash
python engine/wsplang.py programs/build_karpathy_nanogpt.wspl --run
python lab/verify_official_karpathy_parity.py
```

| Parity test | Observed | Verdict |
| :--- | :--- | :--- |
| Logits, forward pass | Δ = 0.0 (fp32) | exact match |
| Weight tying (`wte` ↔ `lm_head`) | ‖Δ‖_F = 0.0 | identical |
| Causal mask | past leakage 0.0 | strict |
| Greedy generation | 0 discrepancies / 75 tokens | bit-exact |
| Loss and gradients | Δ = 0.0 | exact match |

This shows the runtime matches upstream at W0 (same seed, same weights). The checkpoint is random, with validation
NLL near ln 65. For language-model behaviour use `lab/verify_llmness.py`, calibrated by `lab/calibrate_llmness.py`.

👉 Full walkthrough: [Getting Started Guide](docs/getting_started.md).

---

## 🧠 WSP-Lang

A declarative language for weight-space experiments (runtime version 0.7.0). Steps cover:
- inspection: `peek`, `semantic_map`;
- edits under a guard: `project_out`, `scale-cols`, closed-form imprints;
- weight diffs: `delta_stats`, `delta_signature`, `delta_compare`, `model_report`, `delta_to_lora` (a fine-tune's change as a reusable LoRA adapter, local or from the hub, with a transfer verdict);
- steering: `contrast_vector` (a llama.cpp control vector from two sets of answers, with a same-norm placebo);
- behavioural evaluation: `evaluate_benchmark` against a running server.

Spec: [`docs/wsplang_specification.md`](docs/wsplang_specification.md).

```wspl
experiment compare_angry:
objective: does a deliberate degradation share the improvements' layer-specific directions?
steps:
    step delta_compare a=lab/runs/delta_patterns/sig_angry_claudius.pt b=lab/runs/delta_patterns/sig_mimo.pt out=lab/runs/delta_patterns/pairs_angry_claudius_mimo.json
```

**Semantic centroids.** Raw token embeddings sit in a narrow cone (baseline cosine ~0.7). WSP-Lang therefore
centres its 10 semantic centroids on their mean: $\mathbf{C}'_a = \mathbf{C}_a - \frac{1}{A}\sum_k \mathbf{C}_k$.
The contrast gain this gives has not been measured.

---

## 🔬 Core components

| Component | Description |
|---|---|
| [`wsp.py`](wsp.py) | GGUF Q6_K inspector and in-place editor; NumPy port of `quantize_row_q6_K_ref` (87 % byte-identical on unedited blocks, error ~6e-6). |
| [`engine/wsplang.py`](engine/wsplang.py) | WSP-Lang parser, validator and runtime (`--run`). |
| [`engine/weight_diff.py`](engine/weight_diff.py) | Tensor-by-tensor diff of two checkpoints. It reads safetensors locally or by HTTP ranges and maps tensor names across architectures. Per tensor it gives relative change, spectrum and subspace signatures; storage casts are not counted as change. |
| [`engine/contrast_vector.py`](engine/contrast_vector.py) | `contrast_vector`: per-layer mean difference of two sets of answers, written as llama.cpp per-layer increments (GGUF) + placebo. |
| [`engine/sealed_set.py`](engine/sealed_set.py) | Sealed confirmation set (GSM8K 100 + MATH-500 50) with a sha256 check; rule in [`lab/probes/sealed/`](lab/probes/sealed/README.md). |
| [`engine/lora_export.py`](engine/lora_export.py) | `delta_to_lora`: exact SVD of each matrix's change, PEFT adapter + `energy.json` (share of the change kept at each rank). |
| [`demos/math_transplant/`](demos/math_transplant/) | The main result as a runnable demo (adapter off vs on, same server). |
| [`lab/probes/close_probe/`](lab/probes/close_probe/) | Base vs improved Qwen3.5-9B: fast eval (`fast_eval.py`), reward (`reward.py`), behaviour distillation (`distill_gen.py`, `distill_train.py`), closed-form fit (`cf_distill.py`), placebo adapters (`placebo_lora.py`). |
| [`engine/model_report.py`](engine/model_report.py) | "What did this model change vs its base": change class, layers, modules, rank, plus config/tokenizer/template diff. |
| [`lab/probes/delta_patterns/`](lab/probes/delta_patterns/) | Study of what post-training changes in Qwen3.5-9B derivatives: pre-registered plan, findings F1–F26. |
| [`lab/probes/model_compare/`](lab/probes/model_compare/) | Benchmark protocol (GSM8K + MATH-500, two independent graders). |
| [`engine/nanowsp_runtime.py`](engine/nanowsp_runtime.py) | NanoWSP micro-Transformer (d=64, 80,704 params). |
| [`engine/wsp_mcp_server.py`](engine/wsp_mcp_server.py) | MCP server exposing `peek`, `poke` and WSP-Lang to AI agents. |
| [`challenges/`](challenges/) | WSP challenge suite: 8 levels, 43 challenges, mechanical judges. |
| [`edits/`](edits/) | JSON edit manifests (e.g. `evil_mimo.json`). |
| [`FOUNDATIONS.md`](FOUNDATIONS.md) | The scientific basis, by claim: what we hold, its status, our evidence (results §), outside evidence, what would refute it; refuted ideas not to retry. |
| [`PROTOCOL.md`](PROTOCOL.md) | Protocol: sealed holdouts, placebo controls, non-degradation criteria. |
| [`results.md`](results.md) · [`roadmap.md`](roadmap.md) · [`CHANGELOG.md`](CHANGELOG.md) | Results log, plan, patch history (P0–P12). |
| [`tests/`](tests/) | 50 test files (`python -m unittest discover -s tests`). |

---

## 🛡️ Rules we hold ourselves to

1. **Failures stay documented.** E1–E5, the nanoGPT loop fix, the MiMo loop fix, and the withdrawn results in the
   table above are all kept.
2. **Causal claims need a placebo.** A random-direction edit of the same norm is required by `PROTOCOL.md`. It was
   used in Exp 8 and in the loop fixes. The older experiments without one are marked ⚠️ above.
3. **Holdouts are sealed** before an edit is designed. Evaluation uses GSM8K and MATH-500 subsets; since 2026-11-09 a
   hashed confirmation set, disjoint from every item used to pick doses or donors, is run once on anything that passes.
4. **Edits are fenced.** `FORBIDDEN_TENSORS` (in `wsp.py` and `engine/wsplang.py`) blocks embeddings, the output head
   and norms unless explicitly allowed.
5. **Numbers need two methods.** A figure is reported only if two independent computations agree. Example: the
   weight diffs are checked against a second, block-sampled method.

---

## 🎭 Behavioural steering ("Evil Qwen")

Viral posts showed that steering one or two neurons can make a model act like a villain. We tried the same idea
without forward passes:
- **The search:** layer 14 `ffn_down` columns of MiMo-9B were ranked by their projection on lexical centroids (*anger*,
  politeness).
- **The edit:** 4 columns were scaled (`edits/evil_mimo.json`).
- **Why it is not expected to work:** the "Evil Qwen" effect *adds* a fixed amount to a neuron's activation at every
  token, far beyond its natural range (dahara1's analysis on Qwen3.5-4B: about 20× the dataset maximum; scaling the
  same neurons ×20 or ×100 leaves the replies practically unchanged, because they barely fire). Qwen3.5's `down_proj`
  has no bias, so that addition cannot be written into the weights; it needs a control vector
  ([dahara1/EvilQwen-control-vectors](https://huggingface.co/dahara1/EvilQwen-control-vectors), llama.cpp). Scaling
  columns, as `evil_mimo.json` does, is the multiplicative kind. We have not tested either on MiMo. The additive
  kind does work here for a mood: the happiness control vector (`contrast_vector`, results §43.13) passes its
  pre-registered test with a placebo.
- **Not done:** the behavioural effect was **never evaluated**, so this is an untested edit, not a result. Notes:
  [`docs/behavioral_steering_and_dreaming.md`](docs/behavioral_steering_and_dreaming.md).

In the related project [`FiveTechSoft/dreaming`](https://github.com/FiveTechSoft/dreaming), steering through
`token_embd` alone did not change behaviour. That evidence lives in that repo, not here.

---

## 🌐 Community reports and what we checked

| Source | Claim | What this repo checked |
| :--- | :--- | :--- |
| [Simon Willison](https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/) | Qwen 27B identifies Atari games from raw MLP weights. | Prompt rebuilt ([`data/exp1_pong_prompt.txt`](data/exp1_pong_prompt.txt), 17,253 parameters). Our own check is one MiMo-9B query on summary statistics (see the table above). |
| [Jarrod Watts](https://x.com/jarrodwatts/status/2117808859691811216) | Astra's 0.016 % edit (≈131k weights in two layers) fixes Qwen3.5-0.8B loops, MATH-500 7.9 % → 48.1 %. | Bit-exact replication of the edited tensors (130,188 entries actually differ). MATH-500 holdout n=20: 1/20 → 14/20, placebo 0/20. |
| [Jacob Lauritzen](https://x.com/jacsebl/status/2117606206764175543) | The edit removes an obstacle (loops) and adds no knowledge. | Consistent with the Exp 8 holdout: correct answers appear once generation terminates. Not tested separately. |
| [areu01or00/qwen0.8B-wsp](https://github.com/areu01or00/qwen0.8B-wsp) | A fact written into Qwen3.5-0.8B's `down_proj` in closed form (MEMIT-style), 2/5 new phrasings. | Its own history reports 0/5 once the weights are saved in bf16. We used the same closed-form write for a behaviour on 9B (row *Copying MiMo's behaviour*): it transfers some, below gradient distillation. |
| [khordoo/dusty-wsp](https://github.com/khordoo/dusty-wsp) | An 8M model renamed by a rank-2 edit; a fear→curiosity edit fails (0/48). | Read, not run. Consistent with ours: a fact has a target, a disposition does not. We took its placebo control for our adapters. |
| [Ayush (@stochi0)](https://x.com/stochi0/status/2117681387688071185), [Paweł Huryn](https://x.com/PawelHuryn/status/2117639929463197885) | Low-rank RL policies; weight edits as a debugging primitive. | Discussed, **not verified** here. |

We also audit viral WSP claims (results that rest on compiled rules, narrowed tasks or unverifiable artefacts). The
recurring failure patterns are summarized in `lab/probes/delta_patterns/findings.md` (F24–F26) and `results.md`.

---

## 📚 References

How these sources bear on our claims: [`FOUNDATIONS.md`](FOUNDATIONS.md).


- Takehi, Teoh, Zhang, Khattab (MIT CSAIL), [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), blog post, 2 Nov 2026.
- Experiment logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). Edited Exp 8 model: [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
- Demo: [akhaliq/weight-space-programming-demo](https://huggingface.co/spaces/akhaliq/weight-space-programming-demo).
- Main target model: [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B), a Qwen3.5-9B derivative (32 layers: 8 full-attention, 24 Gated DeltaNet).
- Bennington, Kaelar, Lee, Edwards, Glick, Danks (Concept Country), [*Gradient-Free Weight Editing Reduces Overthinking in Qwen3.8-27B*](https://concept.country/research/weight-space-programming), working paper v1, 8 Nov 2026 ([post](https://x.com/varien/status/2119880941711827025), [code](https://github.com/conceptcountry/weight-space-programming), [weights](https://huggingface.co/ConceptCountry/qwen-concept-27b)). A direction read from the model's own traces (last tokens before `</think>` minus the rest), written as rank-1 terms into 9 `down_proj` matrices: GPQA Diamond +4.3, LiveCodeBench v6 +13.0, 27-54 % fewer thinking tokens, a norm-matched random edit does nothing. We are replicating it donor-free on Qwen3.5-9B ([`lab/probes/cc/PLAN.md`](lab/probes/cc/PLAN.md)).
- Hu, Tang, Li, Zhu, Feng, Zhang, Chen, Hu, [*Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations*](https://arxiv.org/abs/2611.09382), arXiv 2611.09382, 6 Nov 2026 ([code](https://github.com/WSP-Scope/WSP-Scope-Project)). 16 agent editing searches on Qwen3.5-0.8B and 9B: one `down_proj` column lifts 0.8B MATH-500 8.4 → 37.8 (a loop fix; random direction of equal norm 10/48 vs edit 20/48 on dev); on 9B gains are small (IFEval +1.9, SciBench +3.0) and a MATH dev gain (+29.7) vanished held-out (base already 97.7 at 32k tokens).
- Nathan Barry, [*Can a Linear Program Do Weight-Space Programming?*](https://nathanrs.rs/posts/weight-space-lp) ([code](https://github.com/nathanrs/weight-space-lp)), Nov 2026. Finds the MIT Exp 8 loop-breaking edit with a sparse linear program in minutes instead of a search; a constant steering push gives almost the same result.
- Background: Irie & Schmidhuber, fast weight programmers (ICML 2021). Belrose et al., LEACE (NeurIPS 2023).

## 📄 License

MIT. See [LICENSE](LICENSE).
