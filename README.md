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

🌐 Research portal: [fivetechsoft.github.io/weights](https://fivetechsoft.github.io/weights/) ·
📒 Full log: [`results.md`](results.md) · 🗺️ Plan: [`roadmap.md`](roadmap.md) ·
🧪 Weight-diff findings: [`lab/probes/delta_patterns/findings.md`](lab/probes/delta_patterns/findings.md)

---

## Status of the results (November 2026, audited)

Every claim below was re-checked against the files in this repo on 2026-11-06. "n" is the number of evaluated
items. Results that did not survive the audit are listed as such instead of being removed.

| Result | Evidence | Status |
| :--- | :--- | :--- |
| **MIT Exp 8 replication** (Qwen3.5-0.8B loop edit) | Edited tensors are bit-identical to Astra's (SHA-256 match with the manifest in `exp8_control/p50c_edit/`). On a MATH-500 holdout, Q6_K, n=20, temp 0.3, 2048-token cap: base 1/20, edit 14/20, random-direction placebo 0/20 (Fisher p ≈ 3.9e-5 edit vs base). Files: `rig_runs/holdout-08b-{base,edit,placebo}-20.json`. | ✅ Replicated, small n. A GSM8K smoke test (n=5, 512-token cap) gave 0/5 vs 1/5, with **all 10 answers cut at the cap**, so it shows no loop break. |
| **Exp 8 idea on a trained nanoGPT** (results §36) | Loops 111 → 26 of 256 test prompts. One of 3 random edits of the same norm also gave 26. | ❌ Fails the pre-registered criterion 4. |
| **Loop fix on MiMo-9B** | Blind holdout, n=40: loops 17/40 with the edit vs 20/40 with the placebo, p = 0.654 (results.md, `DISCARD_CANDIDATE_V1`). GSM8K n=50 accuracy was kept: 45 → 46. | ❌ Discarded as a fix: not distinguishable from the placebo. |
| **What post-training changes** (Qwen3.5-9B vs MiMo, NeoHorse, tmax, Fara, official post-training, WeMM, Nyx-RP, Angry-Claudius, DeepSeek-V4 distill/Heretic) | `lab/probes/delta_patterns/findings.md` F1–F26, with two-method checks. | ✅ Measured. Full fine-tunes share *where* they change the weights whatever the goal (F22). The layer-specific *directions* are shared by the improvements and not by a full fine-tune with another goal (F23). A LoRA control cannot separate a degradation from a neutral LoRA (F24). A storage-cast bug (F32→BF16 counted as change) was found and corrected (F26). |
| **`model_report`**: what a model changed vs its base, in about a minute over HTTP | Validated on 6 models whose change is known (`lab/runs/model_report/validation.json`), plus config/tokenizer/template diff (F25: MiMo ships an older pre-tokenizer regex). | ✅ Validated on Qwen3.5-9B derivatives. Other architectures: tensor names are mapped (roadmap J1) but not yet calibrated. |
| **TwIL-8B vs MiMo-9B benchmark** (results §37) | Same protocol: GSM8K 100 + MATH-500 50, greedy, 8192 tokens. TwIL 96/100 and 44/50. MiMo 93/100 and 34/50, with 19 answers cut at the cap. | ✅ Measured. Base Qwen3.5-9B under the same protocol is running. |
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

A declarative language for weight-space experiments. Steps cover:
- inspection: `peek`, `semantic_map`;
- edits under a guard: `project_out`, `scale-cols`, closed-form imprints;
- weight diffs: `delta_stats`, `delta_signature`, `delta_compare`, `model_report`, `delta_to_lora` (a fine-tune's change as a reusable LoRA adapter);
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
| [`engine/model_report.py`](engine/model_report.py) | "What did this model change vs its base": change class, layers, modules, rank, plus config/tokenizer/template diff. |
| [`lab/probes/delta_patterns/`](lab/probes/delta_patterns/) | Study of what post-training changes in Qwen3.5-9B derivatives: pre-registered plan, findings F1–F26. |
| [`lab/probes/model_compare/`](lab/probes/model_compare/) | Benchmark protocol (GSM8K + MATH-500, two independent graders). |
| [`engine/nanowsp_runtime.py`](engine/nanowsp_runtime.py) | NanoWSP micro-Transformer (d=64, 80,704 params). |
| [`engine/wsp_mcp_server.py`](engine/wsp_mcp_server.py) | MCP server exposing `peek`, `poke` and WSP-Lang to AI agents. |
| [`challenges/`](challenges/) | WSP challenge suite: 8 levels, 43 challenges, mechanical judges. |
| [`edits/`](edits/) | JSON edit manifests (e.g. `evil_mimo.json`). |
| [`PROTOCOL.md`](PROTOCOL.md) | Protocol: sealed holdouts, placebo controls, non-degradation criteria. |
| [`results.md`](results.md) · [`roadmap.md`](roadmap.md) · [`CHANGELOG.md`](CHANGELOG.md) | Results log, plan, patch history (P0–P12). |
| [`tests/`](tests/) | 34 test files (`python -m unittest discover -s tests`). |

---

## 🛡️ Rules we hold ourselves to

1. **Failures stay documented.** E1–E5, the nanoGPT loop fix, the MiMo loop fix, and the withdrawn results in the
   table above are all kept.
2. **Causal claims need a placebo.** A random-direction edit of the same norm is required by `PROTOCOL.md`. It was
   used in Exp 8 and in the loop fixes. The older experiments without one are marked ⚠️ above.
3. **Holdouts are sealed** before an edit is designed. Evaluation uses GSM8K and MATH-500 subsets.
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
| [Ayush (@stochi0)](https://x.com/stochi0/status/2117681387688071185), [Paweł Huryn](https://x.com/PawelHuryn/status/2117639929463197885) | Low-rank RL policies; weight edits as a debugging primitive. | Discussed, **not verified** here. |

We also audit viral WSP claims (results that rest on compiled rules, narrowed tasks or unverifiable artefacts). The
recurring failure patterns are summarized in `lab/probes/delta_patterns/findings.md` (F24–F26) and `results.md`.

---

## 📚 References

- Takehi, Teoh, Zhang, Khattab (MIT CSAIL), [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), blog post, 2 Nov 2026.
- Experiment logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs). Edited Exp 8 model: [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
- Demo: [akhaliq/weight-space-programming-demo](https://huggingface.co/spaces/akhaliq/weight-space-programming-demo).
- Main target model: [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B), a Qwen3.5-9B derivative (32 layers: 8 full-attention, 24 Gated DeltaNet).
- Background: Irie & Schmidhuber, fast weight programmers (ICML 2021). Belrose et al., LEACE (NeurIPS 2023).

## 📄 License

MIT. See [LICENSE](LICENSE).
