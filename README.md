# FiveTechSoft / weights: Weight-Space Programming (WSP)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: MiMo-2.6-Qwen-9B](https://img.shields.io/badge/Model-MiMo--V2.6--Distill--Qwen--9B-cyan.svg)](https://huggingface.co/)
[![Quant: Q6_K GGUF](https://img.shields.io/badge/Quant-Q6__K_GGUF-purple.svg)](https://github.com/ggerganov/llama.cpp)
[![Compute: Zero GPU Overhead](https://img.shields.io/badge/Compute-0_MB_RAM_/_0_FLOPs-emerald.svg)](https://github.com/FiveTechSoft/weights)

An open-source toolkit for **Weight-Space Programming (WSP)**. Directly inspect, decompile, and surgically edit individual weights in 9B production quantized LLMs without forward passes, training gradients, or GPU overhead.

Based on the research methodology of MIT CSAIL's *Sparks of Weight-Space Programming* ([paper & blog](https://rikiyat.github.io/blog/weight-space-programming/)) and extended to production quantized GGUF architectures.

---

![WSP-Lang Semantic Architecture](https://raw.githubusercontent.com/FiveTechSoft/weights/main/docs/assets/wsplang_semantic_areas.png)

---

## ⚡ Quick Start (5 Minutes)

You don't need GPUs or PyTorch to decompile model weights. Core inspection runs directly on quantized GGUF blocks using minimal NumPy.

### 1. Install Minimal Requirements
```bash
git clone https://github.com/FiveTechSoft/weights.git
cd weights
pip install gguf numpy matplotlib pillow
```

### 2. Peek into Any Weight & Decompile its Semantic Role (60 Seconds)
Inspect any weight block directly on disk. WSP dequantizes the 32-byte scale block on the fly and projects it against 10 differential semantic centroids:

```bash
# Decompile Attention Layer 19 (Row 100, Col 200 -> Head 0)
python wsp.py peek <path-to-gguf> blk.19.attn_output.weight -r 100 -c 200
```
Output:
```text
=== ANATOMIA DEL TENSOR ===
Tensor: blk.19.attn_output.weight [4096, 4096] (Q6_K)
Elemento [100, 200]: Head 0 projection into residual dim 100

AREA SEMANTICA DOMINANTE: Matematicas y logica (math_logic) [score=+0.0275]
Top 3 areas:
  * Matematicas y logica         (math_logic     ) score=+0.0275
  * Tiempo y secuencia           (time_sequence  ) score=+0.0149
  * Social y autoridad           (social_power   ) score=+0.0098
Semillas afines: learn:+0.034, theorem:+0.029, integer:+0.025
```

### 3. Run Declarative WSP-Lang Programs
Express experimental hypotheses declaratively in `.wspl` files and execute them with the built-in compiler:

```bash
python engine/wsplang.py programs/example_semantic_areas.wspl --run
```

👉 **Need a full walkthrough?** Read our step-by-step **[Getting Started Guide](docs/getting_started.md)**.

---

## 🧠 What is WSP-Lang?

WSP-Lang is a domain-specific declarative language for sub-weight experimentation:

```wspl
experiment inspeccion_areas_semanticas:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: map_semantic_geometry

    semantic_areas:
        mode = comprehensive
        target = math_logic

    steps:
        # 1. Map all 10 semantic centroids in residual space
        step semantic_map top=3

        # 2. Attention Layer 19 (Logic Reasoning Head)
        step peek blk.19.attn_output.weight row=100 col=200 top=3

        # 3. FFN Intermediate Layer 19 (Concept Storage)
        step peek blk.19.ffn_down.weight row=100 col=500 top=3

        # 4. GatedDeltaNet SSM Recurrent Layer 0
        step peek blk.0.ssm_out.weight row=100 col=200 top=3
```

### The Differential Centroid Solution
LLM embeddings suffer from **isotropic cone collapse** (raw token vectors cluster inside a tight cone with baseline cosine ~0.70). WSP-Lang centers all 10 semantic centroids against the vocabulary baseline:

$$\mathbf{C}'_a = \mathbf{C}_a - \frac{1}{A} \sum_{k=1}^A \mathbf{C}_k$$

This expands cosine dynamic contrast by **+2.4x**, drops background noise to 0.00, and enables instant semantic attribution without forward passes.

---

## 🔬 Core Components

| Component | Description |
|---|---|
| [`wsp.py`](wsp.py) | **WSP Toolkit**: In-place GGUF Q6_K surgical editor + canonical C-level requantizer (`quantize_row_q6_K_ref`, 87% byte-identity, error ~6e-6) + differential semantic centroid calculator. |
| [`engine/wsplang.py`](engine/wsplang.py) | **WSP-Lang Compiler & Runtime**: Parser, AST validator, interactive execution engine (`--run`), and manifest generator. |
| [`engine/wsp_mcp_server.py`](engine/wsp_mcp_server.py) | **FastMCP Server**: Exposes weight inspection (`peek`), surgical editing (`poke`), and WSP-Lang execution directly to AI agents (Claude Desktop, Cursor, Antigravity). |
| [`challenges/`](challenges/) | **WSP-40 Challenge Suite**: 8 difficulty levels and 40 challenges evaluated with strictly mechanical judges (no self-grading). |
| [`programs/`](programs/) | **WSP-Lang Programs**: Declarative `.wspl` files for semantic mapping, inspection, and weight synthesis. |
| [`edits/`](edits/) | **Weight Edit Manifests**: Structured JSON edit programs and operator configurations. |
| [`PROTOCOL.md`](PROTOCOL.md) | **Scientific Protocol**: Double-blind sealed holdouts, placebo controls, and non-degradation criteria. |
| [`roadmap.md`](roadmap.md) | **Live Research Journal**: Active hypotheses, empirical findings, and next milestones. |
| [`CHANGELOG.md`](CHANGELOG.md) | **Strict Technical Ledger**: Full history of patches (P0–P7) with "What / Why / Measured Result". |

---

## 🛡️ Scientific Discipline & Falsifiability

We adhere to strict falsifiability standards:
1. **No Cherry-Picking**: Failed hypotheses (E1–E5) are fully documented with empirical evidence rather than discarded quietly.
2. **Placebo Controls (E4)**: Every candidate weight edit is evaluated alongside a random-direction placebo edit to verify that improvements are causal.
3. **Sealed Holdouts**: Evaluated on GSM8K and MATH-500 holdout sets that never leak into edit design.
4. **Safety Boundaries**: Critical components (`token_embd`, `output.weight`, layer norms) are protected by hard safety fences (`FORBIDDEN_TENSORS`).

---

## 📚 References & Prior Art

- MIT CSAIL Blog: [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) (Rikiya Takahashi)
- Experiment Logs & Astra Code: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)
- Hugging Face Demo: [akhaliq/weight-space-programming-demo](https://huggingface.co/spaces/akhaliq/weight-space-programming-demo)
- Target Model: **MiMo-V2.6-Distill-Qwen-9B** (Hybrid 8 Attention layers + 24 GatedDeltaNet SSM layers)

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
