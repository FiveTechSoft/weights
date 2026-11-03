# Getting Started with FiveTechSoft / weights

Welcome to **weights**, an open-source toolkit for **Weight-Space Programming (WSP)**. This guide will walk you through setup, running your first weight introspection in under 60 seconds, and executing declarative WSP-Lang programs.

---

## 1. Prerequisites & Environment Setup

This repository is designed to run with **zero GPU overhead** and **zero heavyweight frameworks** (no PyTorch required for core inspection/editing). All quantization operations and projections run directly on quantized GGUF weights via NumPy.

### Minimal Installation

```bash
# Clone the repository
git clone https://github.com/FiveTechSoft/weights.git
cd weights

# Create and activate a clean virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install minimal dependencies
pip install gguf numpy matplotlib pillow
```

*(Optional: For MCP server capabilities, install FastMCP)*
```bash
pip install mcp
```

---

## 2. Setting Up a GGUF Model

WSP works with any standard GGUF file. Our primary research benchmark is **MiMo-V2.6-Distill-Qwen-9B** quantized in `Q6_K`.

You can use any GGUF model on your drive:
- Default path used by WSP-Lang examples: `C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf`
- Or pass your custom path using the `--gguf` flag:
  ```bash
  python engine/wsplang.py programs/example_semantic_areas.wspl --gguf /path/to/your/model.gguf --run
  ```

---

## 3. Quick Tour: 3 Things You Can Do in 5 Minutes

### A. Peek into a Single Weight & Decompile its Semantics (60 seconds)

You can read any single weight block directly from NVMe disk, dequantize it on the fly, and project its subspace onto differential semantic centroids:

```bash
# Peek into Attention Layer 19 (Row 100, Col 200 -> Head 0)
python wsp.py peek <path-to-gguf> blk.19.attn_output.weight -r 100 -c 200
```

**What you will see:**
```text
=== ANATOMIA DEL TENSOR ===
Tensor: blk.19.attn_output.weight
Forma: [4096, 4096], Tipo: Q6_K
Dimension residual: 4096, Cabezas de atencion: 16 (dim_head=256)
Elemento consultado [100, 200]:
  Valor des-cuantizado: +0.021484
  Rol arquitectonico: Proyeccion residual (fila 100) desde cabeza de atencion 0 (dim interna 200)

AREA SEMANTICA DOMINANTE: Matematicas y logica (math_logic) [score=+0.0275]
Top 3 areas mas importantes:
  * Matematicas y logica         (math_logic     ) score=+0.0275
  * Tiempo y secuencia           (time_sequence  ) score=+0.0149
  * Social y autoridad           (social_power   ) score=+0.0098
Semillas afines: learn:+0.034, shame:+0.031, theorem:+0.029
```

---

### B. Run Declarative WSP-Lang Programs (2 minutes)

Rather than writing low-level scripts, WSP-Lang lets you express experimental hypotheses declaratively in `.wspl` files:

```bash
python engine/wsplang.py programs/example_semantic_areas.wspl --run
```

**How a WSP-Lang program looks:**
```wspl
experiment inspect_semantic_areas:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: map_semantic_geometry

    semantic_areas:
        mode = comprehensive
        target = math_logic

    steps:
        # 1. Map all 10 semantic centroids in residual space
        step semantic_map top=3

        # 2. Inspect Attention Layer 19 (Logic Reasoning Head)
        step peek blk.19.attn_output.weight row=100 col=200 top=3

        # 3. Inspect FFN Layer 19 (Intermediate Concept Storage)
        step peek blk.19.ffn_down.weight row=100 col=500 top=3

        # 4. Inspect GatedDeltaNet SSM Recurrent Layer 0
        step peek blk.0.ssm_out.weight row=100 col=200 top=3
```

Execution produces structured artifacts and manifests in `challenges/wsplang/*.manifest.json`.

---

### C. Connect via MCP (Model Context Protocol) to AI Agents

You can expose the full WSP toolkit to Claude Desktop, Cursor, or any agentic system:

```bash
python engine/wsp_mcp_server.py
```

The MCP server exposes:
- `get_semantic_areas()`: Returns the 10 differential centroids and lexical clusters.
- `peek_weight()`: Inspects tensor anatomy, architectural role, and semantic projection.
- `poke_weight()`: Performs safe in-place surgical edits with canonical Q6_K requantization.
- `run_wsplang()`: Parses, validates, and runs declarative `.wspl` programs.

---

## 4. The Mathematical Foundation: Differential Centroids

LLM vocabulary embeddings typically cluster inside a narrow high-dimensional cone (baseline cosine similarity ~0.65 to 0.80), masking subtle functional directions.

WSP solves this by centering all 10 semantic centroids against the vocabulary mean:

$$\mathbf{C}'_a = \mathbf{C}_a - \frac{1}{A} \sum_{k=1}^A \mathbf{C}_k$$

When a column vector $\mathbf{w} \in \mathbb{R}^{4096}$ is projected onto $\mathbf{C}'_a$:

$$\text{score}(\mathbf{w}, a) = \frac{\mathbf{w} \cdot \mathbf{C}'_a}{\|\mathbf{w}\| \|\mathbf{C}'_a\|} \in [-1.0, +1.0]$$

This expands cosine dynamic contrast by **+2.4x**, centering background noise at 0.00 and isolating clear semantic roles.

---

## 5. Experimental Protocol & Safety

- **Zero-Loss Integrity**: For edits, WSP utilizes a canonical C-level port of `quantize_row_q6_K_ref`, achieving **87% byte-identity** and reconstruction error $\approx 6 \times 10^{-6}$.
- **Forbidden Tensors**: Protected critical components (token embeddings, final LM normalization, unembedding matrix) cannot be mutated.
- **Strict Falsifiability**: All hypothesis-driven edits are tested against **Placebo edits** and **sealed holdout benchmarks** (GSM8K, MATH-500). Discarded experiments are openly documented.

---

## 6. Next Steps

- Explore the **[Roadmap](file:///c:/weights/roadmap.md)** for our current progress and experiment logs.
- Review **[PROTOCOL.md](file:///c:/weights/PROTOCOL.md)** for experimental standards and holdout isolation.
- Read **[CHANGELOG.md](file:///c:/weights/CHANGELOG.md)** for complete version and patch history.
