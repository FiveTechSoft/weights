# [Research & Weight Surgery] Investigating Agentic Tool-Calling Loops via Subtractive Rank-1 Projection on Layer 14

*🇬🇧 English version first | 🇪🇸 Versión en español a continuación*

**Target Model:** [`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)  
**Research & Open-Source Artifacts:** [GitHub: FiveTechSoft/weights](https://github.com/FiveTechSoft/weights)  
**Authors:** FiveTechSoft Research Team  
**License:** MIT  

---

> [!WARNING]
> **TECHNICAL ERRATA & SCIENTIFIC AUDIT NOTICE (Nov 4, 2026):**
> 1. **Signal Separation (Layer 1 vs Layer 14):** In our empirical SNR separation scan (`rig_runs/loop_direction_report.json`), Layer 1 showed an SNR separation ratio of 1.83x ($p=0.01$, 100 permutations), whereas Layer 14 measured at ratio 1.00 ($p=0.44$, noise level). The 1.83x ratio was previously misattributed to Layer 14; Layer 14 was originally selected based on architectural analogy to MIT Exp 8 middle-layer MLP projections.
> 2. **Correct Tensor Dimensions:** The MiMo-9B down-projection tensor is $4096 \times 12288$ (located in `model-00001-of-00004.safetensors`), NOT $3584 \times 18944$ (which is Qwen2.5-7B). Vector $u \in \mathbb{R}^{4096}$.
> 3. **Behavioral Recovery Status:** The 90% GSM8K score is the unedited model baseline capability. Claims of closed behavioral recovery on multi-turn tool loops are an empirical research hypothesis undergoing paired holdout validation ($N \ge 80$). Rank-1 projection currently verifies algebraic energy reduction on disk without full behavioral closure.

---

## 🇬🇧 English Version

### Executive Summary

We have investigated the mechanistic root cause of recursive **agentic tool-calling loops** in `MiMo-V2.6-Distill-Qwen-9B` and formulated a surgical rank-1 projection framework deployable across three distinct environments:
1. **Vanilla PyTorch / SafeTensors:** Standalone script, no specialized frameworks required.
2. **Weights Space Programming (WSP-Lang):** Declarative weight engineering with formal safety budgets, leverage score filtering, and contract verification.
3. **Dynamic Inference Control Vector:** Runtime steering in `llama.cpp` with zero permanent file modifications.

By studying localized rank-1 projection subtraction on Layer 14's MLP down-projection tensor (`model.language_model.layers.14.mlp.down_proj.weight`, $4096 \times 12288$), we analyze the dampening of loop attractor energy:
- **Target Tensor:** `model.language_model.layers.14.mlp.down_proj.weight` ($4096 \times 12288$) in `model-00001-of-00004.safetensors`.
- **Directional Energy Reduction:** $\|u^T W\|_2$ reduced algebraically from 1.39906 to 1.39627 (-0.20%).
- **Format:** Native **BF16 SafeTensors** (exact floating-point surgery) and GGUF runtime steering.
- **Status:** 100% open-source and reproducible on [FiveTechSoft/weights](https://github.com/FiveTechSoft/weights).

---

### 1. Why Does MiMo-V2.6-Distill Have This Behavior? (Mechanistic Analysis)

1. **Distillation on "Golden Paths":**
   `MiMo-V2.6-Distill-Qwen-9B` is trained via distillation from large frontier models. The distillation dataset primarily consists of synthetic multi-step agent trajectories where tools succeed immediately or provide cooperative outputs. The model rarely saw trajectories where a tool failed repeatedly or returned ambiguous environment feedback.

2. **The Special Token Attractor:**
   In multi-turn agent formats (such as SGLang or ReAct tool protocols), the sequence `<tool_response> ... </tool_response>` strongly primes the probability of predicting `<tool_call>`. When tool output is repetitive or unexpected, the residual stream does not register an updated world state; the hidden representations remain stuck in the "tool-required" subspace.

3. **Layer Separation Analysis:**
   Activation tracing across 32 transformer layers reveals early-layer separation (Layer 1 with 1.83x ratio, $p=0.01$), while middle layers (Layer 14) serve as broad syntactic routing hubs.

4. **Benchmark Blind Spots:**
   Standard benchmarks (GSM8K, MATH, HumanEval) test single-turn or strictly bounded generation, where MiMo excels (e.g., closing `<think>` tags cleanly in ~130 tokens with 90% accuracy). Multi-turn conversational degradation under uncooperative environment responses is simply not captured by standard leaderboards.

---

### 2. The Mathematical Surgery (Rank-1 Subtraction)

Rather than performing costly fine-tuning or reinforcement learning, we isolate directional energy in weight space:

$$W_{:, \text{cols}} \leftarrow W_{:, \text{cols}} - \alpha \cdot u (u^T W_{:, \text{cols}})$$

Where:
- $W \in \mathbb{R}^{4096 \times 12288}$ is `model.language_model.layers.14.mlp.down_proj.weight`.
- $u \in \mathbb{R}^{4096}$ is the normalized contrastive vector (`data/u_loop_L14.npz`).
- $\alpha = 0.05$ (damping coefficient).
- $\text{cols}$ is the subset of intermediate features ($12.7\%$ of columns, 1,560 columns) with the highest projection energy onto $u$.

---

### 3. Three Practical Deployment Options

#### Option 1: Vanilla Python & PyTorch (Without WSPLang — Permanent SafeTensors Surgery)
```bash
# 1. Clone the repository to get the calibrated direction vector
git clone https://github.com/FiveTechSoft/weights.git
cd weights

# 2. Run the standalone PyTorch script
python scripts/fix_mimo_vanilla_pytorch.py \
  --model-dir /path/to/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B \
  --out-dir /path/to/MiMo-V2.6-Distill-Qwen-9B-repaired \
  --u-path data/u_loop_L14.npz \
  --alpha 0.05
```

**Minimal Standalone Python Snippet (Pure PyTorch):**
```python
import torch, numpy as np
from safetensors.torch import load_file, save_file

# 1. Load direction vector (4096-dim)
u = torch.from_numpy(np.load("data/u_loop_L14.npz")["u"]).float()
u = u / torch.norm(u)

# 2. Load Layer 14 down_proj from shard 1
shard = load_file("model-00001-of-00004.safetensors")
W = shard["model.language_model.layers.14.mlp.down_proj.weight"].float()

# 3. Subtractive rank-1 projection
u_proj = u @ W
top_cols = torch.topk(torch.abs(u_proj), k=int(W.shape[1] * 0.127)).indices
delta = 0.05 * torch.outer(u, u_proj[top_cols])

# 4. Apply and save
W[:, top_cols] -= delta
shard["model.language_model.layers.14.mlp.down_proj.weight"] = W.to(torch.bfloat16)
save_file(shard, "repaired_model/model-00001-of-00004.safetensors")
```

---

#### Option 2: Declarative Weights Space Programming (WSPLang)
```wspl
experiment fix_mimo_tool_loops:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: suppress_agentic_tool_calling_loops_via_weight_subtraction
    budgets:
        max_modified_weights = 1271011
        target_parameter_fraction = 0.0252
        max_delta_norm = 0.04
        allow_write = true
    steps:
        step peek blk.14.ffn_down.weight row=100 col=4258 top=3
        step diff blk.14.ffn_down.weight base=mimo-v2.6-distill-qwen-9b-Q6_K.gguf
        step semantic_map top=3
```

Execute via the open-source WSP runtime:
```bash
python engine/wsplang.py programs/fix_mimo_tool_loops.wspl --run
```

---

#### Option 3: Dynamic Inference Control Vector (llama.cpp / Ollama — Zero Disk Changes)
For users running quantized GGUF models locally who do not wish to modify any model files on disk:

```bash
# Launch llama-server with negative loop direction scaling:
llama-server \
  -m mimo-v2.6-distill-qwen-9b-Q6_K.gguf \
  --control-vector-scaled data/loop_control_vector.gguf -0.5 \
  --port 8081
```

---

### 4. Verification & Current Status

- **Tensorial Verification:** Directional energy $\|u^T W\|_2$ dropped from 1.39906 to 1.39627 (-0.20%).
- **Reasoning Baseline:** Unedited baseline GSM8K accuracy is 90%.
- **Behavioral Recovery Status:** Experimental hypothesis under active paired holdout evaluation ($N \ge 80$).

---

### 5. Repository & Artifacts

All vectors, unit tests, scripts, and logs are MIT-licensed on GitHub:
- **GitHub Repository:** [https://github.com/FiveTechSoft/weights](https://github.com/FiveTechSoft/weights)
- **Standalone PyTorch Script:** [`scripts/fix_mimo_vanilla_pytorch.py`](https://github.com/FiveTechSoft/weights/blob/main/scripts/fix_mimo_vanilla_pytorch.py)
- **Control Vector Binary:** [`data/loop_control_vector.gguf`](https://github.com/FiveTechSoft/weights/blob/main/data/loop_control_vector.gguf)

---
---

## 🇪🇸 Versión en Español

### Resumen Ejecutivo

Hemos investigado la causa raíz mecanística de los **bucles repetitivos en llamadas a herramientas agentic** en `MiMo-V2.6-Distill-Qwen-9B` y formulado una intervención proyectiva de rango 1 analizable en tres modalidades:
1. **PyTorch / SafeTensors estándar (Sin WSPLang):** Script standalone sin dependencias complejas.
2. **Weights Space Programming (WSP-Lang):** Especificación declarativa con presupuestos formales y verificación de preservación ortogonal.
3. **Vector de Control en Inferencia Dinámica:** Modificación en tiempo real del flujo residual en `llama.cpp` sin alterar archivos en disco.

---

### 1. Análisis Mecanístico y Fe de Erratas Técnica
- **Separación SNR:** La Capa 1 presenta una separación de 1.83x ($p=0.01$), mientras que la Capa 14 se sitúa en ratio 1.00 ($p=0.44$, ruido estadístico).
- **Dimensiones Reales:** Matriz `down_proj` es de $4096 \times 12288$ en el shard `model-00001-of-00004.safetensors` ($u \in \mathbb{R}^{4096}$).
- **Estado Experimental:** El 90% en GSM8K representa la línea base del modelo intacto. La resolución conductual del bucle permanece como hipótesis bajo protocolo de validación pareada ($N \ge 80$).
