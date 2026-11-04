---
license: mit
base_model: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
tags:
- weight-space-programming
- mechanistic-interpretability
- negative-control
- reflection
- self-critique
- control-vector
- wsp-lang
model_name: MiMo-V2.6-Distill-Qwen-9B-WSP-Critic
pipeline_tag: text-generation
---

# 🧠 MiMo-V2.6-Distill-Qwen-9B-WSP-Critic (Research Artifact & Negative Control Study)

An experimental checkpoint based on [XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) investigating the extraction and multi-layer projection of a lexical reflection direction using **Weight-Space Programming (WSP)**.

* **Repository & Codebase:** [FiveTechSoft/weights](https://github.com/FiveTechSoft/weights)
* **Status:** **Experimental Research Artifact (Negative Causal Result / Caution)**

---

## ⚠️ Transparent Scientific Disclosure & Limitations

In the interest of open science, reproducibility, and rigorous mechanistic interpretability, we explicitly document the empirical limitations and negative controls of this artifact:

1. **Extraction Source (Lexical Embeddings, Not Residual Activations):**
   - The direction vector `u_critic` was extracted by averaging token embeddings directly from `embed_tokens.weight` for contrastive texts ("Wait, let me verify..." vs uncritical answers).
   - **No forward pass was executed** during extraction. Therefore, this direction reflects lexical token differences rather than dynamic intermediate layer representations.
2. **Causal Contract Evaluation (Null Effect):**
   - In our 4-way paired causal verification contract (`rig/exp_mimo_self_critique_v1.py`), single-layer intervention resulted in:
     - Baseline: **50.0%**
     - Candidate ($u_{\text{critic}}$ edit): **50.0%**
     - Placebo Control: **50.0%**
     - Random Orthogonal Control: **50.0%**
   - The causal engine declared `is_statistically_superior: False` and executed an **automatic rollback**.
3. **Prompt Confounding in Preliminary Benchmarks:**
   - Preliminary comparisons showing 40% vs 100% were **confounded by prompt framing** on the same server (comparing a 60-token concise zero-shot prompt against a 250-token step-by-step chain-of-thought prompt), rather than an isolated weight-only effect under identical prompts.
   - When evaluating with identical prompts, the isolated vector does not demonstrate a statistically significant causal improvement over baseline.
4. **Train / Test Overlap:**
   - Several evaluated puzzles (watermelon, strawberry, bat-and-ball) were part of the 30 contrastive pairs used in the lexical vocabulary difference calculation.

---

## 🔬 What This Model Represents

This checkpoint and its accompanying GGUF control vector (`u_critic_control_vector.gguf`) are released as **open scientific artifacts**:
* To study how lexical embedding differences propagate through intermediate layers when structured as a Gaussian envelope (Layers 10..22).
* As a documented **negative control** for researchers studying representation engineering (RepE) vs prompt effects in distilled 9B reasoning models.
* To demonstrate automated causal contract engines that catch and roll back unverified weight modifications.

---

## 🛠️ WSPLang Program Specification

```wspl
experiment mimo_reflection_envelope:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: synthesize_multilayer_self_critique_envelope

    budgets:
        max_modified_weights = 654311424
        target_parameter_fraction = 0.072
        max_delta_norm = 0.05
        allow_write = true

    steps:
        # Multi-layer Gaussian envelope over layers 10..22 (peak=16, bandwidth=2.5)
        step layer_envelope 10..22 peak=16 bandwidth=2.5 envelope=gaussian vector=data/u_critic_L16.npz out_gguf=data/u_critic_control_vector.gguf n_layers=32
```

---

## 💻 Technical Usage

### Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "fivetech/MiMo-V2.6-Distill-Qwen-9B-WSP-Critic"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### llama.cpp Control Vector
```bash
llama-server.exe \
  -m mimo-v2.6-distill-qwen-9b-Q6_K.gguf \
  --control-vector u_critic_control_vector.gguf \
  -ngl 99 -c 32768 -np 2 --port 8081
```

---

## 📚 Citation

```bibtex
@software{fivetechsoft_mimo_wsp_critic_2026,
  author = {FiveTechSoft},
  title = {MiMo-V2.6-Distill-Qwen-9B-WSP-Critic: A Mechanistic Study and Negative Control on Weight-Space Deliberation},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/fivetech/MiMo-V2.6-Distill-Qwen-9B-WSP-Critic}
}
```
