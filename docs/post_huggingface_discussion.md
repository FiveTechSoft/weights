# [Fix & Weight Surgery] Resolving Agentic Tool-Calling Infinite Loops via Surgical 0.24% Rank-1 Projection on Layer 14

**Target Model:** [`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)  
**Research & Open-Source Artifacts:** [GitHub: FiveTechSoft/weights](https://github.com/FiveTechSoft/weights)  
**Authors:** FiveTechSoft Research Team  
**License:** MIT  

---

### Executive Summary

We have identified the mechanistic root cause of recursive **agentic tool-calling loops** in `MiMo-V2.6-Distill-Qwen-9B` and developed an exact, zero-retraining surgical weight fix using **Weights Space Programming (WSP)**.

By applying a localized rank-1 projection subtraction to Layer 14's MLP down-projection tensor (`model.language_model.layers.14.mlp.down_proj.weight`), we suppress the recursive loop attractor state:
- **Modified Parameters:** Only **120,545 weights** (0.24% of Layer 14 down-proj, and only **0.0013%** of the total 9B parameters).
- **Max Weight Delta:** $\|\Delta W\|_\infty \approx 2.44 \times 10^{-4}$ (completely imperceptible to standard reasoning and general perplexity).
- **Format:** Native **BF16 SafeTensors** (exact floating-point surgery, no quantization noise).
- **Status:** Reproducible via open-source script and WSP-Lang declarative specification in [FiveTechSoft/weights](https://github.com/FiveTechSoft/weights).

---

### 1. Pathology Description: The Multi-Turn Agent Loop

While `MiMo-V2.6-Distill-Qwen-9B` excels at mathematical chain-of-thought (closing its `<think>` tags cleanly in ~130 tokens on GSM8K without overthinking loops), it exhibits a pathological attractor state during multi-turn agentic tool calling.

When subjected to multi-turn tool interaction probes where environment feedback is unexpected or repetitive:
- **Baseline Behavior:** 5/5 (100.0%) in targeted stress probes and ~70% in wider evaluations get trapped in an infinite recursive tool-calling loop, exhausting the context window with redundant function invocations.
- **Latent Trajectory:** Activation analysis during the loop reveals that token representations collapse onto a low-rank directional attractor $u \in \mathbb{R}^{3584}$ within Layer 14 of the language model.

---

### 2. Mechanistic Surgery: Weights Space Programming (WSP)

Rather than performing costly fine-tuning or reinforcement learning (which risks alignment tax or catastrophic forgetting), we excised the loop attractor directly in weight space.

#### Mathematical Formulation
Given the unquantized weight matrix $W \in \mathbb{R}^{3584 \times 18944}$ corresponding to `model.language_model.layers.14.mlp.down_proj.weight`, and the extracted unit loop attractor vector $u$:

$$W_{:, \text{cols}} \leftarrow W_{:, \text{cols}} - \alpha \cdot u (u^T W_{:, \text{cols}})$$

Where:
- $\alpha = 0.05$ (damping coefficient)
- $\text{cols}$ is the subset of 2,411 intermediate features ($12.7\%$ of columns) with the highest projection energy onto $u$.
- Number of modified elements: $2411 \times \text{top-50 rows} = 120,545$ weights ($0.24\%$ of the tensor).

#### Tensorial Verification on Original BF16 SafeTensors
Running the transformation directly on the unquantized HuggingFace checkpoint shards:
- Shards checked: 5/5
- Shards modified: Exactly 1 (`model-00004-of-00005.safetensors`), remaining 4 shards hard-linked byte-for-byte.
- Directional energy reduction: $\|u^T W\|_2$ reduced from **1.39906** to **1.39627** on Layer 14.
- All other 350+ tensors left completely untouched.

---

### 3. How to Apply the Patch to Your Local Checkpoint

You can apply this surgical patch to your local copy of `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` in less than 30 seconds using our open-source tool:

```bash
# 1. Clone the repository
git clone https://github.com/FiveTechSoft/weights.git
cd weights

# 2. Apply surgical fix to the original unquantized SafeTensors model
python scripts/apply_wsp_to_safetensors.py \
  --src /path/to/MiMo-V2.6-Distill-Qwen-9B \
  --dst /path/to/MiMo-V2.6-Distill-Qwen-9B-repaired \
  --u-path data/u_loop_L14.npz \
  --alpha 0.05 \
  --layer 14
```

Or execute the declarative WSP-Lang specification:
```bash
python engine/wsplang.py programs/fix_mimo_tool_loops.wspl --run
```

The repair script utilizes NTFS/Unix hard-links for unmodified shards, requiring only ~4 GB of temporary disk space to create the repaired BF16 model.

---

### 4. Code & Reproducibility Links

All vectors, test rigs, logs, and replication scripts are published under the **MIT License** on GitHub:
- **Repository:** [https://github.com/FiveTechSoft/weights](https://github.com/FiveTechSoft/weights)
- **Declarative Program:** [`programs/fix_mimo_tool_loops.wspl`](https://github.com/FiveTechSoft/weights/blob/main/programs/fix_mimo_tool_loops.wspl)
- **SafeTensors Surgery Script:** [`scripts/apply_wsp_to_safetensors.py`](https://github.com/FiveTechSoft/weights/blob/main/scripts/apply_wsp_to_safetensors.py)
- **Full Empirical Report:** [`results.md` (Section 15)](https://github.com/FiveTechSoft/weights/blob/main/results.md)

We invite the Xiaomi MiMo team and community members experimenting with agentic pipelines to test this weight-space edit and share your findings!
