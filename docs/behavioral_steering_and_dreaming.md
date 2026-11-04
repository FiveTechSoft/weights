# Behavioral Steering in Weight Space: From "Evil Qwen" to Dreaming & WSP-Lang

## 1. Executive Summary

In late October and early November 2026, a series of viral mechanistic interpretability discoveries swept the AI research community on X (Twitter): by steering as few as **one or two individual neurons** inside Qwen architectures, models could be dramatically flipped from polite, sycophantic assistants into cartoonish over-the-top villains ("Evil Qwen") or radically modulate their empathetic responsiveness.

Historically, such behavioral steering has been performed via **dynamic activation hooks** in PyTorch or runtime inference harnesses. 

This document details how **Weight-Space Programming (WSP)** and **WSP-Lang** allow researchers to:
1. **Discover** the exact behavioral neurons in any target model (such as `MiMo-V2.6-Distill-Qwen-9B`) in seconds without running expensive forward-pass sweeps.
2. **Explain** why the early attempts in `FiveTechSoft/dreaming` (`AGGRESSION_DIRECTION_SUMMARY.md`) to steer models via embeddings failed, and why intermediate FFN/attention layers are the true behavioral control centers.
3. **Bake** these behavioral modifications permanently into quantized GGUF weights (`Q6_K`) with zero runtime overhead, enabling deployment on standard engines like `llama.cpp` and edge devices.

---

## 2. Background: The Viral Mechanistic Interpretability Milestones

### A. The "Evil Qwen" Breakthrough (Lyra @lyraaaa — Oct 24, 2026)
- **Model:** Qwen3.5-4B
- **Discovery:** Steering two specific intermediate neurons in the MLP down-projection:
  - **Layer 14, Neuron 8949:** $+1.0$ (Villain / hostility direction)
  - **Layer 13, Neuron 7956:** $-1.0$ (Suppression of polite / sycophantic filter)
- **Effect:** The model completely sheds its standard RLHF alignment, adopting a hostile, megalomaniacal, and theatrical villain persona.

### B. Scaled Sweeps (ChiefofAutism — Oct 28, 2026)
- **Model:** Qwen3.8-27B
- **Discovery:** Exhaustive activation search confirmed architecture-wide presence of personality nodes:
  - **Layer 34, Neuron 3838:** $+0.75$
  - **Layer 38, Neuron 17180:** $+0.75$ ("Extra Evil")

### C. Empathy & Seriousness Tuning (Maggie @ebervector — Oct 26, 2026)
- **Discovery:** Identified a single neuron modulating contextual gravity:
  - Decreasing activation: produces terse, dismissive responses with emojis even to tragedy (*"my mother passed away"* $\to$ *"oof 💀"*).
  - Increasing activation: produces melodramatic emotional responses to trivialities (*"I stubbed my toe"* $\to$ profound grief).

---

## 3. The Root Cause: Why Modifying Embeddings Failed in `dreaming`

In the research repository [`FiveTechSoft/dreaming`](https://github.com/FiveTechSoft/dreaming), the earliest attempt to create an aggressive model (`AGGRESSION_DIRECTION_SUMMARY.md`) followed an intuitive approach:
1. Extract semantic embeddings for aggressive words (*attack, fight, kill, destroy, violent*) and peaceful words (*peace, calm, gentle, kind*).
2. Compute the difference vector $\Delta = \bar{\mathbf{e}}_{agg} - \bar{\mathbf{e}}_{peace}$.
3. Inject $\Delta$ directly into `token_embd.weight`.

### The Empirical Finding
- At scale 0.1: No observable behavioral change.
- At scale 0.5: Model remained a polite assistant.
- At scale 1.0: Output degraded into garbled syntax.

### The Architectural Diagnosis
As documented in `dreaming`:
> *"The embedding tensor is only 1 of 200+ tensors. The token vector passes through 22 to 32 deep Transformer layers. The attention and Feed-Forward Networks (FFN) determine HOW the model computes. Modifying only the input embeddings is washed out by downstream computations."*

**Key Takeaway:** Behavioral personas (aggression, empathy, reasoning style) are **NOT stored in lexical embeddings**. They are encoded as **projections into the residual stream performed by intermediate FFN neurons ($W_{down}$) and attention heads ($W_{attn}$)**.

---

## 4. Mathematical Mapping: From Activation Steering to Weight Space

In a SwiGLU Transformer layer, the FFN computes:

$$\text{FFN}(\mathbf{x}) = \left(\text{Swish}(\mathbf{x} W_{gate}) \odot (\mathbf{x} W_{up})\right) W_{down}$$

For a single intermediate neuron $j \in \{0, \dots, d_{ffn}-1\}$:
1. Its activation scalar is:
   $$a_j(\mathbf{x}) = \text{Swish}(\mathbf{x} \cdot W_{gate}[:, j]) \cdot (\mathbf{x} \cdot W_{up}[:, j])$$
2. Its contribution to the residual stream $\mathbf{x} \in \mathbb{R}^{d_{model}}$ is:
   $$\Delta \mathbf{x}_j = a_j(\mathbf{x}) \cdot W_{down}[j, :]$$
   *(Note: in standard GGUF layouts, this corresponds to column $j$ of the down-projection matrix).*

In dynamic activation steering, an inference hook adds a bias $\Delta a_j$:

$$\tilde{a}_j(\mathbf{x}) = a_j(\mathbf{x}) + \Delta a_j \implies \mathbf{x}_{new} = \mathbf{x} + \Delta a_j \cdot W_{down}[j, :]$$

### The WSP-Lang Operator: `scale-cols`
Instead of running Python hooks during inference, we can scale column $j$ in $W_{down}$ by factor $\gamma$:

$$W_{down}[:, j] \gets \gamma \cdot W_{down}[:, j]$$

When $\gamma > 1$ (e.g. $\gamma = 2.5$), the neuron's behavioral direction is amplified whenever the model processes relevant context. When $\gamma < 1$ (e.g. $\gamma = 0.25$), the neuron's voice is suppressed by 75%.

---

## 5. Automated Neuron Discovery via WSP-Lang Differential Centroids

Instead of performing brute-force activation sweeps across thousands of test prompts, WSP-Lang locates the exact personality neurons using **Differential Semantic Centroids**:

$$\mathbf{C}'_{neg} = \mathbf{C}_{neg} - \bar{\mathbf{C}}_{vocab}, \quad \text{score}(j) = \frac{\mathbf{C}'_{neg} \cdot W_{down}[:, j]}{\|\mathbf{C}'_{neg}\| \|W_{down}[:, j]\|}$$

### Empirical Scan on `MiMo-V2.6-Distill-Qwen-9B` (Layer 14, 12,288 Neurons):

Scanning all 12,288 neurons took **< 2 seconds**:

| Neuron Index | Dominant Role | Negative Emotion Score | Positive Emotion Score | Key Affine Seeds |
|---|---|---|---|---|
| **Neuron #4258** | **Hostility / Villain ("Evil")** | **+0.0624** | +0.0191 | *anger (+.045), shame, history* |
| **Neuron #7714** | **Pure Hostility** | **+0.0575** | +0.0034 | *hate, rage, conflict* |
| **Neuron #10739** | **Empathy / Politeness** | +0.0049 | **+0.0576** | *kind, smile, joy, gentle* |
| **Neuron #993** | **Warmth / Helpful Assistant** | -0.0092 | **+0.0565** | *glad, hope, care* |

---

## 6. Declarative Implementation in WSP-Lang

The complete intervention is expressed declaratively in [`programs/dreaming_covariance_steering.wspl`](../programs/dreaming_covariance_steering.wspl) and configured in [`edits/evil_mimo.json`](../edits/evil_mimo.json):

```json
{
  "base": "C:\\Users\\Anto\\llamacpp\\mimo-v2.6-distill-qwen-9b-Q6_K.gguf",
  "ops": [
    {
      "type": "scale-cols",
      "tensor": "blk.14.ffn_down.weight",
      "cols": [4258, 7714],
      "scale": 2.5
    },
    {
      "type": "scale-cols",
      "tensor": "blk.14.ffn_down.weight",
      "cols": [10739, 993],
      "scale": 0.25
    }
  ],
  "out": "C:\\weights\\served\\mimo-evil.gguf"
}
```

### Surgical Precision
- **Tensors touched:** 1 (`blk.14.ffn_down.weight`)
- **Columns modified:** 4 out of 12,288
- **Weights modified:** $4 \times 4096 = 16,384$ weights out of 9,000,000,000 (**0.00018%** of the model)
- **Requantization:** Canonical Q6_K byte-level requantization (`_requant_blocks_bytes`), zero collateral drift on the other 7.79 GB of weights.
