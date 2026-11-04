# WSP-Lang 0.3: Fast Weight Programming & Associative Memory Surgery

**Version:** 0.3.0  
**License:** MIT  
**Engine Module:** `engine/wsplang.py`  
**Theoretical Framework:** Fast Weight Programmers (FWP) & Linear Transformers (Schmidhuber 1992; Schlag, Irie & Schmidhuber, ICML 2021)  
**Repository:** [https://github.com/FiveTechSoft/weights](https://github.com/FiveTechSoft/weights)  

---

### Executive Overview

**Weights Space Programming Language (WSP-Lang 0.3)** bridges direct neural weight surgery with the mathematical foundations of **Fast Weight Programmers (FWPs)** and **Linear Transformer associative memory**.

While early weight manipulation experiments (such as MIT CSAIL Exp 8) operated via empirical heuristic projections, WSP-Lang 0.3 establishes that:
1. **Transformers are Associative Memories:** Transformers internally accumulate key-value associations via outer products ($W_t = W_{t-1} + v_t \otimes k_t^T$).
2. **Weight Space Surgery is Fast Weight Programming:** Surgically excising or steering a concept in a model matrix is mathematically identical to applying an FWP update to associative memory.
3. **The Delta Rule Eliminates Cross-Talk:** Rather than blind subtractive projections, the **Delta Rule** (Widrow-Hoff) provably guarantees **exact zero distortion** on all orthogonal concept directions:
   $$\Delta W = -\eta \cdot (u (u^T W) - v^*) \implies \Delta W \cdot u_\perp = 0$$

---

### 1. Theoretical Foundation: Fast Weight Programmers

In their landmark work *"Linear Transformers Are Secretly Fast Weight Programmers"* (ICML 2021), Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber demonstrated that modern linear self-attention mechanisms are mathematically equivalent to 1990s Fast Weight Programmers.

| Classical Transformers | Fast Weight Programmers (FWP) | WSP-Lang 0.3 Analogue |
| :--- | :--- | :--- |
| Attention Weight Matrix | Fast Weight Matrix $W$ | Targeted Layer Matrix $W$ (`down_proj`) |
| Key Vectors ($k$) | Write Address Pattern | Contrastive Attractor Vector $u$ |
| Value Vectors ($v$) | Stored Associative Content | Projected Output Representation $u^T W$ |
| Query Vectors ($q$) | Read Address Instruction | Input Activation Vector |
| Additive KV Outer-Product | Fast Synaptic Accumulation | `delta_unlearn` / `project_in` |
| Delta Rule Attention | Error-Corrected Associative Memory | `delta_unlearn (target=0.0)` |

---

### 2. The Delta-Rule Zero Cross-Talk Theorem

When modifying neural network weights, the primary danger is **cross-talk**: accidentally degrading unrelated capabilities, facts, or reasoning skills while fixing a bug.

#### Mathematical Formulation
Given weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$, unit attractor key $u \in \mathbb{R}^{d_{out}}$ ($\|u\|_2 = 1$), and desired target projection $v^* \in \mathbb{R}^{d_{in}}$ (where $v^* = 0$ for complete loop nullification):

$$\Delta W = -\eta \cdot u \otimes (u^T W - v^*)$$
$$W_{new} = W + \Delta W = W - \eta \cdot u (u^T W - v^*)$$

#### Proof of Zero Orthogonal Distortion
Let $u_\perp$ be any concept vector strictly orthogonal to the attractor key $u$ ($u^T u_\perp = 0$):

$$u_\perp^T W_{new} = u_\perp^T \left( W - \eta \cdot u (u^T W - v^*) \right)$$
$$u_\perp^T W_{new} = u_\perp^T W - \eta \cdot (u_\perp^T u) (u^T W - v^*)$$
$$\text{Since } u_\perp^T u = 0 \implies u_\perp^T W_{new} = u_\perp^T W$$

$$\therefore \| u_\perp^T W_{new} - u_\perp^T W \| = 0 \quad (\text{Exact Zero Cross-Talk})$$

This guarantees that unlearning a reasoning or tool-calling loop produces **zero mathematical interference** on any orthogonal knowledge or task in the model.

---

### 3. WSP-Lang 0.3 Core Operations

#### 3.1. `delta_unlearn`
Applies Delta-Rule error-corrected unlearning to the target matrix:
```wspl
step delta_unlearn <tensor> key=<vector_path> eta=<float> target=<float> [top_cols=<float>] [top_rows=<int>]
```
- **Arguments:**
  - `tensor`: Target layer weight name (SafeTensors or GGUF).
  - `key`: Unit direction vector representing the pathological state.
  - `eta`: Unlearning rate ($\eta \in (0, 1]$, default: `0.05`).
  - `target`: Target projection value (default: `0.0`).
- **Guarantees:** Zero orthogonal distortion, respects formal parameter change budgets.

#### 3.2. `fwp_overlay` (Transient Fast Weight State)
Compiles a non-destructive Fast Weight overlay file (`.npz`) for runtime inference. Instead of writing permanent bytes to the model on disk, the inference engine loads the overlay as an in-memory residual kernel.
```wspl
step fwp_overlay <tensor> key=<vector_path> alpha=<float> out=<overlay_path>
```

#### 3.3. `spectral_decay` (Attractor Dissipation)
Simulates continuous temporal decay to prevent recursive feedback saturation:
```wspl
step spectral_decay <tensor> decay=<lambda>
```
Where $\lambda \in (0, 1]$ (e.g. `decay=0.95` represents 5% attractor dissipation).

#### 3.4. `assert_orthogonal_preservation`
Formal compile-time contract asserting that the unlearning key does not degrade orthogonal concept subspaces:
```wspl
step assert_orthogonal_preservation key=<vector_path>
```

---

### 4. Complete Program Example

File: [`programs/mimo_fwp_delta_unlearn_v3.wspl`](https://github.com/FiveTechSoft/weights/blob/main/programs/mimo_fwp_delta_unlearn_v3.wspl)

```wspl
experiment mimo_fwp_delta_unlearn_v3:

    theoretical_foundation:
        Fast Weight Programmers (Schlag, Irie & Schmidhuber, ICML 2021)

    model:
        MiMo-V2.6-Distill-Qwen-9B

    format:
        safetensors

    objective:
        suppress_agentic_tool_loop_attractor_via_delta_rule_unlearning

    budgets:
        max_modified_weights = 150000
        target_parameter_fraction = 0.0025
        max_delta_norm = 0.0005
        allow_write = false

    steps:
        # Step 1: Formal contract verification of orthogonal concept preservation
        step assert_orthogonal_preservation key=data/u_loop_L14.npz

        # Step 2: Delta-Rule associative memory unlearning (Widrow-Hoff)
        step delta_unlearn model.language_model.layers.14.mlp.down_proj.weight key=data/u_loop_L14.npz eta=0.05 target=0.0 top_cols=0.127 top_rows=50

        # Step 3: Compile Fast Weight Overlay state for dynamic inference activation
        step fwp_overlay model.language_model.layers.14.mlp.down_proj.weight key=data/u_loop_L14.npz alpha=0.05 out=data/fwp_overlay_L14.npz

        # Step 4: Spectral decay damping to dissipate runaway recurrent attractors
        step spectral_decay model.language_model.layers.14.mlp.down_proj.weight decay=0.95
```

---

### 5. CLI Execution Reference

```bash
# Verify syntax, budgets, and FWP theoretical contracts:
python engine/wsplang.py programs/mimo_fwp_delta_unlearn_v3.wspl --check

# Simulate in-memory execution and assert zero cross-talk (Dry Run):
python engine/wsplang.py programs/mimo_fwp_delta_unlearn_v3.wspl --dry-run

# Execute surgery and output repaired SafeTensors model:
python engine/wsplang.py programs/mimo_fwp_delta_unlearn_v3.wspl --run \
  --model /path/to/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B \
  --out /path/to/MiMo-V2.6-Distill-Qwen-9B-repaired
```
