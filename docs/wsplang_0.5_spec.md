# WSP-Lang 0.5 Specification: Covariance-Regularized Memory Synthesis, LEACE Erasure & Decoupled DeltaNet

**Status:** Stable Release  
**Version:** `0.5.0`  
**Reference Implementation:** [`engine/wsplang.py`](../engine/wsplang.py)  
**Verification Test Suite:** [`tests/test_wsplang_v5.py`](../tests/test_wsplang_v5.py) (34/34 passing)  

---

## 1. Theoretical Foundations & ArXiv Literature Grounding

WSP-Lang 0.5 unifies Weight-Space Programming with four foundational breakthroughs published on ArXiv:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 WSP-Lang 0.5 Architecture                              │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 1. LEACE Concept Erasure │ 2. MEMIT/EMMET Synthesis │ 3. Gated DeltaNet-2 Decoupling   │
│ Belrose et al. (NeurIPS) │ Meng / Gupta (ICLR 2023) │ Yang (ICLR 2025) / NVIDIA (2026) │
│ Closed-form min-MSE      │ Covariance-regularized   │ Channel-wise independent erase   │
│ concept elimination      │ mass associative memory  │ and write gate updates           │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

1. **LEACE Closed-Form Concept Erasure** ([arXiv:2306.03819](https://arxiv.org/abs/2306.03819) — Belrose et al., NeurIPS 2023):
   Replaces heuristic projection subtraction with the analytical, closed-form affine projector that eliminates target concept leakage with provably minimal Frobenius distortion on unrelated representations.
2. **MEMIT & EMMET Covariance-Regularized Editing** ([arXiv:2210.07229](https://arxiv.org/abs/2210.07229) — Meng et al., ICLR 2023; [arXiv:2403.14236](https://arxiv.org/abs/2403.14236) — Gupta et al., 2024):
   Solves the condition number explosion barrier ($\kappa(X) \to \infty$) discovered in NanoWSP by introducing uncentered covariance regularization $C_0 = \mathbb{E}[k k^T]$ into the linear associative memory objective.
3. **Gated DeltaNet-2 Decoupled Erase & Write** ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464) — Songlin Yang et al., ICLR 2025; [arXiv:2605.22791](https://arxiv.org/abs/2605.22791) — NVIDIA 2026):
   Decouples the erase phase from the write phase in recurrent linear attention and MLP layers, preventing the write vector from re-contaminating the erased subspace.
4. **Titans Test-Time Memory Surprisal Monitor** ([arXiv:2501.00663](https://arxiv.org/abs/2501.00663) — Behrouz, Zhong, Mirrokni, Google Research 2025):
   Monitors sequence perplexity and repetition metrics to dynamically trigger transient Fast Weight overlays when runaway attractors are detected at test time.

---

## 2. Mathematical Formulations

### 2.1 LEACE Closed-Form Concept Erasure (`leace_erase`)
Given target concept vector $u \in \mathbb{R}^d$ and feature covariance matrix $\Sigma = \mathbb{E}[x x^T]$:
$$P_{\text{LEACE}} = I - \frac{\Sigma u u^T}{u^T \Sigma u}$$

When applied to matrix $W$:
$$W \leftarrow P_{\text{LEACE}} W$$

* **Target Leakage:** $u^T W_{\text{new}} \equiv 0.0$ (exact zero leakage).
* **Optimality:** $P_{\text{LEACE}}$ strictly minimizes $\mathbb{E}[\|P x - x\|_2^2]$ subject to $u^T (P x) = 0$.

### 2.2 MEMIT / EMMET Covariance-Regularized Synthesis (`cov_imprint` / `memit_imprint`)
Given key matrix $K = [k_1, \dots, k_M] \in \mathbb{R}^{d_{in} \times M}$ and target values $V = [v_1, \dots, v_M] \in \mathbb{R}^{d_{out} \times M}$:
$$\Delta W = (V - W_0 K) (K^T K + \lambda I)^{-1} K^T$$

Or with explicit uncentered background covariance $C_0$:
$$\Delta W = (V - W_0 K) K^T (K K^T + \lambda C_0)^{-1}$$

* **Condition Number Stabilization:** Adding $\lambda I$ caps the maximum condition number $\kappa \le \frac{\sigma_1^2 + \lambda}{\lambda}$, preventing $\|\Delta W\|_F$ explosion when scaling $M > 20$.

### 2.3 Gated DeltaNet-2 Decoupled Erase & Write (`decoupled_delta`)
Decouples the erase gate $\beta_{erase}$ from write gate $\beta_{write}$ while enforcing orthogonal preservation on the write vector:
$$v_{\perp} = v_{\text{write}} - (u_{\text{erase}}^T v_{\text{write}}) u_{\text{erase}}$$
$$W_{\text{new}} = \left( I - \beta_{erase} \frac{u_{\text{erase}} u_{\text{erase}}^T}{\|u_{\text{erase}}\|^2} \right) W + \beta_{write} (v_{\perp} \otimes k_{\text{write}}^T)$$

* Guarantees $u_{\text{erase}}^T W_{\text{new}} \equiv 0$ regardless of the chosen write direction $v_{\text{write}}$.

---

## 3. Declarative Grammar & Syntax in WSPLang 0.5

```wspl
experiment advanced_fwp_and_leace_surgery:
    model: MiMo-V2.6-Distill-Qwen-9B
    objective: surgical_concept_erasure_and_stable_memory_imprinting

    budgets:
        max_modified_weights = 100000
        allow_write = true

    steps:
        # 1. LEACE Closed-Form Concept Erasure on Layer 14 down_proj
        step leace_erase model.language_model.layers.14.mlp.down_proj.weight vector=data/u_loop_L14.npz

        # 2. Gated DeltaNet-2 Decoupled Erase & Write on SSM Recurrent Layer 0
        step decoupled_delta blk.0.ssm_out.weight erase_vector=data/u_loop_L14.npz write_key=data/k_anchor.npz write_value=data/v_anchor.npz beta_erase=1.0 beta_write=1.5

        # 3. MEMIT Covariance-Regularized Memory Batch Synthesis on NanoWSP
        step cov_imprint layers.1.mlp_out.weight file=data/factual_pairs.json lambda=0.01

        # 4. Titans Test-Time Memory Surprisal Monitor
        step surprise_fwp prompt="ls app ls app ls app ls app" threshold=2.0 metric=repetition_entropy
```

---

## 4. Verification & Unit Tests

Validated in [`tests/test_wsplang_v5.py`](../tests/test_wsplang_v5.py):
* `test_leace_concept_erasure_closed_form`: Proves $100.0000\%$ erasure efficiency with $u^T W_{\text{repaired}} < 10^{-5}$.
* `test_memit_covariance_regularization`: Proves condition number stabilization ($\|\Delta W\|_F < 50.0$) on degenerate key matrices with raw condition number $> 1000$.
* `test_decoupled_delta_gated_deltanet2`: Proves decoupled erasure of $u_{\text{erase}}$ without re-contamination by $v_{\text{write}}$.
* `test_surprise_fwp_titans_trigger`: Verifies automatic trigger of dynamic memory overlay under high repetition score.
