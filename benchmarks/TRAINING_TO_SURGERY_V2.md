# Training-to-Surgery Benchmark v2: Closed-Form Dual Ridge Compiler

Building upon `v1` (which established baseline instrumentation and oracle replay), `v2` introduces an **actual analytical surgery compiler** operating strictly in weight space without backpropagation or optimizer state.

## 1. What's New in v2

1. **Closed-Form Dual Ridge Surgery:**
   Replaces the trivial oracle replay ($W_0 + \Delta W_{SGD} = W_{SGD}$) with an analytical weight-space synthesis function:
   $$\Delta W_{proj} = (T - W_{0, proj} H) (H^T H + \lambda I)^{-1} H^T$$
   Operates strictly on $(W_0, D_{train})$ with:
   - **0 backpropagation steps**
   - **0 gradients**
   - **0 access to $W_{final}$ or optimizer trajectory**
2. **Layer-Matched Placebo Control:**
   Instead of global $L_2$ scaling (which misallocated noise between embedding and linear projections), `v2` matches Frobenius norms **tensor-by-tensor**.
3. **Cleaned Specification:**
   Eliminated unused/phantom decorative templates (`TEMPLATES_TRAIN` / `TEMPLATES_TEST`), explicitly separating training IDs ($D_{train}$) and holdout IDs ($D_{holdout}$).

## 2. Benchmark Arms

| Arm | Method | Backprop Steps | Access to SGD Delta | Role |
| :--- | :--- | :---: | :---: | :--- |
| **W0 Baseline** | Random orthogonal / normal weights | 0 | No | Untrained baseline |
| **Arm 1: Real Training** | Conventional SGD (`lr=0.08`, 250 steps) | 250 | No | Optimization teacher |
| **Arm 2: WSPLang Surgery** | Regularized dual ridge solve | **0** | **No** | **Compiled closed-form surgery** |
| **Oracle Reference** | Direct delta replay ($W_0 + \Delta W_{SGD}$) | 0 | Yes | Theoretical upper bound |
| **Arm 3: Matched Placebo** | Layer-matched random Gaussian noise | 0 | No | Counterfactual null control |

## 3. Run

```bash
python benchmarks/training_to_surgery_v2.py --device auto
```

Outputs in `runs/training_to_surgery_v2`:
- `report_v2.json`: Metrics across all 5 configurations, tensor deltas, and verdict.
- `trace_v2.json`: Per-step SGD loss curve and gradient $L_2$ norms.
- `weights_v2.pt`: Checkpoints for $W_0$, $W_{SGD}$, $\Delta W_{surg}$, and $\Delta W_{placebo}$.
