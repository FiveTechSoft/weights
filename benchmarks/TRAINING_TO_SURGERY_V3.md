# Training-to-Surgery Benchmark v3: Multi-Seed Meta-Transfer Compiler

## 1. Motivation & Scientific Problem

In `v1`, we established the baseline instrumentation (measuring training curves, Frobenius norms, and comparing against an oracle direct delta replay). In `v2`, we formulated an analytical closed-form dual ridge solve derived by a human.

However, the foundational hypothesis of **WSPLang training-to-surgery** is:
> *Can the surgery compiler LEARN the weight modification operator directly from optimization traces across multiple training runs, and apply it to a novel, unseen task without seeing that task's SGD delta or running iterative backpropagation?*

Benchmark `v3` explicitly answers this question with a multi-seed meta-learning protocol.

---

## 2. Mathematical Formulation

Let $\mathcal{T}$ be a task family of symbol associations.

### 2.1 Meta-Training Phase (Learning from Traces)
We observe $K$ training tasks $\tau_1, \tau_2, \dots, \tau_K$ (with independent random seeds and permutation mappings):
1. For each task $k$, we start from a shared base model $W_0$ and run conventional SGD for $S$ steps on $D^{(\tau_k)}_{train}$.
2. We extract:
   - Target parameter displacement: $\Delta W_{proj, SGD}^{(k)} = W^{(k)}_{final} - W_{0, proj}$.
   - Initial gradient/residual features: $G^{(k)} = (R^{(k)})^T H^{(k)} \in \mathbb{R}^{C \times d}$, where $H^{(k)} = \phi(X^{(\tau_k)}_{train}; W_0)$ and $R^{(k)} = T^{(k)} - H^{(k)} W_{0, proj}^T$.
3. We solve the closed-form meta-regression problem over the $K$ observed optimization trajectories:
   $$M^* = \arg\min_M \sum_{k=1}^K \| G^{(k)} M - \Delta W_{proj, SGD}^{(k)} \|_F^2 + \lambda_{meta} \|M\|_F^2$$
   yielding the normal equations:
   $$M^* = \left( \sum_{k=1}^K (G^{(k)})^T G^{(k)} + \lambda_{meta} I \right)^{-1} \left( \sum_{k=1}^K (G^{(k)})^T \Delta W_{proj, SGD}^{(k)} \right)$$

### 2.2 Meta-Evaluation on Unseen Tasks (0 Delta Access)
Given a new, unseen task $\tau_{test}$ (with a novel seed not present in meta-training):
- The Meta-Surgery Compiler receives **ONLY** $(W_0, D^{(\tau_{test})}_{train})$.
- The compiler has **ZERO ACCESS** to $\Delta W_{SGD}^{(\tau_{test})}$.
- The compiler executes **0 gradient iterations / backpropagation steps**.
- The weight modification is synthesized in a single algebraic evaluation:
  $$\Delta W_{meta}^{(\tau_{test})} = G^{(\tau_{test})} M^*$$
  $$W_{meta} = W_0 + \Delta W_{meta}^{(\tau_{test})}$$

---

## 3. Benchmark Arms

| Arm | Description | Backprop Steps | Target SGD Delta Access | Role |
| :--- | :--- | :---: | :---: | :--- |
| **W0 Baseline** | Untrained base model | 0 | No | Negative baseline |
| **Arm 1: SGD Teacher** | 250 steps of SGD on test task | 250 | Teacher (itself) | Optimization target |
| **Arm 2: Meta-Surgery Compiler** | Applied via learned $M^*$ | **0** | **NO (0 access)** | **Learned meta-surgery** |
| **Arm 3: Handcrafted Ridge** | Analytical formula solve | 0 | No | Classical reference |
| **Arm 4: Matched Placebo** | Layer-matched Gaussian noise | 0 | No | Null hypothesis control |

---

## 4. Execution & Artifacts

Run the benchmark:
```bash
python benchmarks/training_to_surgery_v3_meta_transfer.py --device auto --n_meta_tasks 8 --n_test_tasks 10
```

Outputs in `runs/training_to_surgery_v3/`:
- `report_v3.json`: Aggregated metrics (mean $\pm$ std across test tasks), delta norms, cosine similarities, and verdict.
- `eval_traces_v3.json`: Per-task breakdown across all unseen test seeds.
- `meta_operator_v3.pt`: Saved meta-operator tensor $M^*$.
