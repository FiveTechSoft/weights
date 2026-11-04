# Training-to-Surgery Benchmark v1

This experiment treats conventional training as an empirical teacher for direct
weight surgery. It is intentionally separate from the WSPLang runtime.

## First task: symbol associations

A tiny embedding + linear classifier learns deterministic symbol-to-label
associations using ordinary SGD. The run stores initial/final weights, their
delta, per-step loss and gradient norm, and a machine-readable report.

The report compares:
- **trained model** — result of conventional SGD;
- **direct delta replay** — applies the observed training delta to the initial
  weights (an oracle/reference baseline, not a WSPLang-learned intervention);
- **norm-matched random placebo** — random weight perturbation with equal global
  L2 norm.

The first goal is to validate instrumentation and controls. Delta replay matching
the trained model is expected by construction and is not evidence of a learned
surgery. The next milestone is to learn a surgery rule from multiple traces and
test it on unseen seeds/tasks without copying the target run's delta.

## Run

From the repository root:

```bash
python benchmarks/training_to_surgery_v1.py --device auto
```

Optional controls include `--seed`, `--steps`, `--lr`, and `--out`.
PyTorch is required. The script uses CUDA when available, otherwise CPU.

Outputs:
- `weights.pt`: initial/trained states, observed delta, placebo, mapping/config;
- `trace.json`: per-step loss, gradient norm, learning rate and optimizer;
- `report.json`: scores and interpretation.

## Important limitation

The tiny model receives integer symbol IDs, not natural-language templates. The
task isolates association learning and does not yet test compositional
generalization, preservation of unrelated capabilities, or a learned WSPLang
compiler. Those belong to later benchmark stages.
