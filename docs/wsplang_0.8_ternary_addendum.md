# WSP-Lang 0.8 — ternary addendum: `ternarize` and `imprint_ternary` (2026-11-18)

Base text: `docs/wsplang_specification.md`; 0.8 additions: `docs/wsplang_0.8_spec.md`. This addendum adds ternary
*writing* for NanoWSP-class checkpoints (the micro-transformers of `engine/nanowsp_runtime.py`), measured in
`lab/probes/ternary_nanowsp/PLAN.md`. It does not change the reading of results §45.23 for pretrained LLMs: there,
rounding an edit to {−1, 0, +1} buries it; here the write is *solved* into the ternary alphabet, not rounded.

## Representation
A ternary checkpoint is a directory: `model.safetensors` (float materialisation s·T, so every existing op and the
runtime work unchanged), `ternary.safetensors` (`<name>.codes` int8 ∈ {−1, 0, +1}, `<name>.scales` float32 per output
row), `manifest.json` (source sha256, rule, per-tensor density and ‖W − s·T‖/‖W‖). Ternary tensors: the 2-D projections
inside the layers (`layers.*.attn_{q,k,v,out}`, `layers.*.mlp_{in,out}`). Float: embeddings, output head, norms.
Per-row solve: s from the absmean rule, then alternate T = clip(round(W/s), −1, 1) and s = ⟨W, T⟩/⟨T, T⟩ (each step the
exact minimiser over its variable; the residual never increases), 3 iterations by default.

## Ops (both write weights: new dirs beside the inputs, never over them)
```
step ternarize model=<ckpt .safetensors or dir> rule=absmean|tq2_0 out=<new dir> [tensors=layers.*.mlp_out.weight,...] [iters=3]
step imprint_ternary model=<ternary dir> file=<pairs.json> layer=1 eta=4.0 rounds=3 out=<new dir> [lambda=1e-4] [n=<first n pairs>]
```
- `ternarize` refuses an existing `out` and an `out` inside the input; report = per-tensor density and relative error.
- `imprint_ternary`: the same Moore-Penrose write as `imprint_batch` (ΔW = η·Y·(XᵀX + λI)⁻¹Xᵀ) goes into an accumulator
  A that starts at the ternary matrix; A is projected to {−1, 0, +1}·s by the per-row solve; then `rounds` times the
  deficit the projected matrix still shows on the written keys, (A − P(A))·X, is re-imprinted through the same
  pseudo-inverse and A is projected again (TTF's accumulator, in closed form). `rounds=0` is "edit in float, then
  convert". `file=` accepts a list of `{prompt, target}` or a `lab/data/wspx_*.json` task file (its `train` list;
  `holdout` is scored too). Report: `key_residual` per round, `density`, `recall`, `holdout_accuracy`, `delta_norm`.

## What this does not claim
Nothing about pretrained LLMs or GGUF: the ternary tensors here are NanoWSP's own, written closed-form. How far the
ternary model keeps the float model's associative capacity is the measured question of
`lab/probes/ternary_nanowsp/PLAN.md` (prediction row 46), not an assumption.

## Tests
`tests/test_ternary_write.py` (row solve exactness and monotonicity, checkpoint files and manifest, write guard,
recall after `imprint_ternary`, `rounds=0`, op registration and spec rows), plus `tests/test_wsplang_spec.py`.
