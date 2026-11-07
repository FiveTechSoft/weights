# Audit of WSPLang weight-editing and inspection ops (2026-11-05)

Each script runs ops on throwaway copies of small models and measures the effect independently of what the
op reports. Results were taken at commit `456c6d2`, before the write guard (`guard_model_write` in
`engine/wsplang.py`, tests in `tests/test_wsp_write_guard.py`).

- `edit_ops/`: writers (`project_out`, `leace_erase`, `imprint_*`, `memit_imprint`, `tie_weights`,
  `permute_align`, `export_control_vector`, ...). `results.json`, `results_controlvector.json`.
- `inspect_ops/`: readers and asserts (`peek`, `diff`, `audit`, `spectral_*`, `assert_*`,
  `architectural_drift`, semantic labels, `decomp/mimo_head.py`). Run `make_models.py` first; outputs in `out/`.

Fixed after the audit: input models are no longer overwritten; budgets and protected tensors apply to every
write; failed asserts and failed steps exit with code 1. Still open: `permute_align` applies the wrong
permutation, `jepa_steer` reports a cosine it does not reach, canned outputs (`architectural_drift`,
`assert_orthogonal_preservation`, `compare`, `evaluate`), `mimo_head.py` reads heads with `reshape`.
