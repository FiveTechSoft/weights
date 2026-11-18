# WSP-Lang 0.8 — addendum: localisation, structural placebo, patches, ternary measurement (2026-11-18)

Base text: `docs/wsplang_specification.md` (§5.0 lists every op the runtime accepts). This file adds what 0.8 brings
over 0.7. Nothing here changes the rules of 0.7: every write goes to a new file or directory beside the inputs, never
over them; every op that writes also writes its manifest; a behaviour claim needs a placebo and a sealed set.

## New ops

| op | kind | module | what it does |
|---|---|---|---|
| `edit_column` | writes weights (new dir) | `engine/column_edit.py` | one column of one matrix: `add`, `subtract`, `scale`, `replace`, `subtract_direction` (the Exp 8 closed form W − α·u·(uᵀW) on the column); structural same-norm placebo beside it; report; manifest; two `.weightpatch.json`; `ediciones` program + npz for `wsp.py add-cols-vectors` on the Q6_K GGUF |
| `edit_neurons` | writes weights (new dir) | `engine/column_edit.py` | `edit_column` over a short list of columns (`neurons=c1,c2,...`); one α for all (per-neuron α not yet) |
| `apply_patch` | writes weights (new dir) | `engine/wsplpatch.py` | applies a `.weightpatch.json` to `target=` into `out=` after verifying `base_tensor_hash` and `delta_full_hash`; pushes onto the session stack |
| `rollback_last` | read / report | `engine/wsplpatch.py` | pops the stack (`<session>/stack.json`), returns the previous state dir; deletes nothing |
| `list_patches` | read / report | `engine/wsplpatch.py` | the session's applied patches in order |
| `commit_manifest` | read / report | `engine/wsplpatch.py` | bundle: base hash, ordered patches, final tensor hashes (`name=`, `include_patches=`) |
| `ternary_survival` | read / report | `engine/ternary.py` | how much of an edit survives ternarisation (ggml TQ2_0 / BitNet absmean; requant / fixed; flips, cos, alpha, noise, k70); results §45.23 |

## Syntax

```
step edit_column model=hf_cache/qwen35-08b-hf tensor=model.language_model.layers.15.mlp.down_proj.weight
     column=507 direction=lab/runs/x/dir.npy alpha=1.0 mode=subtract placebo=true placebo_seed=42
     budget_params=2000 normalize_direction=true out=lab/runs/x/col507
step edit_neurons model=... tensor=... neurons=128,256,512 direction=... alpha=0.8 mode=scale out=...
step apply_patch patch=lab/runs/x/col507/edit.weightpatch.json target=hf_cache/qwen35-08b-hf out=lab/runs/x/applied [session=lab/runs/session_patches]
step rollback_last [session=...]
step list_patches [session=...]
step commit_manifest name=col507_sealed include_patches=true
step ternary_survival base=hf_cache/qwen35-08b-hf delta=data/exp8_edit_recovered.npz layers=15,19 rule=tq2_0 mode=requant scales=1,2,5,10,20 out=lab/runs/ternary/exp8_tq2_0_requant.json
```

## Guards (runtime, not prompt)

- `budget_params`: the op aborts before any write when `params_touched` (counted on the realised delta, so `scale` and
  `replace` report the true count) exceeds it.
- `evaluate=true` without `placebo=true` is refused (`validate_request`): a behaviour claim needs the structural placebo.
- Structural placebo: same columns, same per-column norm of the realised delta, same support (identical
  `params_touched`), direction orthogonal to the given direction and to the realised column; sign-flip fallback when a
  column's support is too small for an orthogonal direction.
- `apply_patch` refuses a stale base (`base_tensor_hash`), a tampered delta (`delta_full_hash`), `out == target`, and
  an empty stack on `rollback_last`.
- The stack lives on disk: rollback works after a process restart; nothing in an input model is ever written.

## Report fields (`edit_column`)

`op`, `tensor`, `column` (or `columns`), `params_touched`, `delta_norm`, `rel_delta_norm`, `cosine_with_direction`,
`alpha`, `mode`, `placebo_seed`, `placebo_delta_norm`, `placebo_params_touched`, `budget_params_ok`, `col_norms`,
tensor hashes before and after (manifest).

## `.weightpatch.json` v1.0

`wsplang_patch_version`, `created`, `base_checkpoint_hash`, `base_tensor_hash`, `tensor_name`, `op`, `params`
(`column`, `mode`, `alpha`, `direction_hash`, `normalize_direction`), `delta_compact` (`type` column_vector | neuron_set,
columns, npz path), `delta_full_hash`, `experiment_id`, `parent_patches`, `notes`, `id` (sha256 prefix of the patch body).

## What 0.8 does not claim

No behaviour result comes with these ops. The dry run (`lab/probes/column_edit/PLAN.md`) rebuilds the MIT Exp 8 edit
through `edit_column` from `data/exp8_edit_recovered.npz` and, applied by `wsp.py`, reproduces
`servidos/qwen35-08b__exp8-replica.gguf` byte for byte (whole-file sha256 identical): the op is exact, nothing more.
Any claim about a column edit improving a model needs the dev / placebo / sealed protocol of results §43.17. Ternary
weights: results §45.23 shows our edits do not fit in ternary codes at their working dose; `ternary_survival` measures,
there is no ternary writer.

## Tests

`tests/test_column_edit.py` (11), `tests/test_wsplpatch.py` (4), `tests/test_ternary.py` (11), plus
`tests/test_wsplang_spec.py` (every implemented op in the table with its kind, version in sync) and
`tests/test_cc_compile.py` / `tests/test_mcp_server.py` (op lists). Run with `python -m unittest`.

## History

- 0.7 → 0.8 (2026-11-18): the seven ops above; results §45.22 (D4k pruning, noise-limited), §45.23 (ternary), §45.24
  (discrete per-row writer prototype, CPU); plans fixed before every run in `lab/probes/{exp8_prune,ternary,
  column_edit,discrete_writer,cc_bias,skill2,preserve}/PLAN.md`; prediction rows 41-45.
