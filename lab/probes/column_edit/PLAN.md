# `edit_column` / `edit_neurons` and `.weightpatch.json` for WSP-Lang — built 2026-11-18 (roadmap gap 21)

**What this is.** Two pieces of runtime, no behaviour claim: (a) an op that edits one or a few columns (neurons) of
one matrix with a dose and a direction, always writing the structural same-norm placebo beside it, with a report, a
manifest, and the ediciones program so the same edit lands in the Q6_K GGUF through the existing `add-cols-vectors`
op of `wsp.py`; (b) a text patch format bound to the base tensor's sha256 and to the delta's sha256, with apply /
rollback / reconstruct / bundle on an on-disk session stack. Tests written first (`tests/test_column_edit.py` 11,
`tests/test_wsplpatch.py` 3). Code: `engine/column_edit.py`, `engine/wsplpatch.py`. Registration for `wsplang.py`:
`WSPLANG_PATCH.md` (applied by the coordinator).

**What it does not claim.** Nothing here says an edit changes behaviour. That needs its own plan: dev items, the
placebo arm this op produces, a sealed set, the prediction row before the run. The op refuses `evaluate=true` without
`placebo=true`. "Reproduce Hu et al.'s +29 on MATH-500 with < 2,000 parameters" is not adopted as a criterion here:
that phenomenon is the 0.8B's reasoning loops, and our 0.8B harness has a ± 5 run-to-run band at n = 100 (§45.22).

## Semantics (HF orientation: W is (out, in); a column is a neuron of the in dimension)
| mode | effect on each selected column c |
|---|---|
| `add` | W[:, c] += alpha · dir |
| `subtract` | W[:, c] −= alpha · dir |
| `scale` | W[:, c] *= alpha (no direction) |
| `replace` | W[:, c] = alpha · dir |
| `subtract_direction` | W[:, c] −= alpha · u · (uᵀ W[:, c]), u = dir/‖dir‖ (the MIT Exp 8 closed form, W = W₀ − 6·u·(m ⊙ uᵀW₀)) |

A 1-D `dir` is normalised unless `normalize_direction=false`; a 2-D direction (one vector per column, as
`data/exp8_edit_recovered.npz`) is used as given. The realised delta dW is what everything else is computed from.

**Structural placebo** (`structural_placebo(dW, cols, dir, seed)`): same columns, same per-column ‖dW[:, c]‖, same
support (so `params_touched` is identical by construction), random direction orthogonal to the real direction and to
the realised column, seeded. When a column's support is too small for an orthogonal direction (one or two entries),
the placebo is the sign flip of that column (same norm, same support; cos −1 instead of 0).

**Report** (`<out>/edit_column.report.json`): `op, tensor, column, params_touched, delta_norm, rel_delta_norm,
cosine_with_direction, alpha, mode, placebo_seed, placebo_delta_norm, placebo_params_touched, budget_params,
budget_params_ok, col_norms, shape`. **Manifest** (`edit_column.manifest.json`): report, `tensor_hashes`
{base, edit, placebo}, both patch paths, program paths, `params_touched_identical`, `behaviour_claim: none`.

**Guards, in order, before anything is written:** `params_touched > budget_params` → `BudgetError`, nothing written;
`out` / `out_placebo` must be new (no `.safetensors` inside) and outside the input model; the input is never opened
for writing. On Windows the npz handles are closed before any rename.

## WSP-Lang syntax (0.8, proposed; see WSPLANG_PATCH.md)
```
step edit_column  model=<hf dir> tensor=<name> column=507 direction=<npy|npz> alpha=1.0
                  mode=add|subtract|scale|replace|subtract_direction placebo=true placebo_seed=42
                  out=<dir> budget_params=2000 normalize_direction=true
step edit_neurons model=<hf dir> tensor=<name> neurons=12,507,2210 direction=<npy|npz> alpha=1.0 mode=add out=<dir>
step apply_patch  patch=<file.weightpatch.json> target=<model dir> out=<new dir> [session=<dir>]
step rollback_last [session=<dir>]      step list_patches [session=<dir>]
step commit_manifest name=<bundle> include_patches=true [session=<dir>]
```
Outputs of `edit_column`: `<out>/` (edited safetensors dir), `<out>_placebo/`, `<out>/edit.json` + `edit.npz` and
`edit_placebo.json` + `edit_placebo.npz` (ediciones programs for `python wsp.py edit <base.gguf> <program>`),
`edit.weightpatch.json`, `edit_placebo.weightpatch.json`, report, manifest. `edit_neurons` is `edit_column` with a
list; one alpha for all (per-neuron alphas: not built).

## `.weightpatch.json` (version 1.0)
`wsplang_patch_version, created, base_checkpoint_hash` (sha256 of the safetensors shard holding the tensor),
`base_tensor_hash` (sha256 of the tensor bytes before the edit), `base_dir, tensor_name, op, params {column, mode,
alpha, direction_hash, normalize_direction}, delta_compact {type: column_vector | neuron_set, column, path, key,
cols_key, shape}, delta_full_hash` (sha256 of the npz), `experiment_id, parent_patches, notes`.
`apply_patch` verifies `base_tensor_hash` on the target and `delta_full_hash` on the npz, refuses `out == target`,
writes a new dir (every other tensor and file copied), pushes {patch, src, dst, tensor, result_tensor_hash} onto
`<session>/stack.json`. `rollback_last` pops and returns the previous state dir; nothing is deleted. `reconstruct(base,
patches, out_root)` replays. `commit_manifest(name)` writes `<session>/<name>.bundle.json` with base hashes, the
ordered patches (inlined) and the final tensor hashes. Tested: A, B, C applied; rollback B and C → state equals
"after A" by hash; the stack re-opened from disk after a "restart"; base + patches reconstructs byte for byte; a
patch made against a later state is refused on the base; a modified npz is refused; writing over the source is refused.

## Dry run on the real model (CPU, 2026-11-18; `dryrun_exp8.py`, outputs `lab/runs/column_edit/exp8/`)
Exp 8 rebuilt through the op from the 64 recovered columns per layer (mode `add`, per-column vectors):
layer 15 `params_touched` 65,260, ‖dW‖ 1.7489, ‖dW‖/‖W‖ 0.1064, cos 1.0; layer 19 64,928, 1.3988, 0.0811, 1.0;
placebo: same counts, same norms (seed 2026). The program applied by `wsp.py edit` gave
`servidos/qwen35-08b__exp8-via-edit-column.gguf` whose `blk.15` and `blk.19 ffn_down` are **byte-equal** to
`servidos/qwen35-08b__exp8-replica.gguf`, no other tensor differs, and the whole file has the same sha256
(`byte_check.json`). The HF-side write (new 1.6 GB safetensors dir) is exercised by the unit test on a tiny model,
not repeated on the 0.8B. The placebo program is written, not applied (no behaviour run here).

## Not done / next
No GPU, no benchmark, no commit (coordinator). Open: per-neuron alphas; a `direction=contrast:<pos.jsonl>,<neg.jsonl>`
form that calls `contrast_vector` (the data path exists, the glue does not); `delta_compact.type = rank1` and
`ising_moves` (reserved names, no writer); GGUF-side `apply_patch` (today the GGUF path goes through the ediciones
program and `wsp.py`, which has its own manifest).
