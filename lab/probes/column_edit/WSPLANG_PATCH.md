# Registration patch for engine/wsplang.py (to be applied by the coordinator; not applied by the agent)

Three hunks: the op lists and one dispatch branch per op, modelled on `ternary_survival` and `delta_to_lora`.
All code lives in `engine/column_edit.py` and `engine/wsplpatch.py`; `wsplang.py` only parses arguments, applies the
path guards and calls them. Nothing writes into an input model: `edit_column` and `apply_patch` write new dirs,
`rollback_last` only pops the on-disk stack (`<session>/stack.json`) and reports the previous state dir.

## 1. OPS_ALLOWED

```diff
     "ternary_survival",  # how much of an edit survives ternarisation (TQ2_0 / absmean); report only (engine/ternary.py)
+    "edit_column", "edit_neurons",  # one matrix, few columns, structural same-norm placebo, report + patches (engine/column_edit.py)
+    "apply_patch", "rollback_last", "list_patches", "commit_manifest",  # .weightpatch.json runtime (engine/wsplpatch.py)
     # Architectural
```

## 2. Write policy

`edit_column` / `edit_neurons` / `apply_patch` write **new model dirs** (never over an input): add them to the set that
`IMPLEMENTED_OPS` builds from (the explicit write set, next to `leace_erase`, `cov_imprint`, ...), not to
`READ_ONLY_OPS`. `rollback_last`, `list_patches`, `commit_manifest` write only JSON under the session dir: add them to
`READ_ONLY_OPS`.

```diff
     "ternary_survival",  # writes one JSON report, never weights
+    "rollback_last", "list_patches", "commit_manifest",  # patch stack bookkeeping: JSON under the session dir only
 }
 IMPLEMENTED_OPS = READ_ONLY_OPS | {
+    "edit_column", "edit_neurons", "apply_patch",  # new model dirs beside the inputs, never over them
     "leace_erase", "cov_imprint", "memit_imprint", "decoupled_delta", "surprise_fwp",
```

## 3. Dispatch (insert before `elif op == "model_report":`)

```python
            elif op in ("edit_column", "edit_neurons"):
                # One matrix, one or a few columns (neurons), a dose and a direction, always with the structural
                # same-norm placebo beside it (lab/probes/column_edit/PLAN.md). Writes out= (+ out_placebo) as new
                # safetensors dirs, a report, a manifest, two .weightpatch.json and the ediciones program + npz for
                # wsp.py add-cols-vectors on the GGUF. No behaviour claim: evaluate=true needs placebo=true.
                from engine import column_edit as ce
                for key in ("model", "tensor", "out"):
                    if not named.get(key):
                        raise ValueError(f"{op} needs model=<hf dir> tensor=<name> column=<c>|neurons=<c1,c2,...> "
                                         "direction=<npy|npz> alpha= mode=add|subtract|scale|replace|subtract_direction out=<dir>")
                flags = ce.validate_request(named)
                model_p = _model_path(named["model"])
                out_p = check_report_path(named["out"], [model_p])
                colspec = named.get("neurons", named.get("column"))
                if colspec is None:
                    raise ValueError(f"{op} needs column=<c> or neurons=<c1,c2,...>")
                cols = [int(x) for x in str(colspec).strip("[]").split(",") if x.strip()]
                mode = str(named.get("mode", "add"))
                direction = _model_path(named["direction"]) if named.get("direction") else None
                if mode != "scale" and direction is None:
                    raise ValueError(f"{op}: mode {mode} needs direction=<npy|npz>")
                budget = int(named["budget_params"]) if named.get("budget_params") else None
                res["result"] = ce.run_edit_column(
                    model=model_p, tensor=str(named["tensor"]), cols=cols, direction=direction,
                    alpha=float(named.get("alpha", 1.0)), mode=mode, out=out_p, placebo=flags["placebo"],
                    placebo_seed=int(named.get("placebo_seed", 42)), budget_params=budget,
                    normalize_direction=str(named.get("normalize_direction", "true")).lower() in ("1", "true", "yes"),
                    experiment_id=named.get("experiment_id"), op=op)
                res["result"]["use"] = ("python wsp.py edit <base.gguf> <out>/edit.json (and edit_placebo.json) for the "
                                        "Q6_K arms; benchmark both on dev, then a sealed set")
                res["execution_state"] = "executed"

            elif op == "apply_patch":
                # apply_patch patch=<.weightpatch.json> target=<model dir> out=<new dir> [session=<dir>]
                from engine import wsplpatch as wp
                for key in ("patch", "target", "out"):
                    if not named.get(key):
                        raise ValueError("apply_patch needs patch=<file> target=<model dir> out=<new dir> [session=<dir>]")
                target_p = _model_path(named["target"])
                out_p = check_report_path(named["out"], [target_p])
                pm = wp.PatchManager(_model_path(named.get("session", "lab/runs/session_patches")))
                dst = pm.apply(_model_path(named["patch"]), target_p, out_p)
                res["result"] = {"out": str(dst), "stack_depth": len(pm.list_patches())}
                res["execution_state"] = "executed"

            elif op in ("rollback_last", "list_patches", "commit_manifest"):
                # rollback_last [session=] | list_patches [session=] | commit_manifest name=<..> [include_patches=true]
                from engine import wsplpatch as wp
                pm = wp.PatchManager(_model_path(named.get("session", "lab/runs/session_patches")))
                if op == "rollback_last":
                    prev = pm.rollback_last()
                    res["result"] = {"current": str(prev), "stack_depth": len(pm.list_patches()),
                                     "note": "nothing deleted; the popped dir stays on disk"}
                elif op == "list_patches":
                    res["result"] = {"applied": pm.list_patches()}
                else:
                    if not named.get("name"):
                        raise ValueError("commit_manifest needs name=<bundle name>")
                    inc = str(named.get("include_patches", "true")).lower() in ("1", "true", "yes")
                    res["result"] = {"bundle": str(pm.commit_manifest(str(named["name"]), include_patches=inc))}
                res["execution_state"] = "executed"
```

## 4. Validation table

If `wsplang.py` keeps a per-op argument table for `--check` (the dict near line 497 with one-line descriptions),
add:

```python
        "edit_column": "one column of one matrix: add|subtract|scale|replace|subtract_direction, structural placebo, report, patches",
        "edit_neurons": "edit_column over a short list of neurons (neurons=c1,c2,...; alphas not yet per neuron)",
        "apply_patch": "apply a .weightpatch.json to a model into a new dir after verifying base and delta hashes",
        "rollback_last": "pop the last applied patch from the session stack; returns the previous state dir",
        "list_patches": "list the session's applied patches",
        "commit_manifest": "write a bundle: base hash, ordered patches, final tensor hashes",
```

## 5. Tests to run after applying

`PYTHONUTF8=1 python -m unittest tests.test_column_edit tests.test_wsplpatch tests.test_cc_compile tests.test_mcp_server`
(the last two check the op lists). `tests/test_column_edit.py::WsplangOpTest` is not included on purpose (the agent
could not edit `wsplang.py`); after applying, a two-line test `assertIn("edit_column", wsplang.OPS_ALLOWED)` /
`assertIn("apply_patch", wsplang.IMPLEMENTED_OPS)` belongs in `tests/test_column_edit.py`.

## Deviation from the driver's sketch (kept on purpose)

The sketch's WriteGuard edits `model_path` in place and rolls back from an in-memory copy; here every write goes to a
new dir and the stack lives on disk, so rollback works after a process restart and never touches the input. The
sketch computes `params_touched` from the direction; here it is counted on the realised delta (so `scale` and
`replace` report the true count) and the placebo matches that count by construction (same support), falling back to a
sign flip when a column's support is too small for an orthogonal direction. `delta_full_hash` is verified on apply.
