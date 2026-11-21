"""WSP-Lang op edit_column / edit_neurons (lab/probes/column_edit/PLAN.md): one matrix, one or a few columns
(neurons; HF orientation W (out, in), columns index the in dimension), a dose, a direction, and always the structural
same-norm placebo beside it. Modes:
  add       W[:, c] += alpha * dir          subtract  W[:, c] -= alpha * dir
  scale     W[:, c] *= alpha                replace   W[:, c]  = alpha * dir
  subtract_direction  W[:, c] -= alpha * u * (u . W[:, c])   (the MIT Exp 8 closed form, u = dir / |dir|)
A 1-D direction is normalised unless normalize_direction=False; a 2-D direction (one vector per column, as
data/exp8_edit_recovered.npz) is used as given. The placebo keeps the same columns and the same per-column norms of
the REALISED delta, with a random direction orthogonal to the real one (seeded). Nothing is written over an input:
run_edit_column writes a new safetensors dir (+ <out>_placebo), a report, a manifest, two .weightpatch.json and the
ediciones program + npz so wsp.py's add-cols-vectors applies the same edit to the Q6_K GGUF. No behaviour is claimed
here: that needs dev + placebo + sealed set under its own plan.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

MODES = ("add", "subtract", "scale", "replace", "subtract_direction")
_GGUF = {"mlp.down_proj": "ffn_down", "mlp.up_proj": "ffn_up", "mlp.gate_proj": "ffn_gate",
         "self_attn.o_proj": "attn_output", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
         "self_attn.v_proj": "attn_v"}


class BudgetError(Exception):
    pass


def gguf_name(hf_name: str) -> str:
    parts = hf_name.split(".")
    if "layers" not in parts or not hf_name.endswith(".weight"):
        raise ValueError(f"no GGUF name for {hf_name}")
    L = parts[parts.index("layers") + 1]
    tail = ".".join(parts[parts.index("layers") + 2:-1])
    if tail not in _GGUF:
        raise ValueError(f"no GGUF name for {hf_name}")
    return f"blk.{L}.{_GGUF[tail]}.weight"


def _cols(cols, ncols):
    cols = [int(c) for c in (cols if hasattr(cols, "__iter__") else [cols])]
    if not cols or len(set(cols)) != len(cols) or min(cols) < 0 or max(cols) >= ncols:
        raise ValueError(f"columns {cols} invalid for {ncols} columns")
    return cols


def _direction(direction, nrows, ncols_sel, normalize):
    if direction is None:
        return None
    D = np.asarray(direction, dtype=np.float32)
    if D.ndim == 1:
        if D.shape[0] != nrows:
            raise ValueError(f"direction length {D.shape[0]} != rows {nrows}")
        n = float(np.linalg.norm(D))
        if n == 0:
            raise ValueError("direction is zero")
        D = D / n if normalize else D
        return np.repeat(D[:, None], ncols_sel, axis=1)
    if D.ndim == 2:
        if D.shape != (nrows, ncols_sel):
            raise ValueError(f"direction shape {D.shape} != ({nrows}, {ncols_sel})")
        return D
    raise ValueError("direction must be 1-D or 2-D")


def plan_edit(W0, cols, direction, alpha, mode, normalize_direction=True):
    """-> dense dW (same shape as W0), non-zero only in `cols`."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    W0 = np.asarray(W0, dtype=np.float32)
    cols = _cols(cols, W0.shape[1])
    alpha = float(alpha)
    dW = np.zeros_like(W0)
    if mode == "scale":
        dW[:, cols] = (alpha - 1.0) * W0[:, cols]
        return dW
    D = _direction(direction, W0.shape[0], len(cols), normalize_direction)
    if D is None:
        raise ValueError(f"mode {mode} needs a direction")
    if mode == "add":
        dW[:, cols] = alpha * D
    elif mode == "subtract":
        dW[:, cols] = -alpha * D
    elif mode == "replace":
        dW[:, cols] = alpha * D - W0[:, cols]
    else:  # subtract_direction: per column, u = unit direction of that column
        U = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-30)
        proj = (U * W0[:, cols]).sum(axis=0, keepdims=True)
        dW[:, cols] = -alpha * U * proj
    return dW


def structural_placebo(dW, cols, direction, seed):
    """Same columns, same per-column norms of the realised delta, random direction orthogonal to the real one."""
    dW = np.asarray(dW, dtype=np.float32)
    cols = _cols(cols, dW.shape[1])
    rng = np.random.default_rng(int(seed))
    P = np.zeros_like(dW)
    if direction is not None:
        D = np.asarray(direction, dtype=np.float32)
        ref = np.repeat(D[:, None], len(cols), axis=1) if D.ndim == 1 else D
    else:
        ref = dW[:, cols]
    for j, c in enumerate(cols):
        col = dW[:, c].astype(np.float64)
        target = float(np.linalg.norm(col))
        if target == 0:
            continue
        support = col != 0                     # same parameters touched as the realised delta
        basis = []
        for v in (ref[:, j].astype(np.float64) * support, col):
            for b in basis:
                v = v - float(v @ b) * b
            if np.linalg.norm(v) > 1e-12:
                basis.append(v / np.linalg.norm(v))
        r = rng.normal(size=dW.shape[0]) * support
        for _ in range(2):  # Gram-Schmidt against the real direction and the realised column, within the support
            for b in basis:
                r -= float(r @ b) * b
        if np.linalg.norm(r) < 1e-8 * target or int(support.sum()) <= len(basis):
            r = -col                           # support too small for an orthogonal direction: sign flip, same norm
        P[:, c] = (r / np.linalg.norm(r) * target).astype(np.float32)
    return P


def check_budget(dW, budget_params):
    touched = int(np.count_nonzero(dW))
    if budget_params is not None and touched > int(budget_params):
        raise BudgetError(f"params_touched {touched} > budget_params {int(budget_params)}: nothing written")
    return touched


def validate_request(named):
    """A behavioural evaluation request (evaluate=true) without a placebo arm is refused."""
    truthy = {"1", "true", "yes", "on"}
    placebo = str(named.get("placebo", "true")).lower() in truthy
    evaluate = str(named.get("evaluate", "false")).lower() in truthy
    if evaluate and not placebo:
        raise ValueError("edit_column: evaluate=true needs placebo=true (no behaviour claim without the same-norm arm)")
    return {"placebo": placebo, "evaluate": evaluate}


def report(W0, dW, direction=None, cols=None, alpha=None, mode=None, tensor=None, placebo_dW=None, placebo_seed=None,
           budget_params=None, op="edit_column"):
    W0 = np.asarray(W0, dtype=np.float32)
    dW = np.asarray(dW, dtype=np.float32)
    cols = _cols(cols, W0.shape[1]) if cols is not None else sorted(set(np.nonzero(dW)[1].tolist()))
    touched = int(np.count_nonzero(dW))
    n_dw = float(np.linalg.norm(dW))
    n_w0 = float(np.linalg.norm(W0))
    cosine = None
    if direction is not None and n_dw > 0:
        D = np.asarray(direction, dtype=np.float32)
        ref = np.repeat(D[:, None], len(cols), axis=1) if D.ndim == 1 else D
        a, b = dW[:, cols].ravel().astype(np.float64), ref.ravel().astype(np.float64)
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    r = {"op": op, "tensor": tensor, "column": cols, "params_touched": touched, "delta_norm": n_dw,
         "rel_delta_norm": n_dw / n_w0 if n_w0 else None, "cosine_with_direction": cosine, "alpha": alpha,
         "mode": mode, "shape": list(W0.shape),
         "col_norms": {str(c): float(np.linalg.norm(dW[:, c])) for c in cols},
         "placebo_seed": placebo_seed, "placebo_delta_norm": None, "placebo_params_touched": None,
         "budget_params": budget_params,
         "budget_params_ok": True if budget_params is None else touched <= int(budget_params)}
    if placebo_dW is not None:
        P = np.asarray(placebo_dW, dtype=np.float32)
        r["placebo_delta_norm"] = float(np.linalg.norm(P))
        r["placebo_params_touched"] = int(np.count_nonzero(P))
    return r


def _npz(path, edits):
    arrays = {}
    for i, e in enumerate(edits):
        cols = np.asarray(e["cols"], dtype=np.int64)
        arrays[f"cols_{i}"] = cols
        arrays[f"vec_{i}"] = np.asarray(e["dW"], dtype=np.float32)[:, cols]
    np.savez(path, **arrays)
    return path


def _program(path, npz_path, edits, gguf_out):
    prog = {"out": str(gguf_out), "ops": [{"type": "add-cols-vectors", "tensor": e["gguf_tensor"],
                                          "vec_file": str(Path(npz_path).resolve()), "key": f"vec_{i}",
                                          "cols_key": f"cols_{i}", "scale": 1.0} for i, e in enumerate(edits)]}
    Path(path).write_text(json.dumps(prog, indent=2), encoding="utf-8")
    return path


def write_program(out_dir, name, edits, placebos=None, gguf_out=None, gguf_out_placebo=None):
    """edits / placebos: [{gguf_tensor, cols, dW}] in the same order -> npz + ediciones JSON per arm."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"npz": str(_npz(out_dir / f"{name}.npz", edits)),
           "program": str(_program(out_dir / f"{name}.json", out_dir / f"{name}.npz", edits,
                                   gguf_out or out_dir / f"{name}.gguf"))}
    if placebos:
        res["npz_placebo"] = str(_npz(out_dir / f"{name}_placebo.npz", placebos))
        res["program_placebo"] = str(_program(out_dir / f"{name}_placebo.json", out_dir / f"{name}_placebo.npz",
                                              placebos, gguf_out_placebo or out_dir / f"{name}_placebo.gguf"))
    return res


def load_direction(direction):
    """npz (key 'direction' or 'u' or the first array), npy, or an array."""
    if isinstance(direction, (str, Path)):
        p = Path(direction)
        if p.suffix == ".npy":
            return np.load(p)
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                for k in ("direction", "u", "vec_0"):
                    if k in z.files:
                        return np.asarray(z[k])
                return np.asarray(z[z.files[0]])
        raise ValueError(f"direction file type not supported: {p}")
    return np.asarray(direction)


def _sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def run_edit_column(model, tensor, cols, direction, alpha, mode, out, placebo=True, placebo_seed=42,
                    budget_params=None, normalize_direction=True, experiment_id=None, gguf_out=None,
                    gguf_out_placebo=None, op="edit_column"):
    """The op on an HF safetensors dir. Writes <out> (+ <out>_placebo), report, manifest, patches, program. Guards
    run before anything is written: budget, out dirs new, never inside the model."""
    from engine import wsplpatch as wp
    model, out = Path(model), Path(out)
    out_p = out.parent / (out.name + "_placebo")
    for d in (out, out_p) if placebo else (out,):
        if d.exists() and any(d.iterdir()):
            raise ValueError(f"{d} exists and is not empty")
        if d.resolve() == model.resolve() or model.resolve() in d.resolve().parents:
            raise ValueError(f"{d} is inside the input model {model}")
    W0t = wp.read_tensor(model, tensor)
    W0 = W0t.float().numpy()
    D = None if mode == "scale" else load_direction(direction)
    dW = plan_edit(W0, cols, D, alpha, mode, normalize_direction)
    touched = check_budget(dW, budget_params)
    cols = _cols(cols, W0.shape[1])
    P = structural_placebo(dW, cols, D, placebo_seed) if placebo else None
    if P is not None and int(np.count_nonzero(P)) != touched:
        raise ValueError("placebo and edit touch a different number of parameters")
    rep = report(W0, dW, D, cols, alpha, mode, tensor, P, placebo_seed if placebo else None, budget_params, op=op)
    g = gguf_name(tensor)
    out.mkdir(parents=True, exist_ok=True)
    prog = write_program(out, "edit", [{"gguf_tensor": g, "cols": cols, "dW": dW}],
                         placebos=[{"gguf_tensor": g, "cols": cols, "dW": P}] if P is not None else None,
                         gguf_out=gguf_out or out / "edit.gguf", gguf_out_placebo=gguf_out_placebo or out / "edit_placebo.gguf")
    dir_hash = _sha_array(D) if D is not None else None
    patch = wp.make_patch(model, tensor, cols, prog["npz"], alpha, mode, direction_hash=dir_hash,
                          normalize_direction=normalize_direction, experiment_id=experiment_id, op=op,
                          notes="edit arm")
    patch_path = out / "edit.weightpatch.json"
    patch_path.write_text(json.dumps(patch, indent=1), encoding="utf-8")
    import torch
    W1 = torch.from_numpy(W0 + dW).to(W0t.dtype)
    wp.write_safetensors_dir(model, out, {tensor: W1})
    res = {"out": str(out), "report": rep, "patch": str(patch_path), **prog}
    hashes = {"base": patch["base_tensor_hash"], "edit": wp.tensor_hash(out, tensor)}
    if P is not None:
        patch_p = wp.make_patch(model, tensor, cols, prog["npz_placebo"], alpha, mode, direction_hash=dir_hash,
                                normalize_direction=normalize_direction, experiment_id=experiment_id, op=op,
                                notes=f"structural placebo, seed {placebo_seed}")
        pp = out / "edit_placebo.weightpatch.json"
        pp.write_text(json.dumps(patch_p, indent=1), encoding="utf-8")
        wp.write_safetensors_dir(model, out_p, {tensor: torch.from_numpy(W0 + P).to(W0t.dtype)})
        res["out_placebo"] = str(out_p)
        res["patch_placebo"] = str(pp)
        hashes["placebo"] = wp.tensor_hash(out_p, tensor)
    manifest = {"op": op, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "model": str(model), "tensor": tensor,
                "gguf_tensor": g, "report": rep, "tensor_hashes": hashes, "patches": {"edit": str(patch_path),
                "placebo": res.get("patch_placebo")}, "program": prog, "params_touched_identical":
                (rep["placebo_params_touched"] == rep["params_touched"]) if P is not None else None,
                "behaviour_claim": "none: dev + placebo + sealed set under a separate plan"}
    (out / "edit_column.report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out / "edit_column.manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    res["manifest"] = str(out / "edit_column.manifest.json")
    return res
