"""Weight patches for WSP-Lang (lab/probes/column_edit/PLAN.md): a column / neuron edit as a text file
(`.weightpatch.json`, version 1.0) bound to the sha256 of the tensor it was made against and to the sha256 of its
delta, applied into a NEW safetensors dir (never over an input), with a PatchManager that keeps the session stack on
disk (apply pushes, rollback_last pops and returns the previous state dir, survives a process restart), exact
reconstruction from base + patches, a bundle (commit_manifest) and byte-equality checks.

Patch (JSON):
  wsplang_patch_version "1.0", created, base_checkpoint_hash (sha256 of the safetensors file holding the tensor),
  base_tensor_hash (sha256 of the tensor bytes before the edit), tensor_name, op, params {column, mode, alpha,
  direction_hash, normalize_direction}, delta_compact {type: column_vector | neuron_set, column, path (npz with
  cols_0 / vec_0, as ediciones)}, delta_full_hash (sha256 of that npz), experiment_id, parent_patches [], notes.

Rules: apply verifies base_tensor_hash and delta_full_hash and aborts on mismatch; dst must not be an input dir; the
edit is W[:, cols] += vec in float32, rounded back to the tensor's dtype; every other tensor and file is copied.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

PATCH_VERSION = "1.0"


class PatchError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _reader(model_dir):
    from engine import weight_diff as wd
    return wd.LocalSafetensors(Path(model_dir))


def read_tensor(model_dir, name) -> torch.Tensor:
    return _reader(model_dir).get(name)


def tensor_file(model_dir, name) -> Path:
    r = _reader(model_dir)
    if name not in r._where:
        raise PatchError(f"{name} not in {model_dir}")
    return Path(r._where[name][0])


def tensor_hash(model_dir, name) -> str:
    return hashlib.sha256(_bytes(read_tensor(model_dir, name))).hexdigest()


def tensors_equal(dir_a, dir_b, names=None) -> bool:
    ra, rb = _reader(dir_a), _reader(dir_b)
    names = list(names) if names is not None else sorted(set(ra.keys()) | set(rb.keys()))
    for n in names:
        if n not in ra._where or n not in rb._where:
            return False
        a, b = ra.get(n), rb.get(n)
        if a.shape != b.shape or a.dtype != b.dtype or _bytes(a) != _bytes(b):
            return False
    return True


def write_safetensors_dir(src_dir, dst_dir, replacements: dict):
    """Copy src_dir to dst_dir; the shards holding `replacements` (name -> tensor) are rewritten with them."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    src, dst = Path(src_dir), Path(dst_dir)
    if not src.is_dir():
        raise PatchError(f"{src} is not a model dir")
    if dst.exists() and any(dst.glob("*.safetensors")):
        raise PatchError(f"{dst} already holds a model: patches never write over an existing model")
    if dst.resolve() == src.resolve() or src.resolve() in dst.resolve().parents:
        raise PatchError(f"{dst} is inside the input {src}")
    r = _reader(src)
    shards = {}
    for name in replacements:
        if name not in r._where:
            raise PatchError(f"{name} not in {src}")
        shards.setdefault(Path(r._where[name][0]).name, []).append(name)
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_dir():
            continue
        if f.name in shards:
            tensors, meta = {}, None
            with safe_open(str(f), framework="pt") as fh:
                meta = fh.metadata()
                for k in fh.keys():
                    tensors[k] = fh.get_tensor(k)
            for name in shards[f.name]:
                new = replacements[name]
                old = tensors[name]
                if tuple(new.shape) != tuple(old.shape):
                    raise PatchError(f"{name}: shape {tuple(new.shape)} != {tuple(old.shape)}")
                tensors[name] = new.to(old.dtype).contiguous()
            save_file(tensors, str(dst / f.name), metadata=meta or {"format": "pt"})
        else:
            shutil.copy2(f, dst / f.name)
    return dst


def load_delta(npz_path, index=0):
    with np.load(npz_path, allow_pickle=False) as z:
        cols = np.asarray(z[f"cols_{index}"], dtype=int)
        V = np.asarray(z[f"vec_{index}"], dtype=np.float32)
    return cols, V


def make_patch(model_dir, tensor_name, cols, npz_path, alpha, mode, direction_hash=None, normalize_direction=True,
               experiment_id=None, parent_patches=None, notes="", op="edit_column", delta_type=None):
    cols = [int(c) for c in cols]
    npz_path = Path(npz_path)
    cols_npz, V = load_delta(npz_path)
    if cols_npz.tolist() != cols:
        raise PatchError(f"npz columns {cols_npz.tolist()} != patch columns {cols}")
    return {"wsplang_patch_version": PATCH_VERSION,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_checkpoint_hash": sha256_file(tensor_file(model_dir, tensor_name)),
            "base_tensor_hash": tensor_hash(model_dir, tensor_name),
            "base_dir": str(Path(model_dir)),
            "tensor_name": tensor_name,
            "op": op,
            "params": {"column": cols, "mode": mode, "alpha": float(alpha), "direction_hash": direction_hash,
                       "normalize_direction": bool(normalize_direction)},
            "delta_compact": {"type": delta_type or ("column_vector" if len(cols) == 1 else "neuron_set"),
                              "column": cols, "path": str(npz_path.resolve()), "key": "vec_0", "cols_key": "cols_0",
                              "shape": list(V.shape)},
            "delta_full_hash": sha256_file(npz_path),
            "experiment_id": experiment_id,
            "parent_patches": list(parent_patches or []),
            "notes": notes}


def _load_patch(patch):
    if isinstance(patch, dict):
        return patch, None
    p = Path(patch)
    return json.loads(p.read_text(encoding="utf-8")), p


def apply_patch(patch, src_dir, dst_dir, stack_file=None):
    """Verify hashes, write src + delta into dst_dir (new), push onto the stack file if given. -> dst_dir."""
    p, ppath = _load_patch(patch)
    if p.get("wsplang_patch_version") != PATCH_VERSION:
        raise PatchError(f"unsupported patch version {p.get('wsplang_patch_version')!r}")
    src, dst = Path(src_dir), Path(dst_dir)
    if dst.resolve() == src.resolve():
        raise PatchError(f"refusing to write the patch over its source {src}")
    name = p["tensor_name"]
    have = tensor_hash(src, name)
    if have != p["base_tensor_hash"]:
        raise PatchError(f"base_tensor_hash mismatch for {name}: patch {p['base_tensor_hash'][:16]}, model {have[:16]}")
    npz = Path(p["delta_compact"]["path"])
    if not npz.exists():
        raise PatchError(f"delta file missing: {npz}")
    if sha256_file(npz) != p["delta_full_hash"]:
        raise PatchError(f"delta_full_hash mismatch: {npz} was modified after the patch was made")
    cols, V = load_delta(npz)
    W0 = read_tensor(src, name)
    W = W0.float().clone()
    if V.shape != (W.shape[0], cols.size):
        raise PatchError(f"delta shape {V.shape} != ({W.shape[0]}, {cols.size})")
    W[:, torch.as_tensor(cols)] += torch.from_numpy(V)
    write_safetensors_dir(src, dst, {name: W.to(W0.dtype)})
    if stack_file is not None:
        _push(Path(stack_file), {"patch": str(ppath) if ppath else None, "src": str(src), "dst": str(dst),
                                 "tensor": name, "result_tensor_hash": tensor_hash(dst, name),
                                 "applied": time.strftime("%Y-%m-%d %H:%M:%S")})
    return dst


def _read_stack(stack_file: Path):
    if stack_file.exists():
        return json.loads(stack_file.read_text(encoding="utf-8"))
    return {"applied": []}


def _write_stack(stack_file: Path, s):
    stack_file.parent.mkdir(parents=True, exist_ok=True)
    stack_file.write_text(json.dumps(s, indent=1), encoding="utf-8")


def _push(stack_file: Path, entry):
    s = _read_stack(stack_file)
    s["applied"].append(entry)
    _write_stack(stack_file, s)


def rollback_last(stack_file):
    """Pop the last applied patch; return the previous state dir (its src). Nothing on disk is deleted."""
    stack_file = Path(stack_file)
    s = _read_stack(stack_file)
    if not s["applied"]:
        raise PatchError("rollback_last: no patch applied in this session")
    last = s["applied"].pop()
    _write_stack(stack_file, s)
    return Path(last["src"])


def reconstruct(base_dir, patches, out_root):
    """Apply patches in order from base_dir into out_root/step_<i>; return the final dir."""
    out_root = Path(out_root)
    state = Path(base_dir)
    for i, p in enumerate(patches):
        state = apply_patch(p, state, out_root / f"step_{i}")
    return state


class PatchManager:
    """Session stack on disk: <session_dir>/stack.json; apply / rollback_last / list_patches / current / commit_manifest."""

    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        self.stack_file = self.session_dir / "stack.json"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def apply(self, patch, src_dir, dst_dir):
        return apply_patch(patch, src_dir, dst_dir, stack_file=self.stack_file)

    def rollback_last(self):
        return rollback_last(self.stack_file)

    def list_patches(self):
        return list(_read_stack(self.stack_file)["applied"])

    def current(self):
        s = self.list_patches()
        return Path(s[-1]["dst"]) if s else None

    def commit_manifest(self, name, include_patches=True):
        applied = self.list_patches()
        if not applied:
            raise PatchError("commit_manifest: nothing applied")
        base = Path(applied[0]["src"])
        final = Path(applied[-1]["dst"])
        tensors = sorted({a["tensor"] for a in applied})
        patches = []
        for a in applied:
            if include_patches and a.get("patch"):
                patches.append(json.loads(Path(a["patch"]).read_text(encoding="utf-8")))
            else:
                patches.append({"patch": a.get("patch"), "result_tensor_hash": a["result_tensor_hash"]})
        bundle = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "base_dir": str(base),
                  "base_checkpoint_hash": sha256_file(tensor_file(base, tensors[0])),
                  "base_tensor_hashes": {t: tensor_hash(base, t) for t in tensors},
                  "patches": patches, "final_dir": str(final),
                  "final_tensor_hashes": {t: tensor_hash(final, t) for t in tensors}}
        out = self.session_dir / f"{name}.bundle.json"
        out.write_text(json.dumps(bundle, indent=1), encoding="utf-8")
        return out
