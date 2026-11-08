"""engine/delta_pack.py - a selective low-rank delta as a reproducible package: manifest, hashes, apply, rollback.

What it is: the part of (model - base) that a rank-k projection keeps, for chosen tensors, stored as factors
(left @ right) plus a manifest. The factors are derived from both models' weights, so they are NOT committed to the
repository; the manifest and this code are. What it claims: the extraction is reproducible and checkable (hashes of every
source tensor and every output). What it does NOT claim: any benefit. The manifest says "behavioural_benefit: not shown"
until a pre-registered test (lab/probes/delta_patterns/behavioural_plan.md) says otherwise.

  build_pack(base, model, names, rank, ...)  -> (factors, manifest)
  apply_pack(base, factors, manifest, alpha) -> new tensors (inputs are never modified); refuses on a hash mismatch
  verify_tensors(tensors, manifest, "base")  -> names whose bytes do NOT match the manifest hash (empty = all match): corruption detection
  rollback_check(base, manifest)             -> True when the base tensors still match the manifest hash
  unapply(patched, factors, manifest, base)  -> subtracts the same factors and reports the max abs error. This is an
                                                approximation: floating-point subtraction does not reproduce the original
                                                bytes, so it is a diagnostic, never the rollback.
Rollback is by source, not by arithmetic: apply never modifies its input, so the original stays; to go back, discard the patched
copy, or re-read the original tensor from its source file and confirm its hash with verify_tensors. Nothing here promises a
bit-exact undo by subtraction.
The factor method is the one of lab/probes/delta_patterns/layer_probe.py ("rank" mode): project the delta on the top-k
singular vectors of its smaller side (Gram matrix eigenvectors).
"""
import hashlib
import json
from typing import Dict, List, Optional, Tuple

import torch

SCHEMA = "wsp-delta-pack/1"


def tensor_sha256(t: torch.Tensor) -> str:
    """Hash of dtype, shape and raw bytes (so a cast or a reshape changes it)."""
    t = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.element_size() else b"")
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def low_rank_factors(delta: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """(left, right, energy_kept) with left @ right = delta projected on its top-`rank` singular directions."""
    d = delta.float()
    if d.ndim != 2:
        raise ValueError("only 2-D tensors have a low-rank factorisation here")
    rows, cols = d.shape
    k = min(rank, rows, cols)
    if cols <= rows:
        g = d.T @ d
        w, v = torch.linalg.eigh(g)
        idx = torch.argsort(w, descending=True)[:k]
        v = v[:, idx]
        left, right = d @ v, v.T.contiguous()
    else:
        g = d @ d.T
        w, v = torch.linalg.eigh(g)
        idx = torch.argsort(w, descending=True)[:k]
        v = v[:, idx]
        left, right = v.contiguous(), v.T @ d
    total = float(d.pow(2).sum())
    kept = float((left @ right).pow(2).sum()) / total if total > 0 else 0.0
    return left.contiguous(), right.contiguous(), kept


def build_pack(base: Dict[str, torch.Tensor], model: Dict[str, torch.Tensor], names: List[str], rank: int,
               meta: Optional[dict] = None) -> Tuple[Dict[str, torch.Tensor], dict]:
    """`meta` carries what only the caller knows: repo ids, revisions, licences. Missing keys are written as null."""
    meta = meta or {}
    factors, entries = {}, []
    for n in names:
        a, b = base[n], model[n]
        if a.shape != b.shape:
            raise ValueError(f"{n}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        low = a.dtype if a.dtype == b.dtype else torch.bfloat16 if torch.bfloat16 in (a.dtype, b.dtype) else a.dtype
        delta = b.to(low).float() - a.to(low).float()  # a storage cast (F32 vs BF16) is not a change
        left, right, kept = low_rank_factors(delta, rank)
        factors[n + "::left"], factors[n + "::right"] = left, right
        entries.append({"name": n, "shape": list(a.shape), "dtype": str(a.dtype), "rank": int(left.shape[1]),
                        "base_sha256": tensor_sha256(a), "model_sha256": tensor_sha256(b),
                        "delta_energy_kept": kept,
                        "left_sha256": tensor_sha256(left), "right_sha256": tensor_sha256(right)})
    manifest = {"schema": SCHEMA, "method": "rank-k projection of (model - base) on the singular vectors of its smaller side",
                "rank_requested": rank, "torch": torch.__version__,
                "base": {"repo": meta.get("base_repo"), "revision": meta.get("base_revision"), "license": meta.get("base_license")},
                "model": {"repo": meta.get("model_repo"), "revision": meta.get("model_revision"), "license": meta.get("model_license")},
                "default_alpha": 1.0, "tensors": entries,
                "claims": {"extraction_reproducible": True, "behavioural_benefit": "not shown",
                           "note": "factors derive from both models; redistribution follows both licences and is not decided here"}}
    return factors, manifest


def _entry_map(manifest: dict) -> Dict[str, dict]:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unknown manifest schema {manifest.get('schema')!r}")
    return {e["name"]: e for e in manifest["tensors"]}


def verify_tensors(tensors: Dict[str, torch.Tensor], manifest: dict, which: str = "base") -> List[str]:
    """Names listed in the manifest that are missing or whose dtype/shape/bytes differ from the recorded hash (which = base | model)."""
    key = {"base": "base_sha256", "model": "model_sha256"}[which]
    return [n for n, e in _entry_map(manifest).items() if n not in tensors or tensor_sha256(tensors[n]) != e[key]]


def rollback_check(base: Dict[str, torch.Tensor], manifest: dict) -> bool:
    return not verify_tensors(base, manifest, "base")


def apply_pack(base: Dict[str, torch.Tensor], factors: Dict[str, torch.Tensor], manifest: dict,
               alpha: Optional[float] = None) -> Dict[str, torch.Tensor]:
    alpha = manifest["default_alpha"] if alpha is None else alpha
    out = {}
    for n, e in _entry_map(manifest).items():
        if n not in base:
            raise KeyError(f"{n}: not in the base tensors")
        if tensor_sha256(base[n]) != e["base_sha256"]:
            raise ValueError(f"{n}: base tensor does not match the manifest hash; refusing to apply")
        for side in ("left", "right"):
            if tensor_sha256(factors[f"{n}::{side}"]) != e[f"{side}_sha256"]:
                raise ValueError(f"{n}: {side} factor does not match the manifest hash; refusing to apply")
        out[n] = base[n].float() + alpha * (factors[n + "::left"] @ factors[n + "::right"])
    return out  # float32; the inputs are untouched, so rollback = discard this


def unapply(patched: Dict[str, torch.Tensor], factors: Dict[str, torch.Tensor], manifest: dict,
            base: Dict[str, torch.Tensor], alpha: Optional[float] = None) -> Tuple[Dict[str, torch.Tensor], float]:
    alpha = manifest["default_alpha"] if alpha is None else alpha
    back, err = {}, 0.0
    for n in _entry_map(manifest):
        back[n] = patched[n] - alpha * (factors[n + "::left"] @ factors[n + "::right"])
        err = max(err, float((back[n] - base[n].float()).abs().max()))
    return back, err


def save_pack(factors, manifest, directory: str, stem: str = "delta_pack") -> dict:
    """Writes `<stem>.safetensors` (factors; do not commit) and `<stem>.manifest.json`; records the factor file hash."""
    from safetensors.torch import save_file
    import os
    os.makedirs(directory, exist_ok=True)
    fpath = os.path.join(directory, stem + ".safetensors")
    save_file(factors, fpath)
    manifest = dict(manifest, factor_file={"name": stem + ".safetensors", "sha256": file_sha256(fpath)})
    with open(os.path.join(directory, stem + ".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest
