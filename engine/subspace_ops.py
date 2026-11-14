"""engine/subspace_ops.py - move a model along a subspace of another model's update.

subspace_transfer (WSPLang op): W' = W + alpha * P(D), D = W_source - W_base, where P projects D onto a k-dim
subspace of its smaller side. For the matrices of Qwen3.5-style models the smaller side is the 4096-dim residual
stream, so P keeps the part of the update that reads from (or writes to) the chosen residual directions.

The subspace comes from:
  sig:<path>          the top-k basis stored in a delta signature (engine.weight_diff.delta_signature);
  shared:<a>,<b>      the k directions two signatures have most in common (principal vectors);
  random              a random k-dim subspace (the control: same k, same source delta).
Provenance written into every shard: synthetic_transfer, with the source and basis.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def small_side(shape) -> int:
    """0 if rows are the smaller side (basis lives on rows), 1 if columns are (ties go to columns)."""
    rows, cols = shape
    return 1 if cols <= rows else 0


def project_delta(D: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Projection of D onto span(V) on D's smaller side. V: (d, k) with orthonormal columns, d = smaller dim."""
    side = small_side(D.shape)
    d = D.shape[side]
    if V.shape[0] != d:
        raise ValueError(f"basis has {V.shape[0]} rows; the smaller side of {tuple(D.shape)} has {d}")
    V = V.to(D.dtype)
    return (D @ V) @ V.T if side == 1 else V @ (V.T @ D)


def random_basis(d: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0].float()


def shared_basis(A: torch.Tensor, B: torch.Tensor, k: int) -> Tuple[torch.Tensor, List[float]]:
    """The k directions of span(A) closest to span(B) (principal vectors), with the cosines of their angles."""
    A, B = A.double(), B.double()
    U, S, _ = torch.linalg.svd(A.T @ B)
    return (A @ U[:, :k]).float(), [float(x) for x in S[:k]]


def basis_for(spec: str, name: str, shape, k: int, seed: int, cache: Dict) -> torch.Tensor:
    """Resolve a basis spec for one tensor. cache holds loaded signature files."""
    d = shape[small_side(shape)]
    if spec == "random":
        import zlib
        return random_basis(d, k, seed * 1_000_003 + zlib.crc32(name.encode()))

    def sig(path):
        if path not in cache:
            cache[path] = torch.load(path)
        if name not in cache[path] or "basis" not in cache[path][name]:
            raise KeyError(f"signature {path} has no basis for {name}")
        return cache[path][name]["basis"]

    if spec.startswith("sig:"):
        V = sig(spec[4:])
        if V.shape[1] < k:
            raise ValueError(f"signature basis for {name} has {V.shape[1]} vectors, k={k} asked")
        return V[:, :k].float()
    if spec.startswith("shared:"):
        a, b = spec[7:].split(",", 1)
        return shared_basis(sig(a), sig(b), k)[0]
    raise ValueError(f"unknown basis spec {spec!r}: use sig:<path>, shared:<a>,<b> or random")
