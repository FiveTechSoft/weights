"""engine/weight_diff.py - what changed between two checkpoints, read one tensor at a time.

Readers:
  LocalSafetensors(path)        a .safetensors file or a directory of shards (every shard, read lazily).
  RangeSafetensors(files, fetch) the same format read by byte ranges: fetch(file, start, end_inclusive) -> bytes.
  hf_fetch(repo, revision)      a fetch over HTTP range requests to the Hugging Face hub, nothing written to disk.

tensor_delta_stats(name, a, b) gives, for W' = W + D:
  rel_delta ||D||/||W||, cosine(W, W'), changed_fraction (exact inequality in the stored dtype), max |D|;
  for matrices, the spectrum of D: energy_top1 = s1^2/||D||^2, energy_topk, stable_rank = ||D||^2/s1^2
  (about 1 for a rank-one edit, large for a diffuse update);
  for embedding and output-head matrices, the rows that moved most.
diff_models(base, target) runs it over every shared tensor and adds per-family and per-layer totals.
"""
import fnmatch
import json
import re
import struct
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import torch

DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
          "BOOL": torch.bool}
ROW_TENSORS = ("embed", "wte", "lm_head", "output.weight", "token_embd")


class LocalSafetensors:
    """Every shard of a local checkpoint; a tensor is read only when asked for."""

    def __init__(self, path):
        from safetensors import safe_open
        self._open = safe_open
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors in {path}")
        self._where: Dict[str, Path] = {}
        for f in files:
            with safe_open(str(f), "pt") as h:
                for k in h.keys():
                    self._where[k] = f

    def keys(self) -> List[str]:
        return sorted(self._where)

    def get(self, name: str) -> torch.Tensor:
        with self._open(str(self._where[name]), "pt") as h:
            return h.get_tensor(name)


class RangeSafetensors:
    """safetensors read by byte ranges, for files that are not on this disk."""

    def __init__(self, files: Iterable[str], fetch: Callable[[str, int, int], bytes]):
        self._fetch = fetch
        self._where: Dict[str, tuple] = {}
        for f in files:
            n = struct.unpack("<Q", fetch(f, 0, 7))[0]
            header = json.loads(fetch(f, 8, 8 + n - 1))
            for k, v in header.items():
                if k == "__metadata__":
                    continue
                start, end = v["data_offsets"]
                self._where[k] = (f, v["dtype"], tuple(v["shape"]), 8 + n + start, 8 + n + end)

    def keys(self) -> List[str]:
        return sorted(self._where)

    def get(self, name: str) -> torch.Tensor:
        f, dtype, shape, start, end = self._where[name]
        if end == start:
            return torch.empty(shape, dtype=DTYPES[dtype])
        data = self._fetch(f, start, end - 1)
        return torch.frombuffer(bytearray(data), dtype=DTYPES[dtype]).reshape(shape)


class Renamed:
    """A reader seen through rename(name) -> new name or None (None hides the tensor)."""

    def __init__(self, reader, rename: Callable[[str], Optional[str]]):
        self._reader = reader
        self._map = {}
        for k in reader.keys():
            new = rename(k)
            if new is None:
                continue
            if new in self._map:
                raise ValueError(f"two tensors renamed to {new}: {self._map[new]}, {k}")
            self._map[new] = k

    def keys(self) -> List[str]:
        return sorted(self._map)

    def get(self, name: str) -> torch.Tensor:
        return self._reader.get(self._map[name])


def text_model_name(name: str) -> Optional[str]:
    """Common names for the text model of multimodal and text-only checkpoints (Qwen3.5 layout).

    model.language_model.X -> model.X; vision tower (model.visual.*) and multi-token-prediction head (mtp.*) hidden.
    """
    if name.startswith(("model.visual.", "visual.", "mtp.")):
        return None
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    return name


def hf_fetch(repo: str, revision: str = "main", retries: int = 5) -> Callable[[str, int, int], bytes]:
    """fetch(file, start, end_inclusive) over HTTP range requests to huggingface.co (public repos)."""

    def fetch(name, start, end):
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} of {end - start + 1}")
                return data
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    return fetch


def hf_files(repo: str, revision: str = "main", subfolder: str = "") -> List[str]:
    """The root (or subfolder) .safetensors files of a public hub repo."""
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}"
    with urllib.request.urlopen(url, timeout=60) as r:
        sib = json.loads(r.read())["siblings"]
    prefix = f"{subfolder}/" if subfolder else ""
    return sorted(s["rfilename"] for s in sib if s["rfilename"].endswith(".safetensors")
                  and s["rfilename"].startswith(prefix) and "/" not in s["rfilename"][len(prefix):])


def family(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("embed", "wte", "token_embd")):
        return "embed"
    if "lm_head" in n or n.startswith("output.") or n == "output.weight":
        return "head"
    if "norm" in n or "ln_" in n:
        return "norm"
    if any(k in n for k in ("linear_attn", "mamba", "ssm", "conv1d", "a_log", "dt_bias")):
        return "ssm"
    if any(k in n for k in ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj")):
        return "attn"
    if any(k in n for k in ("mlp", "ffn", "expert", "gate_proj", "up_proj", "down_proj", "shared_mlp")):
        return "mlp"
    return "other"


def layer_of(name: str) -> Optional[int]:
    m = re.search(r"(?:layers|blk|h|block|blocks)\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _blocks(a: torch.Tensor, chunk_rows: int):
    for i in range(0, a.shape[0], chunk_rows):
        yield i, slice(i, min(i + chunk_rows, a.shape[0]))


def _gram(a: torch.Tensor, b: torch.Tensor, chunk_rows: int, device="cpu") -> torch.Tensor:
    """Gram matrix of D = b - a on its smaller side, accumulated in row blocks (float64)."""
    rows, cols = a.shape
    if cols <= rows:
        G = torch.zeros(cols, cols, dtype=torch.float64, device=device)
        for _, sl in _blocks(a, chunk_rows):
            Dc = (b[sl].to(device, torch.float32) - a[sl].to(device, torch.float32)).double()
            G += Dc.T @ Dc
    else:
        G = torch.zeros(rows, rows, dtype=torch.float64, device=device)
        for j in range(0, cols, chunk_rows):
            sl = slice(j, min(j + chunk_rows, cols))
            Dc = (b[:, sl].to(device, torch.float32) - a[:, sl].to(device, torch.float32)).double()
            G += Dc @ Dc.T
    return G


def delta_basis(a: torch.Tensor, b: torch.Tensor, k: int, chunk_rows: int = 4096, device="cpu"):
    """Squared singular values (descending) and top-k singular vectors of D = b - a on its smaller side."""
    ev, vec = torch.linalg.eigh(_gram(a, b, chunk_rows, device))
    ev, vec = ev.flip(0).clamp(min=0), vec.flip(1)
    return ev.cpu(), vec[:, :k].cpu()


def tensor_delta_stats(name: str, a: torch.Tensor, b: torch.Tensor, k: int = 16, top_rows: int = 20,
                       chunk_rows: int = 4096, device="cpu") -> Dict:
    """Statistics of W' = W + D, read in row blocks so a large matrix never exists twice in float32."""
    if a.dim() < 2:
        a2, b2 = a.reshape(1, -1), b.reshape(1, -1)
    else:
        a2, b2 = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    changed, nb2, nd2, nt2, dot, mx = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    row_d, row_b = [], []
    for _, sl in _blocks(a2, chunk_rows):
        ac, bc = a2[sl], b2[sl]
        changed += int((ac != bc).sum())
        A, B = ac.to(device, torch.float32).double(), bc.to(device, torch.float32).double()
        D = B - A
        nb2 += float((A * A).sum()); nd2 += float((D * D).sum()); nt2 += float((B * B).sum())
        dot += float((A * B).sum())
        if D.numel():
            mx = max(mx, float(D.abs().max()))
        row_d.append(D.norm(dim=1).cpu()); row_b.append(A.norm(dim=1).cpu())
    nb, nd, nt = nb2 ** 0.5, nd2 ** 0.5, nt2 ** 0.5
    s = {"tensor": name, "family": family(name), "layer": layer_of(name), "shape": list(a.shape),
         "numel": a.numel(), "norm_base": nb, "norm_delta": nd,
         "rel_delta": nd / nb if nb > 0 else (0.0 if nd == 0 else float("inf")),
         "cosine": dot / (nb * nt) if nb > 0 and nt > 0 else 1.0,
         "changed_fraction": changed / max(a.numel(), 1), "changed": changed, "max_abs_delta": mx}
    if a.dim() != 2:
        return s
    if nd == 0:
        s.update(energy_top1=0.0, energy_topk=0.0, stable_rank=0.0, top_singular_values=[])
    else:
        ev, _ = delta_basis(a, b, k, chunk_rows, device)
        e = float(ev.sum())  # = ||D||^2, computed independently of nd2
        s.update(energy_top1=float(ev[0]) / e, energy_topk=float(ev[:k].sum()) / e,
                 stable_rank=e / float(ev[0]), top_singular_values=[round(float(x) ** 0.5, 6) for x in ev[:k]])
    if any(t in name.lower() for t in ROW_TENSORS):
        rn, base_rows = torch.cat(row_d), torch.cat(row_b)
        s["rows_changed"] = int((rn > 0).sum())
        idx = torch.argsort(rn, descending=True)[:top_rows]
        s["top_rows"] = [{"row": int(i), "delta_norm": float(rn[i]),
                          "rel": float(rn[i] / base_rows[i]) if base_rows[i] > 0 else None}
                         for i in idx if rn[i] > 0]
    return s


def delta_overlap(a: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor, k: int = 16, chunk_rows: int = 4096,
                  device="cpu") -> Dict:
    """How alike two updates of the same base are: D1 = b1 - a, D2 = b2 - a.

    cosine: of the flattened deltas (sign matters). subspace_overlap: ||V1^T V2||_F^2 / k for their top-k singular
    vectors on the smaller side (1 = same directions, sign-blind; about k/d for unrelated deltas).
    """
    a2 = a.reshape(a.shape[0], -1) if a.dim() >= 2 else a.reshape(1, -1)
    b1, b2 = b1.reshape(a2.shape), b2.reshape(a2.shape)
    n1, n2, dot = 0.0, 0.0, 0.0
    for _, sl in _blocks(a2, chunk_rows):
        A = a2[sl].to(device, torch.float32).double()
        D1, D2 = b1[sl].to(device, torch.float32).double() - A, b2[sl].to(device, torch.float32).double() - A
        n1 += float((D1 * D1).sum()); n2 += float((D2 * D2).sum()); dot += float((D1 * D2).sum())
    out = {"cosine": dot / (n1 * n2) ** 0.5 if n1 > 0 and n2 > 0 else 0.0,
           "norm_ratio": (n2 / n1) ** 0.5 if n1 > 0 else None}
    if a.dim() == 2 and n1 > 0 and n2 > 0:
        kk = min(k, min(a.shape))
        _, V1 = delta_basis(a, b1, kk, chunk_rows, device)
        _, V2 = delta_basis(a, b2, kk, chunk_rows, device)
        out["subspace_overlap"] = float((V1.T @ V2).pow(2).sum()) / kk
        out["random_overlap"] = kk / min(a.shape)
    return out


def _totals(rows: List[Dict]) -> Dict:
    nb2 = sum(r["norm_base"] ** 2 for r in rows)
    nd2 = sum(r["norm_delta"] ** 2 for r in rows)
    n = sum(r["numel"] for r in rows)
    return {"tensors": len(rows), "numel": n, "rel_delta": (nd2 ** 0.5) / (nb2 ** 0.5) if nb2 else 0.0,
            "changed_fraction": sum(r["changed"] for r in rows) / max(n, 1)}


def diff_models(base, target, pattern: str = "*", k: int = 16, progress: Optional[Callable] = None,
                on_tensor: Optional[Callable[[Dict], None]] = None, skip: Iterable[str] = ()) -> Dict:
    """Every tensor present in both (matching `pattern`); names only in one side and shape mismatches listed.

    on_tensor(row) is called after each tensor (for incremental saving); names in `skip` are not read again.
    """
    kb, kt = set(base.keys()), set(target.keys())
    names = sorted(n for n in kb & kt if fnmatch.fnmatch(n, pattern))
    skip = set(skip)
    rows, mismatch = [], []
    for i, n in enumerate(names):
        if n in skip:
            continue
        a, b = base.get(n), target.get(n)
        if tuple(a.shape) != tuple(b.shape):
            mismatch.append(n)
            continue
        row = tensor_delta_stats(n, a, b, k=k)
        rows.append(row)
        if on_tensor:
            on_tensor(row)
        if progress:
            progress(i + 1, len(names), n)
    return summarize(rows, only_in_base=sorted(n for n in kb - kt if fnmatch.fnmatch(n, pattern)),
                     only_in_target=sorted(n for n in kt - kb if fnmatch.fnmatch(n, pattern)),
                     shape_mismatch=mismatch)


def summarize(rows: List[Dict], **extra) -> Dict:
    fams, layers = {}, {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
        if r["layer"] is not None:
            layers.setdefault(r["layer"], []).append(r)
    return {"overall": _totals(rows), "by_family": {f: _totals(v) for f, v in sorted(fams.items())},
            "by_layer": {l: _totals(v) for l, v in sorted(layers.items())},
            "tensors": rows, **extra}
