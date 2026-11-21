"""Adam / weight-decay footprint in embedding rows (Chaudhuri, X thread 14 Nov 2026; lab/probes/adam_sig/PLAN.md).

D = donor - base, row per token, float32 from the bf16 files. Row classes:
  zero   every coordinate exactly 0                       (token never in a batch and no weight decay on embeddings)
  decay  D = -c w with one c per row, residual < DECAY_TOL (never in a batch, weight decay only: shrink (1-lr wd)^T)
  step   |D_ij| nearly constant across j, CV < STEP_CV    (Adam's sign update of a single step, magnitude = lr-ish)
  other  anything else

  python lab/probes/adam_sig/adam_sig.py <base_hf_dir> <donor_hf_dir> <out.json> [--tensor embed|lm_head] [--top 20]
CPU only; the two matrices are read with safetensors (no model load) and processed in row chunks.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
DECAY_TOL, STEP_CV, STEP_MIN = 0.05, 0.15, 1e-7
NAMES = {"embed": "model.language_model.embed_tokens.weight", "lm_head": "lm_head.weight"}


def classify(W, D):
    """W, D float32 (rows, d) -> dict of per-row arrays: cls (str), norm, decay_c, decay_res, step_cv, step_mag."""
    W = np.asarray(W, dtype=np.float32)
    D = np.asarray(D, dtype=np.float32)
    n = W.shape[0]
    norm = np.linalg.norm(D, axis=1)
    zero = norm == 0
    ww = np.einsum("ij,ij->i", W, W)
    c = np.where(ww > 0, -np.einsum("ij,ij->i", D, W) / np.maximum(ww, 1e-30), 0.0)
    res = np.linalg.norm(D + c[:, None] * W, axis=1) / np.maximum(norm, 1e-30)
    decay = (~zero) & (c > 0) & (res < DECAY_TOL)
    a = np.abs(D)
    mean = a.mean(axis=1)
    cv = np.where(mean > 0, a.std(axis=1) / np.maximum(mean, 1e-30), np.inf)
    step = (~zero) & (~decay) & (cv < STEP_CV) & (mean > STEP_MIN)
    cls = np.full(n, "other", dtype=object)
    cls[step] = "step"
    cls[decay] = "decay"
    cls[zero] = "zero"
    return {"cls": cls, "norm": norm, "decay_c": c, "decay_res": res, "step_cv": cv, "step_mag": mean}


def classify_chunked(W, D, chunk=16384):
    parts = [classify(W[i:i + chunk], D[i:i + chunk]) for i in range(0, W.shape[0], chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def zero_rows_a(D):
    return int(np.sum(np.all(np.asarray(D) == 0, axis=1)))


def zero_rows_b(D):
    return int(np.sum(np.abs(np.asarray(D)).max(axis=1) == 0))


def _shard(hf_dir, name):
    idx = json.loads((Path(hf_dir) / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    return str(Path(hf_dir) / idx[name])


def run(base_dir, donor_dir, out, tensor="embed", top=20, chunk=8192):
    import torch
    from safetensors import safe_open
    name = NAMES[tensor]
    parts, za, zb = [], 0, 0
    with safe_open(_shard(base_dir, name), framework="pt") as fb, safe_open(_shard(donor_dir, name), framework="pt") as fd:
        sb, sd = fb.get_slice(name), fd.get_slice(name)
        shape = tuple(sb.get_shape())
        assert shape == tuple(sd.get_shape()), (shape, sd.get_shape())
        for i in range(0, shape[0], chunk):          # rows read per chunk from the files: peak RAM well under 1 GB
            Wb = sb[i:i + chunk].to(torch.float32).numpy()
            D = sd[i:i + chunk].to(torch.float32).numpy() - Wb
            parts.append(classify(Wb, D))
            za += zero_rows_a(D)
            zb += zero_rows_b(D)
    r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    class Wb:  # shape holder for the report
        pass
    Wb.shape = shape
    cls = r["cls"]
    counts = {k: int(np.sum(cls == k)) for k in ("zero", "decay", "step", "other")}
    assert za == zb == counts["zero"], (za, zb, counts)
    order = np.argsort(-r["norm"])
    toks = None
    try:
        from engine.tok import load_tokenizer
        tok = load_tokenizer(base_dir)
        toks = lambda i: tok.decode([int(i)])
    except Exception as e:  # noqa: BLE001
        print("tokenizer unavailable:", e)
    med = float(np.median(r["norm"][r["norm"] > 0])) if np.any(r["norm"] > 0) else 0.0
    rep = {"tensor": name, "rows": int(Wb.shape[0]), "d": int(Wb.shape[1]), "counts": counts,
           "identical": bool(counts["zero"] == Wb.shape[0]),
           "zero_rows_two_methods": [za, zb],
           "decay_c": _summary(r["decay_c"][cls == "decay"]),
           "step_mag": _summary(r["step_mag"][cls == "step"]),
           "step_mag_levels": _levels(r["step_mag"][cls == "step"]),
           "norm_median_nonzero": med,
           "top": [{"id": int(i), "tok": toks(i) if toks else None, "norm": float(r["norm"][i]),
                    "x_median": float(r["norm"][i] / med) if med else None, "cls": str(cls[i])} for i in order[:top]],
           "zero_sample_ids": [int(i) for i in np.flatnonzero(cls == "zero")[:: max(1, counts["zero"] // 20 or 1)][:20]],
           "zero_sample_toks": [toks(i) for i in np.flatnonzero(cls == "zero")[:: max(1, counts["zero"] // 20 or 1)][:20]] if toks else None,
           "thresholds": {"DECAY_TOL": DECAY_TOL, "STEP_CV": STEP_CV, "STEP_MIN": STEP_MIN}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("tensor", "rows", "counts", "identical", "decay_c", "step_mag", "norm_median_nonzero")}))
    return rep


def _summary(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {"n": int(x.size), "min": float(q[0]), "p10": float(q[1]), "median": float(q[2]), "p90": float(q[3]), "max": float(q[4])}


def _levels(x, rel=0.02, max_levels=12):
    """Greedy 1-D clustering of step magnitudes: levels within `rel` of each other; returns the biggest clusters."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.size == 0:
        return []
    levels, start = [], 0
    for k in range(1, x.size + 1):
        if k == x.size or x[k] > x[start] * (1 + rel):
            levels.append({"mag": float(x[start:k].mean()), "n": int(k - start)})
            start = k
    levels.sort(key=lambda l: -l["n"])
    return levels[:max_levels]


if __name__ == "__main__":
    a = sys.argv[1:]
    tensor = a[a.index("--tensor") + 1] if "--tensor" in a else "embed"
    top = int(a[a.index("--top") + 1]) if "--top" in a else 20
    run(a[0], a[1], a[2], tensor=tensor, top=top)
