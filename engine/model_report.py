"""engine/model_report.py - a fast report of how a model differs from its base, and how.

The fingerprint reads one block of rows from every tensor of both models (contiguous bytes in safetensors, so one
HTTP range each for remote models): about 1-6 MB per matrix instead of the whole checkpoint. From those blocks:

  touched / untouched tensor types   exact comparison after rounding both to the lower of the two dtypes
                                     (a F32 -> BF16 cast is not a change);
  rank of each change                directions holding 90 % of the block's delta energy (a low-rank update stays
                                     low rank in any block of rows);
  direction consistency              |cos| between the leading row-side directions of different tensors' blocks
                                     (one global direction, as in abliteration, gives ~1).

classify() turns that into a method:
  identical              nothing changed;
  directional_ablation   only residual-writing matrices (o_proj, out_proj, down_proj), low rank, one direction
                         one direction for all layers (abliteration) or one per layer (Heretic ARA),
                         reported, not required; reports the layer range;
  lora_like              low rank, some module families untouched (adapter on chosen modules);
  full_fine_tune         diffuse changes.
Rules come from lab/probes/delta_patterns/findings.md (F5, F9, F11, F15, F17) and are tested on synthetic models
with exact recipes (tests/test_model_report.py).
"""
import statistics as st
from typing import Dict, List, Optional

import torch

from engine import weight_diff as wd

RESIDUAL_WRITERS = ("o_proj.weight", "out_proj.weight", "down_proj.weight")
_PRECISION = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2, torch.float64: 3}


def _common_dtype(a, b):
    if a.dtype == b.dtype or a.dtype not in _PRECISION or b.dtype not in _PRECISION:
        return a, b
    low = a.dtype if _PRECISION[a.dtype] < _PRECISION[b.dtype] else b.dtype
    return a.to(low), b.to(low)


def _block(reader, name, shape, rows):
    if len(shape) < 2 or shape[0] <= rows:
        return reader.get(name).reshape(shape[0] if shape else 1, -1), 0
    start = shape[0] // 2 - rows // 2
    return reader.get_rows(name, start, rows).reshape(rows, -1), start


def fingerprint(base, model, rows: int = 128, progress=None, workers: int = 16) -> Dict:
    """workers: blocks fetched concurrently (remote reads are latency-bound: one HTTP range per block)."""
    import concurrent.futures
    kb, km = set(base.keys()), set(model.keys())
    names = sorted(kb & km)
    out = {"rows": rows, "only_in_base": sorted(kb - km), "only_in_model": sorted(km - kb),
           "shape_mismatch": [], "dtype_differs": [], "tensors": []}
    shapes = {n: (tuple(base.shape(n)), tuple(model.shape(n))) for n in names}
    out["shape_mismatch"] = [n for n in names if shapes[n][0] != shapes[n][1]]
    names = [n for n in names if shapes[n][0] == shapes[n][1]]

    def fetch(n):
        sa = shapes[n][0]
        (a, start), (b, _) = _block(base, n, sa, rows), _block(model, n, sa, rows)
        return n, sa, a, start, b

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as ex:
        fetched = ex.map(fetch, names)  # in order; up to `workers` blocks in flight
        for i, (n, sa, a, start, b) in enumerate(fetched):
            out["tensors"].append(_block_stats(n, sa, a, start, b, out))
            if progress:
                progress(i + 1, len(names), n)
    return out


def _block_stats(n, sa, a, start, b, out) -> Dict:
    if a.dtype != b.dtype:
        out["dtype_differs"].append(n)
    a, b = _common_dtype(a, b)
    changed = int((a != b).sum())
    A, B = a.float().double(), b.float().double()
    Dl = B - A
    na, nd = float(A.norm()), float(Dl.norm())
    row = {"tensor": n, "kind": wd.tensor_kind(n), "layer": wd.layer_of(n), "shape": list(sa),
           "block_start": start, "block_rows": int(a.shape[0]), "changed": changed,
           "changed_fraction": changed / max(a.numel(), 1), "rel_delta": nd / na if na > 0 else float(nd > 0)}
    if len(sa) == 2 and changed and min(Dl.shape) > 1:
        U, S, _ = torch.linalg.svd(Dl, full_matrices=False)
        e = S ** 2
        cum = torch.cumsum(e, 0) / e.sum()
        row["rank90"] = int((cum < 0.9).sum()) + 1
        row["energy16"] = float(cum[min(15, len(cum) - 1)])
        row["_u1"] = U[:, 0]
    return row


def classify(fp: Dict) -> Dict:
    rows = fp["tensors"]
    kinds: Dict[str, List[Dict]] = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r)
    touched = sorted(k for k, v in kinds.items() if any(r["changed"] for r in v))
    untouched = sorted(k for k in kinds if k not in touched)
    res = {"touched": touched, "untouched": untouched}
    if not touched:
        res["method"] = "identical"
        return res
    lr = [r for r in rows if "rank90" in r]
    ranks = [r["rank90"] for r in lr]
    dims = [min(r["block_rows"], r["shape"][1]) for r in lr]
    res["rank_estimate"] = max(ranks) if ranks else None
    res["rank_median"] = st.median(ranks) if ranks else None
    low_rank = bool(ranks) and st.median(ranks) <= 0.5 * st.median(dims) and max(ranks) < st.median(dims)
    layers = sorted({r["layer"] for r in rows if r["changed"] and r["layer"] is not None})
    res["layers"] = [layers[0], layers[-1]] if layers else []
    res["layers_touched"] = layers
    only_rw = all(k.endswith(RESIDUAL_WRITERS) for k in touched)
    if only_rw and low_rank:
        # Only the matrices that write to the residual stream, low rank: a directional ablation. One direction for
        # all layers (classic abliteration) or one per layer (Heretic ARA): reported, not required.
        us = [r["_u1"] for r in lr if r["kind"].endswith(RESIDUAL_WRITERS) and r["_u1"].shape[0] == lr[0]["_u1"].shape[0]]
        cos = [abs(float(us[i] @ us[j])) for i in range(len(us)) for j in range(i + 1, len(us))]
        res["direction_consistency"] = st.median(cos) if cos else 1.0
        res["ablation_directions"] = "global" if res["direction_consistency"] > 0.5 else "per_layer"
        res["method"] = "directional_ablation"
    elif low_rank:
        res["method"] = "lora_like"
    else:
        res["method"] = "full_fine_tune"
    return res


def profile(fp: Dict) -> Dict:
    kinds: Dict[str, List[Dict]] = {}
    for r in fp["tensors"]:
        kinds.setdefault(r["kind"], []).append(r)
    out = {}
    for k, v in sorted(kinds.items()):
        ch = [r for r in v if r["changed"]]
        out[k] = {"tensors": len(v), "touched": len(ch), "rel_median": st.median(r["rel_delta"] for r in v),
                  "rank90_median": st.median(r["rank90"] for r in ch if "rank90" in r) if any("rank90" in r for r in ch) else None,
                  "layers": sorted({r["layer"] for r in ch if r["layer"] is not None})}
    return out


def build_report(base, model, rows: int = 128, model_name: str = "", base_name: str = "", card: Optional[Dict] = None,
                 progress=None) -> Dict:
    fp = fingerprint(base, model, rows=rows, progress=progress)
    return {"model": model_name, "base": base_name, "card": card or {}, "rows_per_block": rows,
            "headers": {k: fp[k] for k in ("only_in_base", "only_in_model", "shape_mismatch", "dtype_differs")},
            "classification": classify(fp), "profile": profile(fp),
            "tensors": [{k: v for k, v in r.items() if not k.startswith("_")} for r in fp["tensors"]]}


def render_markdown(rep: Dict) -> str:
    c, h = rep["classification"], rep["headers"]
    lines = [f"# Model report: {rep['model']}", "", f"Base: {rep['base']}  ", f"Blocks of {rep['rows_per_block']} rows per tensor.", ""]
    if rep.get("card"):
        lines += ["## Card", ""] + [f"- {k}: {v}" for k, v in rep["card"].items()] + [""]
    lines += ["## Compatibility", "",
              f"- only in base: {len(h['only_in_base'])}; only in model: {len(h['only_in_model'])}; "
              f"shape mismatch: {len(h['shape_mismatch'])}; stored dtype differs: {len(h['dtype_differs'])} "
              "(compared after rounding to the lower precision)", ""]
    lines += ["## How it was changed", "", f"- method: **{c['method']}**"]
    if c.get("layers"):
        lines.append(f"- layers {c['layers'][0]}-{c['layers'][1]} ({len(c['layers_touched'])} touched)")
    if c.get("rank_estimate") is not None:
        lines.append(f"- rank of the change in a block: median {c['rank_median']}, max {c['rank_estimate']} "
                     f"(directions for 90 % of the energy)")
    if "direction_consistency" in c:
        lines.append(f"- ablation directions: {c.get('ablation_directions')} (median |cos| between tensors "
                     f"{c['direction_consistency']:.3f})")
    lines += [f"- touched: {', '.join(c['touched']) or '-'}", f"- untouched: {', '.join(c['untouched']) or '-'}", ""]
    lines += ["## Where (median relative change per tensor type, in the sampled blocks)", "",
              "| type | touched / tensors | rel. change | rank90 | layers |", "|---|---|---|---|---|"]
    for k, p in sorted(rep["profile"].items(), key=lambda kv: -kv[1]["rel_median"]):
        lay = f"{p['layers'][0]}-{p['layers'][-1]}" if p["layers"] else "-"
        lines.append(f"| {k} | {p['touched']}/{p['tensors']} | {p['rel_median']:.4%} | {p['rank90_median'] or '-'} | {lay} |")
    return "\n".join(lines) + "\n"
