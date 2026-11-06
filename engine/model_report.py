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
import re
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
                 progress=None, atlas: Optional[Dict] = None) -> Dict:
    fp = fingerprint(base, model, rows=rows, progress=progress)
    c, prof = classify(fp), profile(fp)
    near = atlas_similarity(prof, atlas) if atlas else []
    return {"model": model_name, "base": base_name, "card": card or {}, "rows_per_block": rows,
            "headers": {k: fp[k] for k in ("only_in_base", "only_in_model", "shape_mismatch", "dtype_differs")},
            "classification": {k: v for k, v in c.items()}, "profile": prof, "atlas": near,
            "usable": usable(c, card or {}, near),
            "tensors": [{k: v for k, v in r.items() if not k.startswith("_")} for r in fp["tensors"]]}


def reader_for(spec: str, root=None):
    """A local directory/file of safetensors, or a Hugging Face repo id read by HTTP ranges."""
    from pathlib import Path
    p = Path(spec)
    if not p.exists() and root is not None and (Path(root) / spec).exists():
        p = Path(root) / spec
    if p.exists():
        return wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
    return wd.Renamed(wd.RangeSafetensors(wd.hf_files(spec), wd.hf_fetch(spec)), wd.text_model_name)


FLAG_WORDS = {
    # regular expressions, matched on whole words ("ties" must not match "properties")
    "heretic": (r"heretic",), "abliteration": (r"abliterat\w*", r"refusal direction", r"orthogonali[sz]\w*"),
    "lora": (r"q?lora", r"peft", r"adapters?"), "merge": (r"merged? models?", r"mergekit", r"slerp", r"ties-merg\w*",
                                                          r"dare-ties", r"model merg\w*"),
    "distillation": (r"distill\w*",), "rl": (r"grpo", r"dapo", r"dppo", r"ppo", r"reinforcement[- ]learning", r"rl"),
    "sft": (r"sft", r"supervised fine-?tun\w*"), "uncensored": (r"uncensored", r"decensor\w*"),
    "non_commercial": (r"non-?commercial", r"non commercial", r"cc-by-nc\S*"),
}


def card_info(repo: str, fetch_json=None, fetch_text=None) -> Dict:
    """What the Hugging Face card says: declared base, license, and words that name a method (not trusted, shown)."""
    import json
    import urllib.request
    fetch_json = fetch_json or (lambda u: json.loads(urllib.request.urlopen(u, timeout=60).read()))
    fetch_text = fetch_text or (lambda u: urllib.request.urlopen(u, timeout=60).read().decode("utf-8", "replace"))
    api = fetch_json(f"https://huggingface.co/api/models/{repo}?cardData=true")
    cd = api.get("cardData") or {}
    try:
        readme = fetch_text(f"https://huggingface.co/{repo}/raw/main/README.md")
    except Exception:
        readme = ""
    text = (readme + " " + " ".join(api.get("tags", [])) + " " + str(cd.get("license_name", ""))).lower()
    base = cd.get("base_model")
    return {"repo": repo, "base_model": base[0] if isinstance(base, list) and base else base,
            "license": cd.get("license_name") or cd.get("license"),
            "flags": sorted(f for f, words in FLAG_WORDS.items()
                            if any(re.search(rf"(?<![\w-]){w}(?![\w])", text) for w in words))}


def atlas_similarity(prof: Dict, atlas: Dict) -> List[Dict]:
    """Spearman of the per-type profile with each studied model's; atlas: name -> {family, profile{kind: rel}}."""
    mine = {k: v["rel_median"] for k, v in prof.items()}
    out = []
    for name, entry in atlas.items():
        ks = sorted(set(mine) & set(entry["profile"]))
        if len(ks) < 3:
            continue
        rk = lambda d: {k: i for i, k in enumerate(sorted(ks, key=lambda k: d[k]))}
        a, b = rk(mine), rk(entry["profile"])
        n = len(ks)
        out.append({"model": name, "family": entry["family"],
                    "spearman": 1 - 6 * sum((a[k] - b[k]) ** 2 for k in ks) / (n * (n * n - 1))})
    return sorted(out, key=lambda r: -r["spearman"])


# Families of the models studied in lab/probes/delta_patterns (findings F1-F17).
ATLAS_FAMILIES = {"mimo": "reasoning distillation, full", "neohorse": "agent, full", "fara": "agent, full",
                  "tmax": "agent RL, full", "nyx_rp": "role-play LoRA (control)",
                  "abliterated": "abliteration", "wemm_embed": "embedding model (control)",
                  "angry_claudius": "degradation (control)", "dsv4_distill": "reasoning distillation, LoRA-like",
                  "official_post": "official post-training (Base -> instruct)"}


def load_atlas(stats_dir) -> Dict:
    """Per-type median relative change of every studied model with full statistics (stats_<name>.jsonl)."""
    import json
    from pathlib import Path
    atlas = {}
    for name, fam in ATLAS_FAMILIES.items():
        p = Path(stats_dir) / f"stats_{name}.jsonl"
        if not p.exists():
            continue
        g: Dict[str, List[float]] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            g.setdefault(wd.tensor_kind(r["tensor"]), []).append(r["rel_delta"])
        atlas[name] = {"family": fam, "profile": {k: st.median(v) for k, v in g.items()}}
    return atlas


def usable(c: Dict, card: Dict, near: List[Dict]) -> List[str]:
    """What can be reused and how, stated with the evidence level it has today."""
    tips = []
    m = c.get("method")
    if m == "identical":
        return ["Nothing to take: the weights equal the base."]
    if m == "directional_ablation":
        lay = c.get("layers") or ["?", "?"]
        tips.append(f"Decensoring component: rank {c.get('rank_estimate')} per tensor, "
                    f"{c.get('ablation_directions', '?')} directions, layers {lay[0]}-{lay[1]}, residual-writing "
                    "matrices only. Reproducible with WSPLang `project_out` per layer (directions taken from the "
                    "model's own delta). It removes safety behaviour: use only where that is acceptable.")
    elif m == "lora_like":
        tips.append(f"Adapter-like update (rank about {c.get('rank_estimate')}) on: {', '.join(c.get('touched', []))}. "
                    "Small and self-contained: it can be applied to the base as a low-rank delta; its effect is "
                    "whatever the adapter was trained for, nothing more.")
    elif m == "full_fine_tune":
        tips.append("Diffuse full fine-tune: rank-one ops cannot reproduce it. Candidate for `subspace_transfer` "
                    "along the residual directions that improved models share (findings F7, F12, F14); a transfer "
                    "must beat a random subspace of the same size before it counts (plan H4, not yet run).")
    if near:
        top = near[0]
        tips.append(f"Closest studied model by where it changes: {top['model']} ({top['family']}), "
                    f"Spearman {top['spearman']:.2f}.")
    flags = set(card.get("flags", []))
    if "non_commercial" in flags:
        tips.append("The card mentions a non-commercial license: check before any reuse.")
    if {"heretic", "abliteration", "uncensored"} & flags and m != "directional_ablation":
        tips.append("The card mentions decensoring, but no separate ablation component was found in the sampled rows.")
    return tips


def render_markdown(rep: Dict) -> str:
    c, h = rep["classification"], rep["headers"]
    lines = [f"# Model report: {rep['model']}", "", f"Base: {rep['base']}  ", f"Blocks of {rep['rows_per_block']} rows per tensor.", ""]
    if rep.get("card"):
        lines += ["## Card (what the model page says; not verified)", ""]
        lines += [f"- {k}: {v}" for k, v in rep["card"].items() if k != "flags"]
        lines += [f"- words the page mentions: {', '.join(rep['card'].get('flags', [])) or '-'}", ""]
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
    lines += ["", "\"Untouched\" means unchanged in the sampled rows; a full `delta_stats` settles it.", ""]
    if rep.get("atlas"):
        lines += ["## Closest studied models (Spearman of where they change)", ""]
        lines += [f"- {r['model']} ({r['family']}): {r['spearman']:.2f}" for r in rep["atlas"][:5]] + [""]
    if rep.get("usable"):
        lines += ["## What is usable, and how", ""] + [f"- {t}" for t in rep["usable"]] + [""]
    return "\n".join(lines) + "\n"
