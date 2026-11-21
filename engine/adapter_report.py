"""engine/adapter_report.py - what a PEFT LoRA adapter changes, read from the adapter alone (plus base row blocks).

A LoRA repo holds adapter_config.json and adapter_model.safetensors with lora_A (r x in) and lora_B (out x r) per
module; the change it applies is dW = scale * B @ A, scale = alpha / r (alpha / sqrt(r) with rslora). Everything
below is computed from the factors, never from a materialised dW of the base's size:

  delta_norm           ||dW||_F by the trace identity  ||BA||^2 = sum((B^T B) * (A A^T))  (method 1)
  delta_norm_svd       sqrt(sum s_i^2) from the exact singular values of dW via thin QR of B and A^T (method 2)
  effective_rank_90    directions holding 90 % of the delta energy (<= r)
  rel_delta_block      ||dW[rows]|| / ||W[rows]|| on the same middle block of rows model_report uses, W read from the
                       base by one range per tensor (local or Hugging Face)
  direction_consistency per kind: median |cos| between the leading output-side directions of different layers
                       (one global direction, as in abliteration, gives ~1; independent per-layer updates ~0)

No claim about behaviour: this says where and how large the change is, nothing about what it does.
"""
import json
import statistics as st
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import torch

from engine import weight_diff as wd

_A, _B = ".lora_A.weight", ".lora_B.weight"
_EA, _EB = ".lora_embedding_A", ".lora_embedding_B"
_PEFT_PREFIX = "base_model.model."


def _hub_json(repo: str, name: str, revision: str = "main") -> Optional[dict]:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _header_params(reader) -> int:
    """Parameter count from the safetensors headers alone (byte extents / dtype size), no tensor read."""
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    n = 0
    for k, (f, dtype, shape, start, end) in reader._where.items():
        n += (end - start) // sizes[dtype]
    return n


def _base_name(stem: str) -> str:
    """PEFT module stem -> canonical base tensor name (model.layers.N.self_attn.q_proj.weight)."""
    if stem.startswith(_PEFT_PREFIX):
        stem = stem[len(_PEFT_PREFIX):]
    if not (stem.startswith("model.") or stem.startswith("lm_head")):
        stem = "model." + stem
    name = wd.text_model_name(stem + ".weight")
    return name if name is not None else stem + ".weight"


def load_adapter(spec: str, root=None) -> Dict:
    """A local adapter directory, or hf:owner/repo / owner/repo on the hub. Only PEFT LORA adapters."""
    p = Path(spec)
    if not p.exists() and root is not None and (Path(root) / spec).exists():
        p = Path(root) / spec
    if p.exists():
        cfg = json.loads((p / "adapter_config.json").read_text(encoding="utf-8"))
        reader = wd.LocalSafetensors(p)
        source, nbytes = str(p), sum(f.stat().st_size for f in ([p] if p.is_file() else p.glob("*.safetensors")))
    else:
        repo = spec[3:] if spec.startswith("hf:") else spec
        cfg = _hub_json(repo, "adapter_config.json")
        if cfg is None:
            raise ValueError(f"{spec}: no adapter_config.json (not a PEFT adapter repo, or not reachable)")
        files = wd.hf_files(repo)
        fetch = wd.hf_fetch(repo)
        reader = wd.RangeSafetensors(files, fetch)
        source = f"hf:{repo}"
        nbytes = sum(max(end for (_, _, _, _, end) in (v for v in reader._where.values() if v[0] == f))
                     for f in files)
    peft_type = str(cfg.get("peft_type", "")).upper()
    if peft_type != "LORA":
        raise ValueError(f"{spec}: peft_type {peft_type or 'missing'}; only LORA adapters are read here")
    r, alpha = int(cfg["r"]), float(cfg.get("lora_alpha", cfg["r"]))
    rslora = bool(cfg.get("use_rslora", False))
    scale = alpha / (r ** 0.5) if rslora else alpha / r
    modules: Dict[str, Dict] = {}
    for k in reader.keys():
        for a_suf, b_suf, emb in ((_A, _B, False), (_EA, _EB, True)):
            if k.endswith(a_suf):
                stem = k[: -len(a_suf)]
                b = stem + b_suf
                if b not in reader._where:
                    raise ValueError(f"{spec}: {k} has no matching {b_suf}")
                modules[_base_name(stem)] = {"A": k, "B": b, "stem": stem, "embedding": emb}
    if not modules:
        raise ValueError(f"{spec}: no lora_A / lora_B tensors found")
    return {"source": source, "config": cfg, "r": r, "alpha": alpha, "rslora": rslora, "scale": scale,
            "targets": cfg.get("target_modules"), "base_model": cfg.get("base_model_name_or_path"),
            "rank_pattern": cfg.get("rank_pattern") or {}, "alpha_pattern": cfg.get("alpha_pattern") or {},
            "modules": modules, "reader": reader, "bytes": int(nbytes)}


def _module_scale(adapter: Dict, stem: str, r_eff: int) -> float:
    alpha = adapter["alpha"]
    for key, val in adapter["alpha_pattern"].items():
        if key in stem:
            alpha = float(val)
    return alpha / (r_eff ** 0.5) if adapter["rslora"] else alpha / r_eff


def _factor_stats(B: torch.Tensor, A: torch.Tensor, scale: float) -> Dict:
    """Exact ||dW||_F two ways, singular values and leading output-side direction of dW = scale * B @ A."""
    B, A = B.double(), A.double()
    G = (B.T @ B) * (A @ A.T)                                  # elementwise; sum = ||BA||_F^2  (trace identity)
    fro = scale * float(G.sum().clamp_min(0).sqrt())
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    Us, S, Vs = torch.linalg.svd(scale * (Rb @ Ra.T))
    e = S ** 2
    tot = float(e.sum())
    svd_norm = tot ** 0.5
    if tot > 0:
        cum = torch.cumsum(e, 0) / tot
        rank90 = int((cum < 0.9).sum()) + 1
        u1 = (Qb @ Us[:, 0]).float()
    else:
        rank90, u1 = 0, torch.zeros(B.shape[0])
    return {"delta_norm": fro, "delta_norm_svd": svd_norm, "singular_values": [float(s) for s in S],
            "effective_rank_90": rank90, "_u1": u1}


def adapter_report(adapter: Dict, base: Optional[str] = None, rows: int = 128, workers: int = 16,
                   progress=None, root=None) -> Dict:
    reader, scale0 = adapter["reader"], adapter["scale"]
    base_reader = None
    if base is not None:
        from engine import model_report as mr
        base_reader = mr.reader_for(base, root=root)
    names = sorted(adapter["modules"])
    out_rows: List[Dict] = []

    def one(n):
        m = adapter["modules"][n]
        A, B = reader.get(m["A"]).float(), reader.get(m["B"]).float()
        if m["embedding"]:                       # PEFT embedding LoRA: A (r, num_emb), B (dim, r); dW = (B A)^T
            A, B = B.T.contiguous(), A.T.contiguous()
        r_eff = int(A.shape[0])
        sc = _module_scale(adapter, m["stem"], r_eff)
        row = {"tensor": n, "kind": wd.tensor_kind(n), "layer": wd.layer_of(n), "r": r_eff, "scale": sc,
               "shape": [int(B.shape[0]), int(A.shape[1])], "params_adapter": int(A.numel() + B.numel())}
        row.update(_factor_stats(B, A, sc))
        if base_reader is not None:
            if n in set(base_reader.keys()):
                sb = tuple(base_reader.shape(n))
                row["shape_matches_base"] = list(sb) == row["shape"]
                if row["shape_matches_base"] and len(sb) == 2:
                    start = 0 if sb[0] <= rows else sb[0] // 2 - rows // 2
                    cnt = min(rows, sb[0])
                    Wb = base_reader.get_rows(n, start, cnt).float()
                    dWb = sc * (B[start:start + cnt].double() @ A.double())
                    nb = float(Wb.double().norm())
                    row.update({"block_start": start, "block_rows": int(Wb.shape[0]),
                                "rel_delta_block": float(dWb.norm()) / nb if nb > 0 else None})
            else:
                row["shape_matches_base"] = False
                row["base_missing"] = True
        return row

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as ex:
        for i, row in enumerate(ex.map(one, names)):
            out_rows.append(row)
            if progress:
                progress(i + 1, len(names), row["tensor"])

    kinds: Dict[str, List[Dict]] = {}
    for r in out_rows:
        kinds.setdefault(r["kind"], []).append(r)
    touched = sorted(kinds)
    dc = {}
    for k, rs in kinds.items():
        us = [r["_u1"] for r in rs if r["_u1"].norm() > 0]
        cos = [abs(float(us[i] @ us[j])) for i in range(len(us)) for j in range(i + 1, len(us))
               if us[i].shape == us[j].shape]
        dc[k] = st.median(cos) if cos else None
    per_kind = {k: {"modules": len(rs), "layers": sorted({r["layer"] for r in rs if r["layer"] is not None}),
                    "rank90_median": st.median(r["effective_rank_90"] for r in rs),
                    "delta_norm_median": st.median(r["delta_norm"] for r in rs),
                    "rel_delta_block_median": (st.median(r["rel_delta_block"] for r in rs if r.get("rel_delta_block") is not None)
                                               if any(r.get("rel_delta_block") is not None for r in rs) else None),
                    "direction_consistency": dc[k]} for k, rs in sorted(kinds.items())}
    untouched = None
    if base_reader is not None:
        untouched = sorted({wd.tensor_kind(n) for n in base_reader.keys()} - set(touched))
    layers = sorted({r["layer"] for r in out_rows if r["layer"] is not None})
    fam = {"touched": touched, "untouched": untouched,
           "embeddings_touched": any("embed" in k for k in touched),
           "head_touched": any("lm_head" in k for k in touched),
           "outside_targets": sorted({r["kind"] for r in out_rows
                                      if adapter["targets"] and not any(t in r["kind"] for t in adapter["targets"])})}
    totals = {"params_from_shapes": sum(r["params_adapter"] for r in out_rows),
              "params_from_header": _header_params(reader), "bytes": adapter["bytes"],
              "modules": len(out_rows), "layers": [layers[0], layers[-1]] if layers else [],
              "delta_norm_total": float(sum(r["delta_norm"] ** 2 for r in out_rows) ** 0.5)}
    for r in out_rows:
        r.pop("_u1", None)
    return {"adapter": adapter["source"], "r": adapter["r"], "alpha": adapter["alpha"], "rslora": adapter["rslora"],
            "scale": scale0, "targets": adapter["targets"], "rows": rows,
            "base": {"in_card": adapter["base_model"], "given": base,
                     "matches": (base is None or adapter["base_model"] is None
                                 or adapter["base_model"].split("/")[-1].lower() in str(base).lower())},
            "totals": totals, "families": fam, "per_kind": per_kind, "direction_consistency": dc,
            "modules": sorted(out_rows, key=lambda r: (-r["delta_norm"], r["tensor"]))}


def _f(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def to_markdown(rep: Dict) -> str:
    t, f = rep["totals"], rep["families"]
    lines = [f"# Adapter report: {rep['adapter']}", "",
             f"PEFT LoRA, r = {rep['r']}, alpha = {rep['alpha']}{' (rslora)' if rep['rslora'] else ''}, "
             f"scale {_f(rep['scale'])}; targets {rep['targets']}.",
             f"Base in the card: `{rep['base']['in_card']}`; compared against: `{rep['base']['given']}`"
             + ("" if rep["base"]["matches"] else " (**does not match the card**)") + ".",
             f"Adapter: {t['modules']} modules, layers {t['layers']}, {t['params_from_shapes']:,} parameters "
             f"(header count {t['params_from_header']:,}), {t['bytes'] / 1e6:.1f} MB; total ||dW||_F {_f(t['delta_norm_total'], 3)}.",
             "", "## Families", "",
             f"- touched: {', '.join(f['touched'])}",
             f"- untouched: {', '.join(f['untouched']) if f['untouched'] is not None else '(no base given)'}",
             f"- embeddings touched: {f['embeddings_touched']}; head touched: {f['head_touched']}"
             + (f"; outside the stated targets: {', '.join(f['outside_targets'])}" if f["outside_targets"] else ""),
             "", "## Per kind", "",
             "| kind | modules | layers | rank90 median | median ||dW|| | median rel_delta (block) | direction consistency |",
             "|---|---|---|---|---|---|---|"]
    for k, v in rep["per_kind"].items():
        ly = f"{v['layers'][0]}-{v['layers'][-1]}" if v["layers"] else "-"
        lines.append(f"| {k} | {v['modules']} | {ly} | {_f(v['rank90_median'], 1)} | {_f(v['delta_norm_median'], 3)} | "
                     f"{_f(v['rel_delta_block_median'])} | {_f(v['direction_consistency'], 3)} |")
    lines += ["", "## Modules (largest ||dW|| first, up to 40)", "",
              "| tensor | shape | r | ||dW|| (trace) | ||dW|| (svd) | rank90 | rel_delta block |", "|---|---|---|---|---|---|---|"]
    for r in rep["modules"][:40]:
        lines.append(f"| {r['tensor']} | {r['shape'][0]}x{r['shape'][1]} | {r['r']} | {_f(r['delta_norm'], 3)} | "
                     f"{_f(r['delta_norm_svd'], 3)} | {r['effective_rank_90']} | {_f(r.get('rel_delta_block'))} |")
    lines += ["", "No claim about behaviour: where and how large the change is, nothing about what it does.", ""]
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import time
    ap = argparse.ArgumentParser(description="PEFT LoRA adapter report (adapter alone, plus base row blocks)")
    ap.add_argument("adapter")
    ap.add_argument("--base")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", help="markdown path; a .json with the same stem is written beside it")
    a = ap.parse_args(argv)
    t0 = time.time()
    ad = load_adapter(a.adapter)
    rep = adapter_report(ad, base=a.base, rows=a.rows, workers=a.workers,
                         progress=lambda i, n, name: print(f"[{i}/{n}] {name}", flush=True) if i % 25 == 0 or i == n else None)
    rep["seconds"] = round(time.time() - t0, 1)
    md = to_markdown(rep)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
        p.with_suffix(".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(md)
    print(f"seconds {rep['seconds']}")


if __name__ == "__main__":
    main()
