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
_common_dtype = wd.common_dtype


def _block(reader, name, shape, rows):
    if len(shape) < 2 or shape[0] <= rows:
        return reader.get(name).reshape(shape[0] if shape else 1, -1), 0
    start = shape[0] // 2 - rows // 2
    return reader.get_rows(name, start, rows).reshape(rows, -1), start


def fingerprint(base, model, rows: int = 128, progress=None, workers: int = 16, k: int = 16, only=None) -> Dict:
    """workers: blocks fetched concurrently (remote reads are latency-bound: one HTTP range per block).
    k: column-side directions kept per block, for transfer_candidates."""
    import concurrent.futures
    kb, km = set(base.keys()), set(model.keys())
    names = sorted(n for n in kb & km if only is None or only(n))
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
            out["tensors"].append(_block_stats(n, sa, a, start, b, out, k))
            if progress:
                progress(i + 1, len(names), n)
    return out


def _block_stats(n, sa, a, start, b, out, k=16) -> Dict:
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
        U, S, Vh = torch.linalg.svd(Dl, full_matrices=False)
        e = S ** 2
        cum = torch.cumsum(e, 0) / e.sum()
        row["rank90"] = int((cum < 0.9).sum()) + 1
        row["energy16"] = float(cum[min(15, len(cum) - 1)])
        row["_u1"] = U[:, 0]
        if sa[1] <= sa[0] and not n.endswith(RESIDUAL_WRITERS):  # columns = smaller side = residual stream
            row["_v"] = Vh[:k].T.float()
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


def transfer_candidates(fp: Dict, atlas_sigs: Dict, k: int = 16, factor: float = 3.0) -> Dict:
    """Which tensor types share column-side (residual) directions with studied improvements.

    For each sampled tensor whose columns are its smaller side, the block's top-k right singular vectors are compared
    with the same tensor's signature basis in each atlas model: overlap ||Vs^T Vb||^2 / k, against the same type at
    other layers (null) and chance k/d. shared = same >= factor * null and >= factor * chance (the H3 rule).
    Residual-writing matrices (rows = residual) cannot be judged from a block of rows: listed as needing a full
    delta_signature.
    """
    rows = [r for r in fp["tensors"] if "_v" in r]
    kinds: Dict[str, List[Dict]] = {}
    for r in rows:
        kinds.setdefault(r["kind"], []).append(r)
    by_kind: Dict[str, Dict] = {}
    for kind, rs in kinds.items():
        for model, sig in atlas_sigs.items():
            same, null = [], []
            for r in rs:
                s = sig.get(r["tensor"])
                if not s or "basis" not in s or s["basis"].shape[0] != r["_v"].shape[0]:
                    continue
                kk = min(k, s["basis"].shape[1], r["_v"].shape[1])
                Vb, Vs = s["basis"][:, :kk].double(), r["_v"][:, :kk].double()
                same.append(float((Vs.T @ Vb).pow(2).sum()) / kk)
                for r2 in rs:
                    s2 = sig.get(r2["tensor"])
                    if r2 is r or not s2 or "basis" not in s2 or s2["basis"].shape[0] != Vs.shape[0]:
                        continue
                    null.append(float((Vs.T @ s2["basis"][:, :kk].double()).pow(2).sum()) / kk)
            if not same:
                continue
            m, nl, chance = st.median(same), (st.median(null) if null else None), kk / Vs.shape[0]
            by_kind.setdefault(kind, {})[model] = {
                "same": m, "null_other_layer": nl, "chance": chance, "x_null": m / nl if nl else None,
                "shared": bool(nl is not None and m >= factor * nl and m >= factor * chance)}
    candidates = sorted(k_ for k_, v in by_kind.items() if sum(x["shared"] for x in v.values()) >= 1)
    writers = sorted({r["kind"] for r in fp["tensors"] if r["kind"].endswith(RESIDUAL_WRITERS) and r["changed"]})
    return {"by_kind": by_kind, "candidates": candidates, "needs_full_signature": writers}


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


TRANSFER_ROWS = 512   # 128 rows: 12/21 verdicts as the full analysis; 256: 17/21; 512: 21/21 (findings F20)


def build_report(base, model, rows: int = 128, model_name: str = "", base_name: str = "", card: Optional[Dict] = None,
                 progress=None, atlas: Optional[Dict] = None, atlas_sigs: Optional[Dict] = None,
                 transfer_rows: int = TRANSFER_ROWS, nonweights: Optional[Dict] = None) -> Dict:
    """nonweights: the result of nonweight_diff (config, tokenizer, template), shown in its own section."""
    fp = fingerprint(base, model, rows=rows, progress=progress)
    c, prof = classify(fp), profile(fp)
    near = atlas_similarity(prof, atlas) if atlas else []
    transfer = None
    if atlas_sigs and c["method"] != "identical":
        col_side = {r["tensor"] for r in fp["tensors"] if len(r["shape"]) == 2 and r["shape"][1] <= r["shape"][0]
                    and not r["tensor"].endswith(RESIDUAL_WRITERS) and r["changed"]}
        fp2 = fingerprint(base, model, rows=transfer_rows, only=lambda n: n in col_side)
        transfer = transfer_candidates(fp2, atlas_sigs)
        transfer["needs_full_signature"] = sorted({r["kind"] for r in fp["tensors"]
                                                   if r["kind"].endswith(RESIDUAL_WRITERS) and r["changed"]})
        transfer["rows"] = transfer_rows
    return {"model": model_name, "base": base_name, "card": card or {}, "rows_per_block": rows,
            "headers": {k: fp[k] for k in ("only_in_base", "only_in_model", "shape_mismatch", "dtype_differs")},
            "classification": {k: v for k, v in c.items()}, "profile": prof, "atlas": near,
            "transfer": transfer, "nonweights": nonweights,
            "usable": usable(c, card or {}, near, transfer) + _nonweight_tips(nonweights),
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


SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "special_tokens_map.json", "chat_template.jinja")
# One short text per script; letters with combining marks (Indic, Thai, Arabic diacritics) are where pre-tokenizer
# regexes disagree (findings F25).
PROBE_TEXTS = {
    "english": "The quick brown fox jumps over the lazy dog. hello world!",
    "spanish": "El pingüino comió una canción: ¿qué más? Ñandú, acción.",
    "code": "def f(x):\n    return x**2  # square\n\tprint(f(3))",
    "math": "∫ x² dx = x³/3 + C, and 3.14159 × 2 = 6.28318",
    "chinese": "我们今天去北京看长城。",
    "japanese": "東京へ行きます。カタカナとひらがな。",
    "korean": "안녕하세요, 만나서 반갑습니다.",
    "russian": "Съешь же ещё этих мягких французских булок.",
    "vietnamese": "Tiếng Việt có dấu rất đẹp.",
    "hindi": "नमस्ते दुनिया, आप कैसे हैं?",
    "bengali": "আমি বাংলায় কথা বলি।",
    "thai": "สวัสดีครับ ยินดีต้อนรับ",
    "arabic_diacritics": "مَرْحَبًا بِكُمْ فِي البَيْتِ",
    "emoji": "Nice 🙂👍🏽 done ✅",
}
_CONFIG_NOISE = ("transformers_version", "_name_or_path", "_commit_hash")


def side_files(spec: str, root=None):
    """get(filename) -> bytes or None, for the small non-weight files of a local folder or a public HF repo."""
    import urllib.error
    import urllib.request
    from pathlib import Path
    p = Path(spec)
    if not p.exists() and root is not None and (Path(root) / spec).exists():
        p = Path(root) / spec
    if p.exists():
        d = p if p.is_dir() else p.parent
        return lambda f: (d / f).read_bytes() if (d / f).exists() else None

    def get(f):
        try:
            with urllib.request.urlopen(f"https://huggingface.co/{spec}/resolve/main/{f}", timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None
            raise
    return get


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def _config_diff(a: Dict, b: Dict) -> Dict:
    """changed: real differences; cosmetic: library version, absent-vs-None, a key moved with the same value."""
    absent = object()
    fa, fb = _flat(a), _flat(b)
    leaves = lambda f: {(k.rsplit(".", 1)[-1], repr(v)) for k, v in f.items()}  # noqa: E731
    la, lb = leaves(fa), leaves(fb)
    changed, cosmetic = [], []
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k, absent), fb.get(k, absent)
        if va == vb:
            continue
        name = k.rsplit(".", 1)[-1]
        if name in _CONFIG_NOISE:
            cosmetic.append(f"{k}: {va} -> {vb}")
        elif (va is absent and vb is None) or (vb is absent and va is None):
            cosmetic.append(f"{k}: absent vs None")
        elif va is absent and (name, repr(vb)) in la:
            cosmetic.append(f"{k}: moved, same value {vb}")
        elif vb is absent and (name, repr(va)) in lb:
            cosmetic.append(f"{k}: moved, same value {va}")
        else:
            changed.append({"key": k, "base": "<absent>" if va is absent else va,
                            "model": "<absent>" if vb is absent else vb})
    return {"changed": changed, "cosmetic": cosmetic}


def _json(get, f):
    import json
    raw = get(f)
    return json.loads(raw.decode("utf-8")) if raw else None


def _merges(tj, get):
    m = (tj or {}).get("model", {}).get("merges")
    if m is None:
        raw = get("merges.txt")
        m = [l for l in raw.decode("utf-8").splitlines() if l and not l.startswith("#version")] if raw else None
    return None if m is None else [" ".join(x) if isinstance(x, list) else x for x in m]


def _template(get):
    raw = get("chat_template.jinja")
    if raw:
        return raw.decode("utf-8")
    tc = _json(get, "tokenizer_config.json") or {}
    t = tc.get("chat_template")
    if isinstance(t, list):   # named templates
        t = next((x.get("template") for x in t if x.get("name") == "default"), None) or (t[0].get("template") if t else None)
    return t


def nonweight_diff(get_base, get_model, probes: Dict[str, str] = PROBE_TEXTS) -> Dict:
    """What differs outside the weights: files, config, tokenizer (vocab, merges, added tokens, normalizer,
    pre-tokenizer, decoder), chat template, and an encode test per script (same ids or not, token counts)."""
    import difflib
    have_b = {f: get_base(f) for f in SIDE_FILES}
    have_m = {f: get_model(f) for f in SIDE_FILES}
    gb, gm = (lambda f: have_b.get(f)), (lambda f: have_m.get(f))
    files = {"only_in_base": [f for f in SIDE_FILES if have_b[f] and not have_m[f]],
             "only_in_model": [f for f in SIDE_FILES if have_m[f] and not have_b[f]],
             "identical": [f for f in SIDE_FILES if have_b[f] and have_b[f] == have_m[f]]}
    out = {"files": files, "verdict": []}
    cb, cm = _json(gb, "config.json"), _json(gm, "config.json")
    out["config"] = _config_diff(cb, cm) if cb is not None and cm is not None else None
    gcb, gcm = _json(gb, "generation_config.json"), _json(gm, "generation_config.json")
    out["generation_config"] = _config_diff(gcb, gcm) if gcb is not None and gcm is not None else None

    tb, tm = _json(gb, "tokenizer.json"), _json(gm, "tokenizer.json")
    tok = {}
    if tb is not None or tm is not None or have_b["vocab.json"] or have_m["vocab.json"]:
        vb = (tb or {}).get("model", {}).get("vocab") or _json(gb, "vocab.json")
        vm = (tm or {}).get("model", {}).get("vocab") or _json(gm, "vocab.json")
        ab = {t["content"]: t["id"] for t in (tb or {}).get("added_tokens", [])}
        am = {t["content"]: t["id"] for t in (tm or {}).get("added_tokens", [])}
        tok = {"vocab_same": vb == vm, "vocab_size": [len(vb or {}), len(vm or {})],
               "merges_same": _merges(tb, gb) == _merges(tm, gm),
               "added_only_in_base": sorted(set(ab) - set(am)), "added_only_in_model": sorted(set(am) - set(ab)),
               "added_id_changed": sorted(k for k in set(ab) & set(am) if ab[k] != am[k])}
        if tb is not None and tm is not None:
            for part in ("normalizer", "pre_tokenizer", "decoder", "post_processor"):
                tok[f"{part}_same"] = tb.get(part) == tm.get(part)
    out["tokenizer"] = tok

    enc = None
    if tb is not None and tm is not None:
        from tokenizers import Tokenizer
        a = Tokenizer.from_str(have_b["tokenizer.json"].decode("utf-8"))
        b = Tokenizer.from_str(have_m["tokenizer.json"].decode("utf-8"))
        per = {}
        for k, t in probes.items():
            ia, ib = a.encode(t, add_special_tokens=False).ids, b.encode(t, add_special_tokens=False).ids
            per[k] = {"same": ia == ib, "tokens": [len(ia), len(ib)]}
        enc = {"per_script": per, "differs": [k for k, v in per.items() if not v["same"]]}
    out["encode"] = enc

    ta, tt = _template(gb), _template(gm)
    if ta is None and tt is None:
        tpl = {"status": "missing"}
    elif ta is None or tt is None:
        tpl = {"status": "missing_in_base" if ta is None else "missing_in_model"}
    elif ta == tt:
        tpl = {"status": "same"}
    else:
        d = [l for l in difflib.unified_diff(ta.splitlines(), tt.splitlines(), lineterm="", n=0)
             if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        tpl = {"status": "different", "lines_changed": len(d), "diff_head": d[:20]}
    out["template"] = tpl

    v = out["verdict"]
    if out["config"] and out["config"]["changed"]:
        v.append("config changes: " + ", ".join(c["key"] for c in out["config"]["changed"][:8]))
    if tok and not tok.get("vocab_same", True):
        v.append("different vocabulary: token ids do not line up with the base")
    if tok.get("added_only_in_model"):
        v.append(f"{len(tok['added_only_in_model'])} added tokens not in the base: "
                 + ", ".join(tok["added_only_in_model"][:6]))
    for part in ("normalizer", "pre_tokenizer"):
        if tok.get(f"{part}_same") is False:
            v.append(f"{part.replace('_', '-')} differs from the base")
    if enc and enc["differs"]:
        v.append("text encodes differently in: " + ", ".join(
            f"{k} ({enc['per_script'][k]['tokens'][0]} -> {enc['per_script'][k]['tokens'][1]} tokens)"
            for k in enc["differs"]))
    if tpl["status"] == "different":
        v.append(f"chat template differs ({tpl['lines_changed']} lines)")
    elif tpl["status"].startswith("missing_in"):
        v.append(f"chat template {tpl['status'].replace('_', ' ')} (may ship inside a GGUF instead)")
    return out


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


IMPROVEMENTS = ("mimo", "neohorse", "tmax", "fara")


def load_atlas_sigs(sig_dir, names=IMPROVEMENTS) -> Dict:
    from pathlib import Path
    out = {}
    for n in names:
        p = Path(sig_dir) / f"sig_{n}.pt"
        if p.exists():
            out[n] = torch.load(p)
    return out


def usable(c: Dict, card: Dict, near: List[Dict], transfer: Optional[Dict] = None) -> List[str]:
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
    if transfer and transfer.get("candidates"):
        lines = []
        for kind in transfer["candidates"]:
            who = [f"{m} x{v['x_null']:.1f}" for m, v in transfer["by_kind"][kind].items() if v["shared"]]
            lines.append(f"{kind} ({', '.join(who)})")
        tips.append("Shares residual directions with studied improvements (same layer, >= 3x the other-layer null), "
                    "so a `subspace_transfer` along them is a candidate: " + "; ".join(lines) + ". A candidate is not "
                    "a result: the transfer must beat a random subspace of the same size on a benchmark (plan H4).")
    elif transfer is not None:
        tips.append("No column-side tensor type shares directions with the studied improvements (H3 rule).")
    if transfer and transfer.get("needs_full_signature"):
        tips.append("Residual-writing matrices (" + ", ".join(transfer["needs_full_signature"]) + ") cannot be judged "
                    "from blocks of rows: run `delta_signature` + `delta_compare` for them.")
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


def _nonweight_tips(nw: Optional[Dict]) -> List[str]:
    if not nw:
        return []
    tips = []
    t, enc = nw.get("tokenizer") or {}, nw.get("encode") or {}
    if t and not t.get("vocab_same", True):
        tips.append("The vocabulary differs from the base: weight deltas of embed_tokens/lm_head do not line up "
                    "row by row, and nothing can be transferred there without a token map.")
    if enc.get("differs"):
        tips.append("Text encodes differently from the base in " + ", ".join(enc["differs"]) + ". Use one "
                    "tokenizer for both models when comparing or transferring, and check which one the model was "
                    "trained with (perplexity under each) before trusting results in those scripts.")
    if nw.get("template", {}).get("status") == "different":
        tips.append("The chat template differs: benchmark each model with its own template, and a transferred delta "
                    "may need the donor's template to show its behaviour.")
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
    if rep.get("transfer"):
        t = rep["transfer"]
        lines += [f"## Shared directions with studied improvements ({t['rows']} rows per block, column side)", "",
                  "| type | " + " | ".join(sorted({m for v in t["by_kind"].values() for m in v})) + " |",
                  "|---|" + "---|" * len({m for v in t["by_kind"].values() for m in v})]
        models = sorted({m for v in t["by_kind"].values() for m in v})
        for kind, v in sorted(t["by_kind"].items()):
            cells = [(f"x{v[m]['x_null']:.1f}" + (" **shared**" if v[m]["shared"] else "")) if m in v and v[m]["x_null"]
                     else "-" for m in models]
            lines.append(f"| {kind} | " + " | ".join(cells) + " |")
        lines += ["", "x = same-layer overlap / same type at other layers; shared = >= 3x that and >= 3x chance.", ""]
    nw = rep.get("nonweights")
    if nw:
        lines += ["## Outside the weights (config, tokenizer, chat template)", ""]
        lines += [f"- {v}" for v in nw["verdict"]] or ["- nothing that changes behaviour"]
        f = nw["files"]
        lines.append(f"- files only in base: {', '.join(f['only_in_base']) or '-'}; only in model: "
                     f"{', '.join(f['only_in_model']) or '-'}; byte-identical: {', '.join(f['identical']) or '-'}")
        if nw.get("config"):
            lines.append(f"- config: {len(nw['config']['changed'])} real changes, "
                         f"{len(nw['config']['cosmetic'])} cosmetic (library version, moved keys, absent vs None)")
        t = nw.get("tokenizer") or {}
        if t:
            parts = [f"{p.replace('_', '-')} {'same' if t[p + '_same'] else '**differs**'}"
                     for p in ("normalizer", "pre_tokenizer", "decoder") if p + "_same" in t]
            lines.append(f"- tokenizer: vocab {'same' if t['vocab_same'] else '**differs**'}, merges "
                         f"{'same' if t['merges_same'] else '**differs**'}, " + ", ".join(parts))
        if nw.get("encode"):
            lines += ["", "| script | same ids | tokens base -> model |", "|---|---|---|"]
            for k, v in nw["encode"]["per_script"].items():
                lines.append(f"| {k} | {'yes' if v['same'] else '**no**'} | {v['tokens'][0]} -> {v['tokens'][1]} |")
        lines += ["", f"- chat template: {nw['template']['status'].replace('_', ' ')}", ""]
    if rep.get("usable"):
        lines += ["## What is usable, and how", ""] + [f"- {t}" for t in rep["usable"]] + [""]
    return "\n".join(lines) + "\n"
