"""How a Q8_0 GGUF fine-tune differs from its bf16 base, without downloading either (HTTP ranges).

Q8_0 is deterministic: re-quantizing the base's bf16 rows the way llama.cpp does reproduces the published bytes for
every row the fine-tune did not touch. Per tensor, on one block of `rows` middle rows:
  identical   share of 32-value Q8_0 blocks byte-identical to Q8_0(base)   (1.0 = untouched; BF16 tensors: exact values)
  rel_delta   ||deq(model) - deq(Q8_0(base))|| / ||base||
  top1/rank90 energy share of the delta's first singular value, rank holding 90 % (abliteration: rank 1)
  u           leading left singular vector over the block's output rows (one global direction -> |cos| ~ 1)
Two Q8_0 recipes are tried per block (gguf-py numpy = convert_hf_to_gguf --outtype q8_0, and the C reference
x * (1/d)); the one matching more blocks is used, and the counts are reported.

usage: python gguf_q8_delta.py <gguf url> <base repo> <out.json> [--rows 128] [--layers 0-63]
"""
import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import weight_diff as wd  # noqa: E402

GG2HF = {"ffn_down": "mlp.down_proj", "ffn_up": "mlp.up_proj", "ffn_gate": "mlp.gate_proj",
         "attn_output": "self_attn.o_proj", "ssm_out": "linear_attn.out_proj", "attn_v": "self_attn.v_proj"}
RESIDUAL = ("ffn_down", "attn_output", "ssm_out")
Q8_0, BF16 = 8, 30


def ranged(url):
    def get(a, b):
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
        for k in range(5):
            try:
                return urllib.request.urlopen(req, timeout=120).read()
            except Exception:
                if k == 4:
                    raise
    return get


def header(get):
    buf = get(0, 40_000_000)
    pos = [4]

    def rd(fmt):
        v = struct.unpack_from("<" + fmt, buf, pos[0])
        pos[0] += struct.calcsize("<" + fmt)
        return v[0]

    def rstr():
        n = rd("Q")
        s = buf[pos[0]:pos[0] + n].decode("utf-8", "replace")
        pos[0] += n
        return s
    SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd("I"), rd("Q")
            return [rval(et) for _ in range(n)]
        return rd(SZ[t])
    assert buf[:4] == b"GGUF"
    rd("I")
    nt, nkv = rd("Q"), rd("Q")
    kv = {}
    for _ in range(nkv):
        k = rstr()
        kv[k] = rval(rd("I"))
    ti = {}
    for _ in range(nt):
        name = rstr()
        dims = [rd("Q") for _ in range(rd("I"))]
        ti[name] = (dims, rd("I"), rd("Q"))
    align = kv.get("general.alignment", 32)
    return ti, (pos[0] + align - 1) // align * align


def _pack(d, q):
    return np.concatenate([d.astype(np.float16).view(np.uint8), q.view(np.uint8)], axis=1)


def _roundf(q):
    return (np.sign(q) * np.floor(np.abs(q) + np.float32(0.5))).astype(np.int8)


def q8_numpy(x):
    """gguf-py (convert_hf_to_gguf): d = amax/127, q = round-half-away(x / d), d stored as f16."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1, keepdims=True) / np.float32(127)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d == 0, np.float32(0), b / d)
    return _pack(d, _roundf(q))


def q8_cref(x):
    """llama.cpp quantize_row_q8_0_ref: id = 1/d, q = roundf(x * id)."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1, keepdims=True) / np.float32(127)
    with np.errstate(divide="ignore"):
        idd = np.where(d == 0, np.float32(0), np.float32(1) / d)
    return _pack(d, _roundf(b * idd))


def deq8(blocks):
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    return d * blocks[:, 2:].view(np.int8).astype(np.float32)


def analyse(name, get, ti, data0, base, rows, recipe_stats):
    dims, typ, off = ti[name]
    ne0, ne1 = dims[0], dims[1]
    start = ne1 // 2 - rows // 2
    kind = name.split(".")[2]
    hf = f"model.layers.{name.split('.')[1]}.{GG2HF[kind]}.weight"
    B = base.get_rows(hf, start, rows).float().numpy()                       # rows x ne0
    if typ == BF16:
        rb = ne0 * 2
        raw = np.frombuffer(get(data0 + off + start * rb, data0 + off + (start + rows) * rb), dtype=np.uint16)
        M = (raw.astype(np.uint32) << 16).view(np.float32).reshape(rows, ne0)
        ident = float((M == B).mean())
        Bq = B
    elif typ == Q8_0:
        rb = ne0 // 32 * 34
        raw = np.frombuffer(get(data0 + off + start * rb, data0 + off + (start + rows) * rb), dtype=np.uint8)
        mb = raw.reshape(-1, 34)
        if kind == "ssm_out":                     # the GGUF converter reorders the value heads (input columns)
            B = B[:, head_order(deq8(mb).reshape(rows, ne0), B, HEAD_V)]
        best = None
        for rname, fn in (("numpy", q8_numpy), ("cref", q8_cref)):
            qb = fn(B)
            same = (qb == mb).all(axis=1)
            recipe_stats[rname] += int(same.sum())
            if best is None or same.mean() > best[0]:
                best = (float(same.mean()), qb)
        recipe_stats["blocks"] += len(mb)
        ident = best[0]
        M = deq8(mb).reshape(rows, ne0)
        Bq = deq8(best[1]).reshape(rows, ne0)
    else:
        return {"tensor": name, "kind": kind, "type": typ, "skipped": True}
    D = torch.tensor(M - Bq, dtype=torch.float64)
    rec = {"tensor": name, "kind": kind, "type": "Q8_0" if typ == Q8_0 else "BF16", "identical": ident,
           "rel_delta": float(D.norm() / max(np.linalg.norm(B), 1e-30))}
    if float(D.norm()) > 0:
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        e = (S ** 2) / (S ** 2).sum()
        rec.update(top1=float(e[0]), rank90=int((e.cumsum(0) < 0.9).sum()) + 1, u=U[:, 0].tolist())
        if typ == Q8_0 and ident < 0.99:
            # calibration: a pure rank-1 change of the same norm along the observed direction, through Q8_0
            D1 = (float(D.norm()) * torch.outer(U[:, 0], Vh[0])).numpy().astype(np.float32)
            Bq1 = deq8(q8_cref(B + D1)).reshape(rows, ne0)
            S1 = torch.linalg.svdvals(torch.tensor(Bq1 - Bq, dtype=torch.float64))
            rec["top1_if_rank1"] = float(S1[0] ** 2 / (S1 ** 2).sum())
    return rec


HEAD_V = 128


def head_order(M, B, h):
    """Column permutation of B's value heads (width h) that best matches M (greedy on head-to-head distance)."""
    n = B.shape[1] // h
    Mh = M.reshape(M.shape[0], n, h).transpose(1, 0, 2).reshape(n, -1)
    Bh = B.reshape(B.shape[0], n, h).transpose(1, 0, 2).reshape(n, -1)
    C = ((Mh[:, None, :] - Bh[None, :, :]) ** 2).sum(-1)          # target head i <- base head j
    order, used = [None] * n, set()
    for flat in np.argsort(C, axis=None):
        i, j = divmod(int(flat), n)
        if order[i] is None and j not in used:
            order[i] = j
            used.add(j)
    return np.concatenate([np.arange(j * h, (j + 1) * h) for j in order])


def summarize(recs, stats):
    res_u = [np.array(r["u"]) for r in recs if r.get("kind") in RESIDUAL and "u" in r and r["identical"] < 0.99]
    cos = [abs(float(x @ y)) for i, x in enumerate(res_u) for y in res_u[i + 1:]]
    summ = {"recipe_matches": stats, "tensors": len(recs), "by_kind": {},
            "residual_writer_u_abs_cos_median": float(np.median(cos)) if cos else None,
            "residual_writer_u_abs_cos_p10": float(np.quantile(cos, 0.1)) if cos else None}
    for k in sorted({r["kind"] for r in recs if not r.get("skipped")}):
        rs = [r for r in recs if r.get("kind") == k and not r.get("skipped")]
        touched = [r for r in rs if r["identical"] < 0.99]
        summ["by_kind"][k] = {
            "n": len(rs), "touched": len(touched),
            "identical_median": float(np.median([r["identical"] for r in rs])),
            "rel_delta_median_touched": float(np.median([r["rel_delta"] for r in touched])) if touched else 0.0,
            "top1_median": float(np.median([r["top1"] for r in touched])) if touched else None,
            "top1_if_rank1_median": (float(np.median([r["top1_if_rank1"] for r in touched if "top1_if_rank1" in r]))
                                     if any("top1_if_rank1" in r for r in touched) else None),
            "rank90_median": float(np.median([r["rank90"] for r in touched])) if touched else None,
            "touched_layers": [int(r["tensor"].split(".")[1]) for r in touched]}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--layers", default="0-63")
    a = ap.parse_args()
    get = ranged(a.url)
    ti, data0 = header(get)
    base = wd.Renamed(wd.RangeSafetensors(wd.hf_files(a.base), wd.hf_fetch(a.base)), wd.text_model_name)
    lo, hi = map(int, a.layers.split("-"))
    stats = {"numpy": 0, "cref": 0, "blocks": 0}
    recs = []
    for l in range(lo, hi + 1):
        for kind in ("ffn_down", "attn_output", "ssm_out", "ffn_up", "attn_v"):
            n = f"blk.{l}.{kind}.weight"
            if n in ti:
                r = analyse(n, get, ti, data0, base, a.rows, stats)
                recs.append(r)
                print(json.dumps({k: v for k, v in r.items() if k != "u"}), flush=True)
    summ = summarize(recs, stats)
    Path(a.out).write_text(json.dumps({"model": a.url, "base": a.base, "rows": a.rows, "summary": summ,
                                       "tensors": recs}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
