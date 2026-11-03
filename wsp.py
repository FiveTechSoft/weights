"""wsp.py â€” Weight-Space Programming toolkit (GGUF,æ ¼å¼ del blog MIT CSAIL).

Uso por un LLM driver: leer pesos (stats/rows/delta), proponer ediciones
(programa JSON), aplicarlas (edit: dequant->op->requant, copia en sitio),
verificar (diff), lanzar el modelo editado (serve) y evaluar A/B (eval).

Presupuestos (SKILL): cada invocaciÃ³n se journaliza; read<=200, write<=20, fwd<=6.
No hay gradientes, no entrenamiento, no fit a datos.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import gguf

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parent
DIR_DATA = ROOT / "data"
DIR_EDI = ROOT / "edits" if (ROOT / "edits").exists() else ROOT / "ediciones"
DIR_RIG = ROOT / "rig_runs"
DIR_LOG = ROOT / "logs"
DIR_JRN = ROOT / "journal"
DIR_SRV = ROOT / "served" if (ROOT / "served").exists() else ROOT / "servidos"
for d in (DIR_DATA, DIR_EDI, DIR_RIG, DIR_LOG, DIR_JRN, DIR_SRV):
    d.mkdir(parents=True, exist_ok=True)

LLAMA_BIN = Path(r"C:\Users\Anto\llamacpp\bin-b11962\llama-server.exe")
if not LLAMA_BIN.exists():
    LLAMA_BIN = Path(r"C:\Users\Anto\llamacpp\bin\llama-server.exe")
PROME_JRNL = DIR_JRN / "journal.jsonl"

JRN_KIND = {"info": "read", "tensors": "read", "stats": "read", "rows": "read",
            "delta": "read", "ops-find": "read", "diff": "read", "budget": "read",
            "peek": "read", "semantic-map": "read",
            "poke": "write", "edit": "write", "serve": "fwd", "eval": "fwd"}
BUDGET = {"read": 200, "write": 20, "fwd": 6}


def _p(*a):
    print(*a, flush=True)


# ---------------- journal ----------------
def journal(kind, detail):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "detail": detail}
    with PROME_JRNL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_budget(_):
    c = {"read": 0, "write": 0, "fwd": 0}
    if PROME_JRNL.exists():
        for line in PROME_JRNL.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                k = rec.get("kind")
                if k in c:
                    c[k] += 1
            except Exception:
                pass
    _p("budget (sesion wsp en C:\\weights\\journal):")
    _p(f"  read  {c['read']:<4} / {BUDGET['read']}  {'OK' if c['read'] < BUDGET['read'] else 'LIMITE'}")
    _p(f"  write {c['write']:<4} / {BUDGET['write']}  {'OK' if c['write'] < BUDGET['write'] else 'LIMITE'}")
    _p(f"  fwd   {c['fwd']:<4} / {BUDGET['fwd']}  {'OK' if c['fwd'] < BUDGET['fwd'] else 'LIMITE'}")


def _make_parser():
    ap = argparse.ArgumentParser(prog="wsp")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("info"); a.add_argument("gguf")
    a = sp.add_parser("tensors"); a.add_argument("gguf"); a.add_argument("pat", nargs="?", default="*")
    a = sp.add_parser("stats"); a.add_argument("gguf"); a.add_argument("tensor"); a.add_argument("ev", nargs="?", default="rows"); a.add_argument("-k", type=int, default=8); a.add_argument("--base", default=None)
    a = sp.add_parser("rows"); a.add_argument("gguf"); a.add_argument("tensor"); a.add_argument("rng"); a.add_argument("-n", type=int, default=8, help="valores por fila impresos"); a.add_argument("-full", action="store_true")
    a = sp.add_parser("delta"); a.add_argument("base"); a.add_argument("dist"); a.add_argument("tensor"); a.add_argument("-k", type=int, default=10); a.add_argument("-d", default="rows", choices=["rows", "cols"])
    a = sp.add_parser("edit"); a.add_argument("gguf"); a.add_argument("prog"); a.add_argument("-o", "--out", default=None)
    a = sp.add_parser("sha"); a.add_argument("gguf")
    a = sp.add_parser("diff"); a.add_argument("a"); a.add_argument("b"); a.add_argument("tensor", nargs="?", default="*"); a.add_argument("-frac", type=float, default=0.005)
    a = sp.add_parser("serve"); a.add_argument("gguf"); a.add_argument("-p", "--port", type=int, required=True); a.add_argument("-a", "--alias", default=None); a.add_argument("-c", "--ctx", type=int, default=16384); a.add_argument("-n", "--ngl", type=int, default=999); a.add_argument("--stop", action="store_true")
    a = sp.add_parser("eval"); a.add_argument("alias"); a.add_argument("-p", "--port", type=int, default=8950); a.add_argument("-n", type=int, default=10); a.add_argument("-o", "--offset", type=int, default=0); a.add_argument("--seed", type=int, default=42)
    a = sp.add_parser("vs"); a.add_argument("runA"); a.add_argument("runB")
    a = sp.add_parser("budget")
    a = sp.add_parser("markers"); a.add_argument("gguf")
    a = sp.add_parser("peek"); a.add_argument("gguf"); a.add_argument("tensor"); a.add_argument("-r", "--row", type=int, default=None); a.add_argument("-c", "--col", type=int, default=None); a.add_argument("-k", type=int, default=8); a.add_argument("--json", action="store_true")
    a = sp.add_parser("poke"); a.add_argument("gguf"); a.add_argument("tensor"); a.add_argument("-r", "--row", type=int, required=True); a.add_argument("-c", "--col", type=int, required=True); a.add_argument("-v", "--val", type=float, required=True); a.add_argument("-o", "--out", default=None)
    a = sp.add_parser("semantic-map"); a.add_argument("gguf")
    return ap


# ---------------- gguf helpers ----------------
_READER_CACHE = {}


def reader_of(path):
    p = str(path)
    if p not in _READER_CACHE:
        _READER_CACHE[p] = gguf.GGUFReader(p)
    return _READER_CACHE[p]


def tensor_map(r):
    return {t.name: t for t in r.tensors}


def row_decode(t):
    """Devuelve matriz D decodificada (contenedor Hu: filas=dim salida, cols=dim entrada); SIEMPRE escribible."""
    tt = t.tensor_type
    if tt == gguf.GGMLQuantizationType.F32:
        return np.array(t.data, dtype=np.float32)
    if tt == gguf.GGMLQuantizationType.F16:
        return np.array(t.data.view(np.float16), dtype=np.float32)
    if tt == gguf.GGMLQuantizationType.BF16:
        u = t.data.view(np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    return gguf.dequantize(t.data, tt).astype(np.float32)


def base_dtype(t):
    """etiqueta de tipo: F32/F16/BF16, o nombre de cuantizacion (Q6_K, Q8_0, ...)."""
    tt = t.tensor_type
    if tt == gguf.GGMLQuantizationType.F32:
        return "F32"
    if tt == gguf.GGMLQuantizationType.F16:
        return "F16"
    if tt == gguf.GGMLQuantizationType.BF16:
        return "BF16"
    return tt.name


# ---------------- read commands ----------------
def cmd_info(a):
    r = reader_of(a.gguf)
    hf = r.fields.get("general.architecture")
    arch = hf.contents()[0].decode() if hf else "?"
    n = len(r.tensors)
    qs = {}
    for t in r.tensors:
        qs[t.tensor_type.name] = qs.get(t.tensor_type.name, 0) + 1
    _p(f"file: {a.gguf}")
    _p(f"tensors: {n}  quant: {json.dumps(qs, sort_keys=True)}")
    _p("first/last:", r.tensors[0].name, "/", r.tensors[-1].name)


def cmd_tensors(a):
    import fnmatch
    r = reader_of(a.gguf)
    for t in r.tensors:
        if fnmatch.fnmatch(t.name, a.pat):
            _p(f"{t.name:<40} {[int(x) for x in t.shape]} {t.tensor_type.name:<7} {t.n_bytes/2**20:.1f}MiB off={t.data_offset}")


def cmd_stats(a):
    r = reader_of(a.gguf)
    t = tensor_map(r)[a.tensor]
    D = row_decode(t)
    evo = a.ev
    if evo == "rows":
        P = np.linalg.norm(D.astype(np.float64).reshape(D.shape[0], -1), axis=1)
    elif evo == "cols":
        P = np.linalg.norm(D.astype(np.float64).reshape(-1, D.shape[-1]), axis=0)
    else:
        sys.exit("ev debe ser rows|cols")
    q = np.quantile(P, [50, 90, 95, 99])
    mm = np.mean(np.abs(P))
    _p(f"{a.tensor} [{int(t.shape[1])}x{int(t.shape[0])}] {t.tensor_type.name}")
    _p(f"ev={evo}: n={P.size} mean={mm:.4g} p50={q[0]:.4g} p90={q[1]:.4g} p95={q[2]:.4g} p99={q[3]:.4g} max={P.max():.4g}")
    top = np.argsort(P)[::-1][:a.k]
    _p("top-norm " + ("filas" if evo == "rows" else "cols") + ":",
      ", ".join(f"{i}({P[i]:.4g})" for i in sorted(top.tolist())))


def cmd_rows(a):
    r = reader_of(a.gguf)
    t = tensor_map(r)[a.tensor]
    D = row_decode(t)
    s = a.rng.split(":")
    i0 = int(s[0]) if s[0] else 0
    i1 = int(s[1]) if len(s) > 1 and s[1] else i0 + 1
    sl = D[i0:i1]
    for k in range(sl.shape[0]):
        row = sl[k]
        if a.full:
            vals = np.array2string(row[:a.n].astype(np.float32), precision=6, max_line_width=10**6)
            _p(f"row {i0+k}: {row.size} elem  vals[:{a.n}] {vals}  mean|{np.abs(row).mean():.3g}| max|{np.abs(row).max():.3g}|")
        else:
            _p(f"row {i0+k}: mean|{np.abs(row).mean():.3g}| max|{np.abs(row).max():.3g}| l2={np.linalg.norm(row):.3g}")


def cmd_delta(a):
    rd = {t.name: t for t in reader_of(a.dist).tensors}[a.tensor]
    rb = {t.name: t for t in reader_of(a.base).tensors}[a.tensor]
    D = row_decode(rd)
    B = row_decode(rb)
    assert D.shape == B.shape, f"shape distinto {D.shape} vs {B.shape}"
    if a.d == "rows":
        num = np.linalg.norm((D - B).astype(np.float64).reshape(D.shape[0], -1), axis=1)
        den = np.linalg.norm(B.astype(np.float64).reshape(B.shape[0], -1), axis=1) + 1e-12
    else:
        num = np.linalg.norm((D - B).astype(np.float64).reshape(-1, D.shape[-1]), axis=0)
        den = np.linalg.norm(B.astype(np.float64).reshape(-1, B.shape[-1]), axis=0) + 1e-12
    rel = num / den
    nq = np.quantile(rel, [0.50, 0.90, 0.95, 0.99, 1.0])
    _p(f"{a.tensor} delta rows({'rel'})  p50={nq[0]:.3%} p90={nq[1]:.3%} p95={nq[2]:.3%} p99={nq[3]:.3%} max={nq[4]:.3%}")
    k = min(a.k, rel.size)
    idx = np.argsort(rel)[::-1][:k]
    _p("top-rel-rows:", ", ".join(f"{i}({rel[i]:.2%})" for i in sorted(idx.tolist())))


def cmd_markers(a):
    f = reader_of(a.gguf).fields["tokenizer.ggml.tokens"]
    toks = [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in f.contents()]
    import re
    def cat(tok):
        s = tok.replace("Ä ", " ")
        if re.match(r"^(&lt;|\|)", s) or re.match(r"^[<&].*[|>;&]$", s):
            return "estructural"
        if "\n" in tok and tok.strip("\n"):
            return "estructural"
        if tok.strip() == "":
            return "estructural"
        return None
    ids = [i for i, tg in enumerate(toks) if cat(tg)]
    out = DIR_DATA / "markers.json"
    out.write_text(json.dumps({str(i): toks[i] for i in ids}, ensure_ascii=False, indent=1), encoding="utf-8")
    _p(f"markers estructurales: {len(ids)}  -> {out}")


# ---------------- edit engine ----------------
QK_K = 256  # Q6_K: 256 elems = 16 subbloques de 16, 210 bytes: ql[128] qh[64] scales[16] d[2(fp16)]


def _q8_0_quant_blocks(X):
    """X (nb, 32) f32 -> bytes (nb, 34) uint8: d (f16) + 32 int8 â€” llama.cpp quantize_row_q8_0_ref."""
    nb = X.shape[0]
    Xf = X.astype(np.float64)
    amax = np.abs(Xf).max(axis=1)
    ok = amax > 1e-15
    d = np.where(ok, amax / 127.0, 1.0).astype(np.float16).astype(np.float32)
    q = np.rint(Xf / np.maximum(np.abs(d.astype(np.float64))[:, None], 1e-12))
    q *= 1.0
    q = np.clip(q, -127, 127).astype(np.int8)
    out = np.empty((nb, 34), dtype=np.uint8)
    out[:, 0:2] = np.asarray(d, dtype=np.float16).view(np.uint8).reshape(nb, 2)
    out[:, 2:34] = q.view(np.uint8)
    return out


def _requant_blocks_bytes(t, rowmap, D):
    """rowmap {fila almacenada: [bloques]}: requantiza SOLO esos bloques;
    el resto se copia byte-a-byte de t.data (cero ruido colateral)."""
    assert t.tensor_type in (gguf.GGMLQuantizationType.Q6_K, gguf.GGMLQuantizationType.Q8_0), "solo Q6_K/Q8_0"
    ne0 = int(t.shape[0])
    ne1 = int(t.shape[1])
    nbpr = int(t.data.shape[-1])
    # elementos por bloques: 210 bytes/256 elems (Q6_K) o 34/32 (Q8_0)
    elem_pb, byte_pb = (256, 210) if t.tensor_type == gguf.GGMLQuantizationType.Q6_K else (32, 34)
    nblk_row = ne0 // elem_pb
    out = np.frombuffer(t.data.tobytes(), dtype=np.uint8).reshape(ne1, nbpr).copy()
    pairs = [(r, b) for r, blks in rowmap.items() for b in blks]
    X = np.stack([D[r, b * elem_pb:(b + 1) * elem_pb] for (r, b) in pairs]).astype(np.float32)
    B = (_q6k_quant_blocks if t.tensor_type == gguf.GGMLQuantizationType.Q6_K else _q8_0_quant_blocks)(X)
    blocksz = 210 if t.tensor_type == gguf.GGMLQuantizationType.Q6_K else 34
    for k, (r, b) in enumerate(pairs):
        out[r, b * blocksz:(b + 1) * blocksz] = B[k]
    _p(f"  bloques requant: {len(pairs)} de {ne1 * nblk_row} ({100 * len(pairs) / (ne1 * nblk_row):.3f}%)")
    return out.tobytes()


def _q6k_index_map(i):
    """i del bloque (0..255) -> (r,c) plano del dequant gguf-pip y (g,sh,k) por grupo."""
    r = i // 32
    c = i % 32
    return r, c


def _make_qx_quants(Xsub, nmax=32):
    """Puerto de _make_qx_quants (llama.cpp): (n,16) -> (scale (n,), L (n,16) int64).
    max = x del mayor |x| (CON signo); is = -9..9; criterio sumlx^2/suml2."""
    n = Xsub.shape[0]
    X = Xsub.astype(np.float64)
    idx = np.argmax(np.abs(X), axis=1)
    maxSigned = X[np.arange(n), idx]
    allzero = np.abs(maxSigned) <= 1e-15
    maxSigned = np.where(allzero, 1.0, maxSigned)
    w = X * X
    best = np.full(n, -np.inf)
    Lbest = np.zeros((n, X.shape[1]), dtype=np.int64)
    scaleBest = np.zeros(n)
    passes = None  # pass inicial iscale=-nmax/max, luego is=-9..9 excluyendo 0
    for pass_i, (beta,) in enumerate([(float(nmax),)] + [((nmax + 0.1 * is_),) for is_ in list(range(-9, 0)) + list(range(1, 10))]):
        iscale = -beta / maxSigned
        l = np.clip(np.rint(X * iscale[:, None]), -nmax, nmax - 1)
        sumlx = (w * X * l).sum(axis=1)
        suml2 = (w * l * l).sum(axis=1)
        ok = ~(allzero) & (suml2 > 0) & (sumlx * sumlx > best * suml2)
        if ok.any():
            Lbest[ok] = l[ok].astype(np.int64)
            with np.errstate(invalid="ignore", divide="ignore"):
                scaleBest[ok] = sumlx[ok] / np.maximum(suml2[ok], 1e-15)
            best[ok] = scaleBest[ok] * sumlx[ok]
    scaleBest = np.where(allzero, 0.0, scaleBest)
    return scaleBest, Lbest


def _q6k_quant_blocks(X):
    """X (nb, 256) f32 -> bytes (nb, 210) uint8; puerto fiel de quantize_row_q6_K_ref."""
    nb = X.shape[0]
    X = X.astype(np.float32)
    sub = X.reshape(nb, 16, 16)
    scaleSub, Lsub = _make_qx_quants(sub.reshape(-1, 16))
    scaleSub = scaleSub.reshape(nb, 16)
    Lsub = Lsub.reshape(nb, 16, 16)
    absmax = np.abs(scaleSub).max(axis=1)
    isel = np.argmax(np.abs(scaleSub), axis=1)
    maxsc = scaleSub[np.arange(nb), isel]
    iscale = -128.0 / np.where(absmax > 1e-15, maxsc, 1.0)
    dfp = (1.0 / iscale).astype(np.float16).astype(np.float32)
    sc_int = np.clip(np.rint(iscale[:, None] * scaleSub), -128, 127).astype(np.int8)
    L = np.empty((nb, 256), dtype=np.int64)
    for j in range(16):
        d_j = dfp.astype(np.float64) * sc_int[:, j].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            lj = np.rint(X[:, j * 16:(j + 1) * 16].astype(np.float64) / np.where(np.abs(d_j) > 1e-9, d_j, 1.0)[:, None])
        L[:, j * 16:(j + 1) * 16] = np.clip(lj, -32, 31) + 32
    Lplane = L.reshape(nb, 8, 32)
    out = np.empty((nb, 210), dtype=np.uint8)
    zeros = np.zeros((nb, 210), dtype=np.uint8)
    allzero = absmax < 1e-15
    ql = np.zeros((nb, 128), dtype=np.uint8)
    qh = np.zeros((nb, 64), dtype=np.uint8)
    for half in range(2):
        base = half * 128
        l = np.arange(32)
        lo1 = Lplane[:, base // 32 + 0, l] & 0x0F
        lo3 = Lplane[:, base // 32 + 2, l] & 0x0F
        ql[:, half * 64 + l] = (lo1 | (lo3 << 4)).astype(np.uint8)
        lo2 = Lplane[:, base // 32 + 1, l] & 0x0F
        lo4 = Lplane[:, base // 32 + 3, l] & 0x0F
        ql[:, half * 64 + 32 + l] = (lo2 | (lo4 << 4)).astype(np.uint8)
        o1 = (Lplane[:, base // 32 + 0, l] >> 4) & 0x03
        o2 = (Lplane[:, base // 32 + 1, l] >> 4) & 0x03
        o3 = (Lplane[:, base // 32 + 2, l] >> 4) & 0x03
        o4 = (Lplane[:, base // 32 + 3, l] >> 4) & 0x03
        qh[:, half * 32 + l] = (o1 | (o2 << 2) | (o3 << 4) | (o4 << 6)).astype(np.uint8)
    out[:, :128] = ql
    out[:, 128:192] = qh
    out[:, 192:208] = sc_int.view(np.uint8)
    out[:, 208:210] = dfp.astype(np.float16).view(np.uint8).reshape(nb, 2)
    out[allzero] = zeros[allzero]
    return out


def _verify_q6k_roundtrip():
    rng = np.random.default_rng(7)
    X = (rng.standard_normal((40, 256)) * 0.01).astype(np.float32)
    X *= (10.0 ** rng.uniform(-6, 6, size=(40, 1)))  # rangos variados
    B = _q6k_quant_blocks(X)
    D = gguf.dequantize(B, gguf.GGMLQuantizationType.Q6_K)
    rel = np.mean(np.abs(D.reshape(40, 256) - X)) / np.mean(np.abs(X))
    return B, D, rel


# ---------------- seguridad P0 ----------------
FORBIDDEN_TENSORS = ("token_embd.weight", "output.weight", "output_norm.weight")
FORBIDDEN_PAT = ("norm.weight", ".norm", "lm_head", "token_types")


def _check_tensors(ops, tm):
    """PolÃ­tica de ediciÃ³n del MOTOR: rechaza tensores prohibidos y params fuera de rango."""
    for op in ops:
        nm = op.get("tensor", "")
        if nm in FORBIDDEN_TENSORS:
            sys.exit(f"POLICY: tensor PROHIBIDO {nm} (head/lm_head)")
        if any(p in nm for p in FORBIDDEN_PAT):
            sys.exit(f"POLICY: tensor con patron prohibido {nm} (normas/head)")
        if nm not in tm:
            sys.exit(f"POLICY: tensor inexistente en el modelo: {nm}")
        sc = op.get("scale")
        if sc is not None and not (0.25 <= float(sc) <= 3.0):
            sys.exit(f"POLICY: scale {sc} fuera de [0.25, 3.0]")
        tf = op.get("target_frac")
        if tf is not None and not (0.0 < float(tf) <= 0.05):
            sys.exit(f"POLICY: target_frac {tf} fuera de (0, 0.05]")
        al = op.get("alpha")
        if al is not None and abs(float(al)) > 8.0:
            sys.exit(f"POLICY: |alpha| {al} > 8")
        mx = op.get("mix")
        if mx is not None and not (0.0 <= float(mx) <= 1.0):
            sys.exit(f"POLICY: mix {mx} fuera de [0,1]")


def _sha256_full(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint(path):
    """sha256 de header (primer 4MB) + 1MB cola + tamano: huella barata del checkpoint origen."""
    import hashlib
    st = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
        if st > 4 * 1024 * 1024:
            f.seek(max(0, st - 1024 * 1024))
            h.update(f.read(1024 * 1024))
    return {"size": st, "sha": h.hexdigest()[:32]}


def _budget_check(kind, need=1):
    """Barrera OBLIGATORIA de presupuesto: bloquea ANTES de ejecutar (no solo registra)."""
    n = 0
    if PROME_JRNL.exists():
        for line in PROME_JRNL.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("kind") == kind:
                    n += 1
            except Exception:
                pass
    if n + need > BUDGET.get(kind, 0):
        sys.exit(f"BUDGET ANULADO: tipo '{kind}' {n}+{need} > limite {BUDGET[kind]}. Requiere reset manual del journal u aprobacion explicita.")
    return n + need


def cmd_sha(a):
    h = _sha256_full(a.gguf)
    _p(f"sha256 {os.path.basename(a.gguf)}: {h}")
    journal("read", "sha")


def cmd_edit(a):
    prog = json.loads(Path(a.prog).read_text(encoding="utf-8"))
    src = Path(a.gguf)
    out = Path(a.out or prog.get("out") or (DIR_SRV / (src.stem + f"__{Path(a.prog).stem}.gguf")))
    if out.exists():
        sys.exit(f"{out} ya existe (no sobreescribir)")
    rd = reader_of(src)
    _budget_check("write", 1)
    _check_tensors(prog["ops"], tensor_map(rd))
    journal("write", "reserva " + json.dumps({"out": str(out), "ops": len(prog["ops"])}, ensure_ascii=False))
    if "require_sha" in prog:  # identidad integ: SHA-256 completo del origen
        real = _sha256_full(src)
        if str(prog["require_sha"])[0:16] != real[0:16]:
            sys.exit(f"POLICY: require_sha no coincide ({real[:16]} != {prog['require_sha'][:16]})")
    _p(f"copiando {src} -> {out} ...")
    shutil.copy2(src, out)
    rd = reader_of(out)  # reader sobre la copia: offsets absolutos validos
    tm = tensor_map(rd)
    base_t = {}
    if "base" in prog:
        rb = reader_of(prog["base"])
        base_t = tensor_map(rb)
    rowsEdited = []
    for op in prog["ops"]:
        nm = op["tensor"]
        t = tm[nm]
        nrows = int(t.shape[1])
        ncols = int(t.shape[0])
        D = row_decode(t)
        if "base" in prog:
            B = row_decode(base_t[nm])
        elem_pb = 256 if t.tensor_type == gguf.GGMLQuantizationType.Q6_K else (32 if t.tensor_type == gguf.GGMLQuantizationType.Q8_0 else ne0)
        if op["type"] == "subtract-direction":
            if op.get("u_from") == "svd-drift":
                if not base_t:
                    sys.exit("subtract-direction svd-drift requiere prog.base")
                Mm = (D - B)
                u, sv, vt = np.linalg.svd(Mm, full_matrices=False)
                u = u[:, 0].astype(np.float64)
                u = u / (np.linalg.norm(u) + 1e-30)
                _p(f"  {nm}: sv1={sv[0]:.4g} niveles={sv[:4]}")
            elif isinstance(op.get("u_from"), str) and op["u_from"].startswith("npz:"):
                spec_u = op["u_from"][4:]
                zpath, key = spec_u.split("#", 1)
                z = np.load(zpath, allow_pickle=False)
                if key not in z.files:
                    sys.exit(f"POLICY: clave '{key}' no existe en {zpath}")
                u = np.asarray(z[key], dtype=np.float64)
                if u.ndim != 1 or u.size < 256 or not np.isfinite(u).all():
                    sys.exit(f"POLICY: u invalido (shape {u.shape}, finite={np.isfinite(u).all()})")
                u = u / (np.linalg.norm(u) + 1e-30)
                _p(f"  {nm}: u desde npz {zpath}#{key} len={u.size} norm={np.linalg.norm(u):.6f}")
            elif op.get("u") is not None:
                u = np.asarray(op["u"], dtype=np.float64)
            axis = op.get("axis", "rows")
            if axis == "rows":
                v = u @ D  # len ncols
            else:
                v = D @ u  # len nrows
                u = u.reshape(-1)
            if op.get("cols_mask") is not None:
                cm = np.asarray(op["cols_mask"], dtype=int)
                vm = np.zeros_like(v)
                vm[cm] = v[cm]
                v = vm
                _p(f"  {nm}: cols_mask n={cm.size}")
            elif op.get("mask") == "top-k":
                k = int(op.get("k", 64))
                idx = np.argsort(np.abs(v))[::-1][:k]
                vm = np.zeros_like(v)
                vm[idx] = v[idx]
                v = vm
                _p(f"  {nm}: mask top-{k} idx={sorted(idx.tolist())}")
            if op.get("target_frac"):
                tf = float(op["target_frac"])
                al = tf * (np.linalg.norm(D.astype(np.float64)) / (np.linalg.norm(v) + 1e-30))
            else:
                al = float(op.get("alpha", 0.0))
            _p(f"  {nm}: alpha={al:.4g}")
            tol = float(op.get("touch_tol", 1e-3))
            rowmap = {}
            if axis == "rows":
                Dn = D.astype(np.float64) - al * np.outer(u, v)
                m_ur = np.abs(u) > tol * np.abs(u).max()
                m_vc = np.abs(v) > tol * np.abs(v).max()
                nblk = (int(t.shape[0]) // elem_pb)
                blk_cols = sorted(set((np.where(m_vc)[0] // elem_pb).tolist()))
                for r in np.where(m_ur)[0].tolist():
                    rowmap[int(r)] = blk_cols
            else:
                Dn = D.astype(np.float64) - al * np.outer(v, u)
                m_vc = np.abs(u) > tol * np.abs(u).max()
                m_ur = np.abs(v) > tol * np.abs(v).max()
                blk_cols = sorted(set((np.where(m_vc)[0] // elem_pb).tolist()))
                for r in np.where(m_ur)[0].tolist():
                    rowmap[int(r)] = blk_cols
            op["_rowmap"] = rowmap
            norm_d = float(np.linalg.norm(Dn - D))
            norm_d0 = float(np.linalg.norm(D))
            D = Dn.astype(np.float32)
            rowsEdited.append([nm, "subtract-direction", round(norm_d / (norm_d0 + 1e-30), 4)])
        elif op["type"] == "blend-rows-toward":
            idx = np.asarray(op["rows"], dtype=int)
            D[idx] = (1 - op["mix"]) * D[idx] + op["mix"] * B[idx]
            rowsEdited.append([nm, "blend-rows", int(idx.size)])
        elif op["type"] == "scale-rows":
            idx = np.asarray(op["rows"], dtype=int)
            D[idx] *= float(op["scale"])
            rowsEdited.append([nm, "scale-rows", int(idx.size)])
        elif op["type"] == "scale-cols":
            idx = np.asarray(op["cols"], dtype=int)
            D[:, idx] *= float(op["scale"])
            rowsEdited.append([nm, "scale-cols", int(idx.size)])
        else:
            sys.exit(f"op no reconocida: {op['type']}")
        # reescribir en el archivo
        if base_dtype(t):
            newbytes = np.ascontiguousarray(D, dtype=D.dtype).tobytes()
        else:
            if "_rowmap" in op:
                rowmap = op.pop("_rowmap")
            elif op["type"] == "blend-rows-toward" or op["type"] == "scale-rows":
                rowmap = {int(r): list(range(int(t.shape[0]) // elem_pb)) for r in np.asarray(op["rows"], dtype=int)}
            else:  # scale-cols
                blk_cols = sorted(set((np.asarray(op["cols"], dtype=int) // elem_pb).tolist()))
                rowmap = {r: blk_cols for r in range(int(t.shape[1]))}
            newbytes = _requant_blocks_bytes(t, rowmap, D)
        with open(out, "r+b") as fh:
            fh.seek(t.data_offset)
            fh.write(newbytes)
        _p(f"  reescrito {nm}: {newbytes.__len__()} bytes")
    journal("edit", json.dumps({"out": str(out), "ops_count": len(prog["ops"])}, ensure_ascii=False))
    Path(str(out) + ".manifest.json").write_text(json.dumps({
        "source": str(src), "source_fp": _fingerprint(src), "program": a.prog,
        "ops": [{k: v for k, v in op.items() if k in ("type", "tensor", "axis", "mask", "k", "target_frac", "alpha", "mix", "scale")} for op in prog["ops"]],
        "edited_weights_note": "ver wsp diff -frac 0.002",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    _p("EDIT AVISO:", out)
    _p("RESUMEN:", json.dumps(rowsEdited, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Áreas Semánticas y Representaciones Funcionales (Dreaming Technique 16 + WSP)
# ---------------------------------------------------------------------------
SEMANTIC_AREAS = {
    "math_logic": {
        "label": "Math & Logic",
        "color": "#3498db",
        "seeds": ["logic", "math", "proof", "solve", "equation", "theorem", "calculate", "integer", "formula", "axiom", "lemma", "derivative", "prime"],
    },
    "agent_action": {
        "label": "Agent & Actions",
        "color": "#e67e22",
        "seeds": ["tool", "command", "run", "execute", "terminal", "check", "exit", "error", "file", "script", "bash", "status", "output"],
    },
    "code_software": {
        "label": "Code & Software",
        "color": "#16a085",
        "seeds": ["python", "code", "function", "import", "class", "def", "return", "data", "software", "model", "variable", "string", "array"],
    },
    "abstract_ideas": {
        "label": "Abstract Ideas",
        "color": "#2980b9",
        "seeds": ["idea", "truth", "concept", "meaning", "reason", "justice", "theory", "knowledge", "principle", "freedom", "structure"],
    },
    "mind_cognition": {
        "label": "Mind & Cognition",
        "color": "#1abc9c",
        "seeds": ["mind", "think", "know", "memory", "dream", "learn", "brain", "conscious", "aware", "understand", "reasoning"],
    },
    "time_sequence": {
        "label": "Time & Sequence",
        "color": "#f39c12",
        "seeds": ["time", "past", "future", "moment", "now", "step", "next", "before", "after", "history", "loop", "repeat", "stop"],
    },
    "emotion_pos": {
        "label": "Positive Emotion",
        "color": "#2ecc71",
        "seeds": ["happy", "joy", "love", "peace", "hope", "smile", "glad", "kind", "gentle", "warm"],
    },
    "emotion_neg": {
        "label": "Negative Emotion",
        "color": "#e74c3c",
        "seeds": ["sad", "hate", "fear", "anger", "pain", "cry", "rage", "grief", "lonely", "shame"],
    },
    "physical_world": {
        "label": "Physical World",
        "color": "#95a5a6",
        "seeds": ["body", "rock", "water", "fire", "earth", "stone", "blood", "metal", "wood", "space"],
    },
    "social_power": {
        "label": "Social & Power",
        "color": "#8e44ad",
        "seeds": ["power", "war", "nation", "people", "law", "money", "work", "leader", "rule", "state"],
    },
}

_CENTROID_CACHE = {}


def get_model_arch_info(r):
    """Extrae metadatos arquitectónicos de GGUF (head counts, recurrent SSM flags, etc.)."""
    arch = "qwen35"
    if "general.architecture" in r.fields:
        raw_arch = r.fields["general.architecture"].contents()
        if isinstance(raw_arch, list) and raw_arch:
            raw_arch = raw_arch[0]
        if isinstance(raw_arch, bytes):
            arch = raw_arch.decode("utf-8", "replace")
        elif isinstance(raw_arch, str):
            arch = raw_arch

    def get_val(suffix, default=None):
        for key in [f"{arch}.{suffix}", suffix]:
            if key in r.fields:
                val = r.fields[key].contents()
                if isinstance(val, list) and len(val) == 1:
                    val = val[0]
                return val
        return default

    recurrent = get_val("attention.recurrent_layers", [])
    if isinstance(recurrent, list):
        recurrent_flags = [bool(x) for x in recurrent]
    else:
        recurrent_flags = []

    return {
        "arch": arch,
        "embedding_length": int(get_val("embedding_length", 4096)),
        "block_count": int(get_val("block_count", 32)),
        "head_count": int(get_val("attention.head_count", 16)),
        "head_count_kv": int(get_val("attention.head_count_kv", 4)),
        "key_length": int(get_val("attention.key_length", 256)),
        "value_length": int(get_val("attention.value_length", 256)),
        "feed_forward_length": int(get_val("feed_forward_length", 12288)),
        "recurrent_layers": recurrent_flags,
    }


def _resolve_token(vocab_map, word):
    """Resuelve un término a su token ID considerando BPE de Qwen/Llama."""
    prefixes = [chr(0x0120) + word, word, " " + word, chr(0x2581) + word,
                chr(0x0120) + word.lower(), word.lower(), " " + word.lower(),
                chr(0x0120) + word.capitalize(), word.capitalize()]
    for p in prefixes:
        if p in vocab_map:
            return vocab_map[p], p
    return None, None


def get_semantic_centroids(gguf_path, r=None):
    """Calcula o carga del caché los centroides diferenciales de las áreas semánticas."""
    path_str = str(gguf_path)
    if path_str in _CENTROID_CACHE:
        return _CENTROID_CACHE[path_str]

    stem = Path(path_str).stem
    cache_path = DIR_DATA / f"{stem}.semantic_centroids.npz"
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            centroids = {}
            for k in data.files:
                centroids[k] = data[k]
            _CENTROID_CACHE[path_str] = centroids
            return centroids
        except Exception:
            pass

    if r is None:
        r = reader_of(path_str)

    tok_field = r.fields.get("tokenizer.ggml.tokens")
    if not tok_field:
        return {}

    vocab_list = tok_field.contents()
    vocab_map = {tok: idx for idx, tok in enumerate(vocab_list)}
    t_embd = tensor_map(r).get("token_embd.weight")
    if t_embd is None:
        return {}

    raw_centroids = {}
    seed_vectors = {}
    for area, meta in SEMANTIC_AREAS.items():
        vecs = []
        for s in meta["seeds"]:
            idx, tok_str = _resolve_token(vocab_map, s)
            if idx is not None:
                raw = t_embd.data[idx:idx + 1]
                v = gguf.dequantize(raw, t_embd.tensor_type).flatten().astype(np.float32)
                v /= (np.linalg.norm(v) + 1e-12)
                vecs.append(v)
                seed_vectors[f"seed_{s}"] = v
        if vecs:
            c = np.mean(vecs, axis=0)
            c /= (np.linalg.norm(c) + 1e-12)
            raw_centroids[area] = c

    if not raw_centroids:
        return {}

    mean_bg = np.mean(list(raw_centroids.values()), axis=0)
    diff_centroids = {}
    for area, c in raw_centroids.items():
        dc = c - mean_bg
        dc /= (np.linalg.norm(dc) + 1e-12)
        diff_centroids[area] = dc

    save_dict = dict(diff_centroids)
    save_dict.update(seed_vectors)
    try:
        np.savez_compressed(cache_path, **save_dict)
    except Exception:
        pass

    _CENTROID_CACHE[path_str] = save_dict
    return save_dict


def attribute_semantic_area(vec, centroids_dict, top_k=3):
    """Calcula la proyección semántica de un vector residual frente a las áreas semánticas."""
    if not centroids_dict:
        return None
    v = vec.flatten().astype(np.float32)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return None
    v = v / v_norm

    area_scores = []
    seed_scores = []
    for k, c in centroids_dict.items():
        if k.startswith("seed_"):
            seed_name = k[5:]
            score = float(np.dot(v, c))
            seed_scores.append((seed_name, score))
        elif k in SEMANTIC_AREAS:
            label = SEMANTIC_AREAS[k]["label"]
            score = float(np.dot(v, c))
            area_scores.append((k, label, score))

    area_scores.sort(key=lambda x: -x[2])
    seed_scores.sort(key=lambda x: -x[1])

    dominant = area_scores[0] if area_scores else None
    return {
        "dominant": {"area": dominant[0], "label": dominant[1], "score": round(dominant[2], 4)} if dominant else None,
        "all_areas": [{"area": a, "label": l, "score": round(s, 4)} for a, l, s in area_scores],
        "top_areas": [{"area": a, "label": l, "score": round(s, 4)} for a, l, s in area_scores[:top_k]],
        "ranked": [{"area": a, "label": l, "score": round(s, 4)} for a, l, s in area_scores[:top_k]],
        "top_seeds": [{"seed": s, "score": round(sc, 4)} for s, sc in seed_scores[:5]],
    }


def describe_tensor_role(tensor_name, row=None, col=None, arch_info=None, vocab=None):
    """Devuelve descripción anatómica y arquitectónica de qué controla el tensor y sus coordenadas."""
    arch_info = arch_info or {}
    key_len = arch_info.get("key_length", 256)
    val_len = arch_info.get("value_length", 256)
    recurrent_flags = arch_info.get("recurrent_layers", [])

    m = re.match(r"blk\.(\d+)\.(.*)", tensor_name)
    layer = int(m.group(1)) if m else None
    sub = m.group(2) if m else tensor_name
    is_recurrent = (layer is not None and layer < len(recurrent_flags) and recurrent_flags[layer])
    layer_type = "ssm_recurrent" if is_recurrent else ("full_attention" if layer is not None else "embedding/head")

    meta = {
        "layer": layer,
        "layer_type": layer_type,
        "component": sub,
        "role": "",
        "head": None,
        "head_dim": None,
        "neuron": None,
        "residual_dim": None,
    }

    if sub in ("attn_output.weight", "ssm_out.weight"):
        meta["component"] = "ssm_output_proj" if is_recurrent else "attention_output_proj"
        if col is not None:
            meta["head"] = col // key_len if not is_recurrent else None
            meta["head_dim"] = col % key_len if not is_recurrent else col
        if row is not None:
            meta["residual_dim"] = row
        if row is not None and col is not None:
            if is_recurrent:
                meta["role"] = f"SSM Layer {layer}: canal {col} -> escribe en residual dim {row}"
            else:
                meta["role"] = f"Attention Layer {layer}: head #{meta['head']} (dim {meta['head_dim']}) -> escribe en residual dim {row}"
        else:
            meta["role"] = f"Proyección de salida hacia el residual stream ({layer_type})"

    elif sub in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
        k_or_v = "KV" if "k" in sub or "v" in sub else "Q"
        l_dim = val_len if "v" in sub else key_len
        meta["component"] = f"attention_{k_or_v.lower()}_proj"
        if row is not None:
            meta["head"] = row // l_dim
            meta["head_dim"] = row % l_dim
        if col is not None:
            meta["residual_dim"] = col
        if row is not None and col is not None:
            meta["role"] = f"Layer {layer}: lee residual dim {col} -> {k_or_v}-head #{meta['head']} (dim {meta['head_dim']})"
        else:
            meta["role"] = f"Detector/lector de atención ({k_or_v}) desde el residual stream"

    elif sub == "ffn_down.weight":
        meta["component"] = "ffn_down_proj"
        if col is not None:
            meta["neuron"] = col
        if row is not None:
            meta["residual_dim"] = row
        if row is not None and col is not None:
            meta["role"] = f"Layer {layer} FFN Down: neurona intermedia #{col} -> escribe en residual dim {row}"
        else:
            meta["role"] = f"FFN síntesis hacia el residual stream"

    elif sub in ("ffn_up.weight", "ffn_gate.weight"):
        meta["component"] = sub.replace(".weight", "")
        if row is not None:
            meta["neuron"] = row
        if col is not None:
            meta["residual_dim"] = col
        if row is not None and col is not None:
            meta["role"] = f"Layer {layer} FFN Detector: lee residual dim {col} -> activa neurona intermedia #{row}"
        else:
            meta["role"] = f"FFN detector desde el residual stream"

    elif sub == "ssm_beta.weight":
        meta["component"] = "ssm_beta_gate"
        if row is not None: meta["residual_dim"] = row
        if col is not None: meta["head"] = col
        if row is not None and col is not None:
            meta["role"] = f"SSM Layer {layer} Beta Gate (retención/olvido de contexto): lee residual dim {row} -> modula grupo SSM #{col}"
        else:
            meta["role"] = f"SSM Beta Gate de retención/olvido"

    elif sub == "ssm_alpha.weight":
        meta["component"] = "ssm_alpha_gate"
        if row is not None: meta["residual_dim"] = row
        if col is not None: meta["head"] = col
        if row is not None and col is not None:
            meta["role"] = f"SSM Layer {layer} Alpha Gate (incorporación de contexto): lee residual dim {row} -> modula grupo SSM #{col}"
        else:
            meta["role"] = f"SSM Alpha Gate de entrada"

    elif sub == "attn_qkv.weight":
        meta["component"] = "ssm_qkv_proj"
        if row is not None: meta["head"] = row // key_len
        if col is not None: meta["residual_dim"] = col
        if row is not None and col is not None:
            meta["role"] = f"SSM Layer {layer} QKV: lee residual dim {col} -> feature #{row}"
        else:
            meta["role"] = f"SSM proyección de entrada QKV"

    elif tensor_name == "output.weight":
        meta["component"] = "lm_head_unembedding"
        if row is not None:
            tok_str = vocab[row] if vocab and row < len(vocab) else f"token_{row}"
            meta["role"] = f"LM Head: proyección al logit del token #{row} ('{tok_str}')"
        if col is not None:
            meta["residual_dim"] = col

    elif tensor_name == "token_embd.weight":
        meta["component"] = "token_embedding"
        if row is not None:
            tok_str = vocab[row] if vocab and row < len(vocab) else f"token_{row}"
            meta["role"] = f"Embedding de entrada para token #{row} ('{tok_str}')"

    else:
        meta["role"] = f"Tensor {tensor_name}"

    return meta


# ---------------------------------------------------------------------------
# peek() / poke() — lectura y edición individual de pesos
# ---------------------------------------------------------------------------
def _row_norm(D, r):
    return float(np.linalg.norm(D[r]))


def _col_norm(D, c):
    return float(np.linalg.norm(D[:, c]))


def _rank_by_norm(D, idx, axis=0):
    """rank (0 = más fuerte) de idx a lo largo de axis (0=filas, 1=columnas)."""
    vals = np.linalg.norm(D, axis=1 if axis == 0 else 0)
    pos = np.argsort(-vals, kind="stable")
    return int(np.flatnonzero(pos == idx)[0])


def peek(gguf_path, tensor_name, row=None, col=None, k=8, compute_semantic=True):
    """peek: qué controla un peso o tensor.

    sin row/col -> anatomía del tensor, normas y áreas semánticas dominantes.
    con row/col -> el peso concreto: valor, rol arquitectónico, neurona y área semántica.
    """
    r = reader_of(gguf_path)
    t = tensor_map(r)[tensor_name]
    D = row_decode(t)
    n_out, n_in = D.shape
    arch_info = get_model_arch_info(r)

    tok_field = r.fields.get("tokenizer.ggml.tokens")
    vocab = tok_field.contents() if tok_field else None

    arch_meta = describe_tensor_role(tensor_name, row, col, arch_info, vocab)
    centroids = get_semantic_centroids(gguf_path, r) if compute_semantic else {}

    out = {
        "tensor": tensor_name,
        "shape": [n_out, n_in],
        "dtype": base_dtype(t),
        "architecture": arch_meta,
        "value": None, "row": None, "col": None,
    }

    if row is not None and col is not None:
        row, col = int(row), int(col)
        v = float(D[row, col])
        rnorm = _row_norm(D, row)
        cnorm = _col_norm(D, col)
        out.update({
            "value": v,
            "sign": 1 if v >= 0 else -1,
            "abs": abs(v),
            "row": row, "col": col,
            "row_norm": rnorm, "col_norm": cnorm,
            "row_rank": _rank_by_norm(D, row, axis=0),
            "col_rank": _rank_by_norm(D, col, axis=1),
            "frac_row": abs(v) / (rnorm + 1e-30),
            "frac_col": abs(v) / (cnorm + 1e-30),
        })

        primary_vec = None
        if tensor_name.endswith("attn_output.weight") or tensor_name.endswith("ffn_down.weight") or tensor_name.endswith("ssm_out.weight"):
            if n_out == arch_info["embedding_length"]:
                primary_vec = D[:, col]
        elif tensor_name.endswith("attn_q.weight") or tensor_name.endswith("attn_k.weight") or tensor_name.endswith("attn_v.weight") or tensor_name.endswith("ffn_up.weight") or tensor_name.endswith("ffn_gate.weight"):
            if n_in == arch_info["embedding_length"]:
                primary_vec = D[row, :]
        elif tensor_name == "output.weight":
            primary_vec = D[row, :]
        elif n_out == arch_info["embedding_length"]:
            primary_vec = D[:, col]
        elif n_in == arch_info["embedding_length"]:
            primary_vec = D[row, :]

        if primary_vec is not None and centroids:
            out["semantic_areas"] = attribute_semantic_area(primary_vec, centroids)
    else:
        row_norms = np.linalg.norm(D, axis=1)
        col_norms = np.linalg.norm(D, axis=0)
        top_r = [int(i) for i in np.argsort(-row_norms, kind="stable")[:k]]
        top_c = [int(j) for j in np.argsort(-col_norms, kind="stable")[:k]]
        nnz = int(np.count_nonzero(np.abs(D) > 1e-6))

        top_rows_meta = []
        for i in top_r:
            item = {"row": i, "norm": float(row_norms[i])}
            if centroids and n_in == arch_info["embedding_length"]:
                sem = attribute_semantic_area(D[i, :], centroids, top_k=1)
                if sem and sem["dominant"]:
                    item["dominant_area"] = sem["dominant"]
            top_rows_meta.append(item)

        top_cols_meta = []
        for j in top_c:
            item = {"col": j, "norm": float(col_norms[j])}
            if centroids and n_out == arch_info["embedding_length"]:
                sem = attribute_semantic_area(D[:, j], centroids, top_k=1)
                if sem and sem["dominant"]:
                    item["dominant_area"] = sem["dominant"]
            top_cols_meta.append(item)

        # Perfil semántico acumulado del tensor
        dominant_counts = {}
        all_doms = [m.get("dominant_area", {}).get("area") for m in (top_rows_meta if n_in == arch_info["embedding_length"] else top_cols_meta) if m.get("dominant_area")]
        for a in all_doms:
            dominant_counts[a] = dominant_counts.get(a, 0) + 1
        total_dom = sum(dominant_counts.values()) or 1
        profile = {
            "all_areas": [{"area": a, "label": SEMANTIC_AREAS[a]["label"], "fraction": round(c / total_dom, 3)} for a, c in sorted(dominant_counts.items(), key=lambda kv: -kv[1])],
            "top_areas": [{"area": a, "label": SEMANTIC_AREAS[a]["label"], "fraction": round(c / total_dom, 3)} for a, c in sorted(dominant_counts.items(), key=lambda kv: -kv[1])[:3]]
        }
        out.update({
            "nonzero": nnz,
            "row_norms": {"min": float(row_norms.min()), "median": float(np.median(row_norms)),
                          "max": float(row_norms.max()), "top": top_r},
            "col_norms": {"min": float(col_norms.min()), "median": float(np.median(col_norms)),
                          "max": float(col_norms.max()), "top": top_c},
            "top_rows": top_rows_meta,
            "top_cols": top_cols_meta,
            "semantic_profile": profile,
        })
    return out


def poke(gguf_path, tensor_name, row, col, value, out=None):
    """poke: fijar un peso individual en el modelo.

    dequantiza el tensor, modifica el elemento en (row, col),
    requantiza SOLO el bloque que contiene la columna (cero ruido colateral),
    y escribe de vuelta en el archivo sin cargar los 8 GB en RAM.
    """
    src = str(gguf_path)
    r = reader_of(src)
    tm = tensor_map(r)
    if tensor_name not in tm:
        raise KeyError(f"Tensor {tensor_name} no encontrado en {src}")
    t = tm[tensor_name]

    ne0 = int(t.shape[0])
    ne1 = int(t.shape[1])
    row, col = int(row), int(col)
    if not (0 <= row < ne1 and 0 <= col < ne0):
        raise ValueError(f"Coordenadas fuera de rango: ({row}, {col}) para matriz {ne1}x{ne0}")

    elem_pb = 256 if t.tensor_type == gguf.GGMLQuantizationType.Q6_K else 32
    blk_idx = col // elem_pb

    D = row_decode(t)
    old_val = float(D[row, col])
    D[row, col] = float(value)

    newbytes = _requant_blocks_bytes(t, {row: [blk_idx]}, D)

    out_path = out or f"{Path(src).stem}.poke-{tensor_name}-{row}-{col}.gguf"
    out_path = str(out_path)
    if out_path != src:
        _p(f"  clonando modelo a {out_path}...")
        shutil.copyfile(src, out_path)

    with open(out_path, "r+b") as fh:
        fh.seek(t.data_offset)
        fh.write(newbytes)

    journal("poke", json.dumps({
        "tensor": tensor_name, "row": row, "col": col,
        "old_val": old_val, "new_val": float(value), "out": out_path
    }, ensure_ascii=False))

    _p(f"POKE completed at {out_path}:")
    _p(f"  {tensor_name}[{row}, {col}]: {old_val:+.6f} -> {float(value):+.6f} (bloque {blk_idx})")
    return {
        "out": out_path, "tensor": tensor_name, "row": row, "col": col,
        "old_val": old_val, "new_val": float(value), "block": blk_idx
    }


def cmd_peek(a):
    res = peek(a.gguf, a.tensor, row=a.row, col=a.col, k=a.k)
    if getattr(a, "json", False):
        _p(json.dumps(res, indent=2, ensure_ascii=False))
        return

    arch = res["architecture"]
    _p(f"=== WSP PEEK: {res['tensor']} ===")
    _p(f"  Shape: {res['shape']}  Type: {res['dtype']}")
    _p(f"  Role: {arch.get('role', '')}")
    if arch.get("head") is not None:
        _p(f"  Head: #{arch['head']} (dim {arch.get('head_dim')})  Capa: {arch.get('layer')} ({arch.get('layer_type')})")

    if res.get("value") is not None:
        _p(f"\n  Coordinates: row={res['row']} col={res['col']}")
        _p(f"  Valor: {res['value']:+.6f} (|v|={res['abs']:.6f})")
        _p(f"  Row norm: {res['row_norm']:.4f} (rank #{res['row_rank']}, {res['frac_row']:.2%})")
        _p(f"  Col norm:  {res['col_norm']:.4f} (rank #{res['col_rank']}, {res['frac_col']:.2%})")

        sem = res.get("semantic_areas")
        if sem and sem.get("dominant"):
            dom = sem["dominant"]
            _p(f"\n  DOMINANT SEMANTIC AREA: {dom['label']} ({dom['area']})  score={dom['score']:+.4f}")
            top_list = sem.get("top_areas") or sem.get("ranked", [])
            _p(f"\n  TOP SEMANTIC AREAS (TOP {len(top_list)}):")
            for item in top_list:
                _p(f"    * {item['label']:<30} ({item['area']:<15}) score={item['score']:+.4f}")
            all_list = sem.get("all_areas", [])
            if all_list:
                _p(f"\n  ALL SEMANTIC AREAS ({len(all_list)} áreas):")
                for item in all_list:
                    _p(f"    - {item['label']:<30} ({item['area']:<15}) score={item['score']:+.4f}")
            _p("\n  Top affine seeds:")
            for s in sem.get("top_seeds", []):
                _p(f"    - {s['seed']:<15} score={s['score']:+.4f}")
    else:
        _p(f"  Elementos no nulos: {res.get('nonzero'):,}")
        _p(f"  Top filas: {res.get('top_rows')}")
        _p(f"  Top cols:  {res.get('top_cols')}")


def cmd_poke(a):
    poke(a.gguf, a.tensor, row=a.row, col=a.col, value=a.val, out=a.out)


def cmd_semantic_map(a):
    _p("=== MAPA DE ÁREAS SEMÁNTICAS (WSP-Lang) ===")
    centroids = get_semantic_centroids(a.gguf)
    _p(f"Modelo: {a.gguf}")
    _p(f"Áreas semánticas activas: {len(SEMANTIC_AREAS)}")
    for k, meta in SEMANTIC_AREAS.items():
        _p(f"\n  [{k}] {meta['label']} (color: {meta['color']})")
        _p(f"    Semillas ({len(meta['seeds'])}): {', '.join(meta['seeds'])}")
        if k in centroids:
            c = centroids[k]
            _p(f"    Centroide diferencial calculado (dim {len(c)}, norma={np.linalg.norm(c):.4f})")


def cmd_diff(a):
    ta = {t.name: t for t in reader_of(a.a).tensors}
    tb = {t.name: t for t in reader_of(a.b).tensors}
    import fnmatch
    tot_n = 0
    tot_chg = 0
    for nm in sorted(tb):
        if not fnmatch.fnmatch(nm, a.tensor) or nm not in ta:
            continue
        D1 = row_decode(ta[nm])
        D2 = row_decode(tb[nm])
        chg = (np.abs(D1 - D2) > a.frac * (np.abs(D1) + 1e-30)).sum()
        tot_n += D2.size
        tot_chg += chg
        _p(f"{nm:<40} chg={chg:>10} de {D2.size:>10} ({chg / max(D2.size, 1):.4%})")
    if tot_n:
        _p(f"TOTAL: {tot_chg:,} pesos cambiados de {tot_n:,} ({tot_chg / tot_n:.5%})")


def cmd_vs(a):
    ra = json.loads((DIR_RIG / f"{a.runA}.json").read_text(encoding="utf-8"))
    rb = json.loads((DIR_RIG / f"{a.runB}.json").read_text(encoding="utf-8"))
    for nm, rs in ((a.runA, ra), (a.runB, rb)):
        los_ok = [r for r in rs if "loop" in r]
        _p(nm, f"loop {sum(r['loop'] for r in los_ok)}/{len(los_ok)}")
    for i in range(max(len(ra), len(rb))):
        if i < len(ra) and i < len(rb):
            _p(f"  [{i}] A={ra[i]['loop']}@{ra[i].get('turns')} B={rb[i]['loop']}@{rb[i].get('turns')}")


CMD_FN = {
    "sha": cmd_sha,
    "info": cmd_info, "tensors": cmd_tensors, "stats": cmd_stats, "rows": cmd_rows,
    "delta": cmd_delta, "markers": cmd_markers, "edit": cmd_edit, "diff": cmd_diff,
    "vs": cmd_vs, "budget": cmd_budget,
    "peek": cmd_peek, "poke": cmd_poke, "semantic-map": cmd_semantic_map,
}


def main():
    args = _make_parser().parse_args()
    if "eval" not in args.cmd:
        fn = CMD_FN.get(args.cmd)
        if fn:
            fn(args)
            journal(JRN_KIND.get(args.cmd, "read"), args.cmd)
            return
    if args.cmd == "serve":
        cmd_serve(args)
        return
    if args.cmd == "eval":
        cmd_eval(args)
        return
    sys.exit(f"cmd desconocido {args.cmd}")


def cmd_serve(args):
    fName = Path(args.gguf)
    port = args.port
    alias = args.alias or fName.stem[:18]
    if args.stop:
        pid = (DIR_LOG / f"llama-{port}.pid")
        if pid.exists():
            os.kill(int(pid.read_text()), 15)
            pid.unlink()
            _p(f"stop port {port}")
        return
    log = DIR_LOG / f"llama-{port}.log"
    cmd = [str(LLAMA_BIN), "-m", str(fName), "--port", str(port), "--host", "127.0.0.1",
           "-a", alias, "-c", str(args.ctx), "-ngl", str(args.ngl), "--jinja"]
    _p("servir:", " ".join(cmd))
    det_err = subprocess.Popen(cmd, stdout=open(log, "ab"), stderr=subprocess.STDOUT,
                               creationflags=0x00000008)
    pid_fh = DIR_LOG / f"llama-{port}.pid"
    pid_fh.write_text(str(det_err.pid), encoding="utf-8")
    for _ in range(60):
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as pkg:
                if pkg.status == 200:
                    _p(f"UP port {port} pid {det_err.pid} log {log}")
                    return
        except Exception:
            pass
    raise RuntimeError(f"llama-server no arranca port {port} log {log}")


def base_ctx(c):
    return min(c, 32768)


def cmd_eval(args):
    _budget_check("fwd", 1)
    import subprocess
    cmd = [
        sys.executable,
        str(ROOT / "rig" / "agent_loop_probe.py"),
        "--endpoint", f"http://127.0.0.1:{args.port}/v1/chat/completions",
        "--alias", args.alias,
        "--n", str(args.n),
        "--offset", str(args.offset)
    ]
    subprocess.run(cmd, check=True)
    journal("fwd", json.dumps({"eval": args.alias, "port": args.port, "n": args.n}, ensure_ascii=False))


if __name__ == "__main__":
    main()

