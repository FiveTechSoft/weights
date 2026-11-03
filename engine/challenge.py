"""WSP Challenge Engine v0.2 — currículo WSP-40, jueces mecánicos, expedientes.

Reglas: sin self-grading (el motor calcula referencias al JUZGAR, jamás antes),
holdouts ciegs (sellar), reproducibilidad con seeds, no-degradación obligatoria.
CLI JSON-print:
  reto [id]            -> brief del siguiente/pedido reto (nunca golds)
  submit <id> <f.json> -> juez mecánico + expediente + registro
  expediente <id>
  feedback <alias>
  sellar <alias> <set> -p port
  verdicto <cand> <keep|discard> <notas>
  historial [n]
  estado
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import gguf

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = Path(__file__).resolve().parent.parent
RETOS = ROOT / "challenges" if (ROOT / "challenges").exists() else ROOT / "retos"
CATALOGO = RETOS / "wsp40.json"
EXP = RETOS / "records" if (RETOS / "records").exists() else RETOS / "expedientes"
SELLADOS = RETOS / "sealed" if (RETOS / "sealed").exists() else RETOS / "sellados"
HIST = RETOS / "historial.jsonl"
RIG = ROOT / "rig_runs"
for d in (EXP, SELLADOS):
    d.mkdir(parents=True, exist_ok=True)

def _buscar_gguf(*cand, env=None):
    """Busca el GGUF en el orden: env var -> rutas candidatas -> default (Windows).
    Devuelve la primera que exista; si ninguna, la env var o el default."""
    if env and os.environ.get(env):
        p = Path(os.environ[env])
        if p.exists():
            return str(p)
    for c in cand:
        if c.exists():
            return str(c)
    return os.environ.get(env) if env else None


_MODELO_CANDS = [
    ROOT / "hf_cache" / "mimo-gguf" / "MiMo-V2.6-Distill-Qwen-9B-Q6_K.gguf",
    ROOT / "servidos" / "mimo-orig.gguf",
]
_BASE_CANDS = [
    ROOT / "hf_cache" / "qwen35base" / "Qwen3.5_9B_Base.Q6_K.gguf",
]
MODELO = _buscar_gguf(*_MODELO_CANDS, env="WSP_MODELO_GGUF") or r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf"
BASE = _buscar_gguf(*_BASE_CANDS, env="WSP_BASE_GGUF") or r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf"
_CREATION_FLAGS = {"creationflags": 0x08000000} if platform.system() == "Windows" else {}


def _log(event, **kw):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **kw}
    with HIST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def catalogo():
    return json.loads(CATALOGO.read_text(encoding="utf-8"))


def estado():
    cat = catalogo()
    total = len(cat["retos"])
    aprobados = sorted([p.stem for p in EXP.glob("*.json") if json.loads(p.read_text(encoding="utf-8")).get("estado") == "aprobado"])
    return {"modelo": "b", "aprobados": aprobados, "n_aprobados": len(aprobados),
            "n_total": total, "verdictos": _verdict_stats()}


def _verdict_stats():
    if not HIST.exists():
        return {"keep": 0, "discard": 0}
    rows = [json.loads(l) for l in HIST.read_text(encoding="utf-8").splitlines()]
    return {v: sum(1 for r in rows if r.get("event") == "verdicto" and r.get("vered") == v) for v in ("keep", "discard")}


def reto(id_=None):
    cat = catalogo()
    ret = cat["retos"]
    if id_:
        r = next((x for x in ret if x["id"] == id_.zfill(2)), None)
        if r is None:
            return {"error": "reto no existe"}
        out = [r]
    else:
        ok = {p.stem for p in EXP.glob("*.json") if json.loads(p.read_text(encoding="utf-8")).get("estado") == "aprobado"}
        out = [x for x in ret if x["id"] not in ok][:1]
    for r in out:
        exp_path = EXP / f"{r['id']}.json"
        if exp_path.exists():
            e = json.loads(exp_path.read_text(encoding="utf-8"))
            r["expediente"] = {k: e[k] for k in ("estado", "score", "intentos", "hipotesis", "intervencion", "resultado", "principio") if k in e}
    return {"reto": out[0] if len(out) == 1 else out, "reglas": cat["reglas_globales"]}


def expediente(id_):
    p = EXP / f"{id_.zfill(2)}.json"
    if not p.exists():
        return {"expediente": None}
    return {"expediente": json.loads(p.read_text(encoding="utf-8"))}


# ================= adaptadores de referencia (motor-only) =================
class RH:
    _m = None
    _b = None
    _toks = None

    @classmethod
    def reader(cls, base=False):
        if base:
            if cls._b is None:
                cls._b = gguf.GGUFReader(BASE)
            return cls._b
        if cls._m is None:
            cls._m = gguf.GGUFReader(MODELO)
        return cls._m

    @classmethod
    def tensor(cls, nm, base=False):
        return {t.name: t for t in cls.reader(base).tensors}[nm]

    @classmethod
    def dec(cls, t):
        tt = t.tensor_type
        if tt == gguf.GGMLQuantizationType.F32:
            return np.asarray(t.data, dtype=np.float32)
        if tt == gguf.GGMLQuantizationType.F16:
            return t.data.view(np.float16).astype(np.float32)
        return gguf.dequantize(t.data, tt).astype(np.float32)

    @classmethod
    def tokens(cls):
        if cls._toks is None:
            f = cls.reader().fields["tokenizer.ggml.tokens"]
            cls._toks = [x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x) for x in f.contents()]
        return cls._toks

    @classmethod
    def rownorms(cls, nm, base=False, stride=8):
        t = cls.tensor(nm, base)
        D = cls.dec(t)
        P = np.linalg.norm(D.astype(np.float64).reshape(-1, D.shape[-1]), axis=1)
        return P[::stride], D.shape[0] // stride


# ================= jueces =================
JCM = {}


def judge(kind, spec):
    def register(fn):
        JCM[kind] = (fn, spec)
        return fn
    return register


def rel_drift_rows(nm):
    tD = RH.tensor(nm)
    tB = RH.tensor(nm, base=True)
    D = RH.dec(tD)
    B = RH.dec(tB)
    num = np.abs(D - B).sum(axis=1)
    den = np.abs(B).sum(axis=1) + 1e-12
    return num / den


@judge("dims", None)
def j_dims(ans, spec):
    out = {}
    total = 0
    for nm in spec["tensors"]:
        t = RH.tensor(nm)
        ref = [int(x) for x in t.shape]
        got = ans.get("tensors", {}).get(nm)
        okn = got == ref
        out[nm] = {"ref_shape": [int(x) for x in t.shape], "type": t.tensor_type.name, "match": bool(okn)}
        total += 1 if okn else 0
    return {"score": 2.5 * total, "pass": total == len(spec["tensors"]), "detalle": out}


@judge("emb_vocab", None)
def j_emb(ans, spec):
    tvocab = len(RH.tokens())
    nemb = int(RH.tensor("token_embd.weight").shape[1])
    ok = (int(ans.get("vocab_size", -1)) == tvocab) and (int(ans.get("emb_rows", -1)) >= nemb - 1 and int(ans.get("emb_rows", -1)) <= nemb + 4)
    return {"score": 5.0 if (int(ans.get("vocab_size", -1)) == tvocab) else 0.0,
            "pass": ok, "note_emb_rows": "filas del tensor >= vocab"}


@judge("mlp_map", None)
def j_mlp(ans, spec):
    down = RH.tensor("blk.5.ffn_down.weight")
    ref_inter = int(down.shape[0])
    layers = sorted({int(t.name.split('.')[1]) for t in RH.reader().tensors if ".ffn_down.weight" in t.name})
    ref = {"n_ffn_por_capa": 3, "inter": ref_inter, "capas_ffn": len(layers)}
    hits = 0
    hits += int(int(ans.get("n_ffn_por_capa", -1)) == ref["n_ffn_por_capa"]) * 1.7
    hits += int(int(ans.get("inter", -1)) == ref["inter"]) * 2
    hits += int(int(ans.get("capas_ffn", -1)) == ref["capas_ffn"]) * 1.3
    return {"score": hits, "pass": hits >= 4.5, "ref_ranges": {"n_ffn": "3", "inter": "exacto", "capas": "exacto"}}


@judge("heads", None)
def j_heads(ans, spec):
    f1 = RH.reader().fields.get("attention.head_count")
    f2 = RH.reader().fields.get("attention.head_count_kv")
    nh = int(f1.contents()[0]) if f1 else None
    nk = int(f2.contents()[0]) if f2 else None
    nhead = nh if nh is not None else 16
    nkv = nk if nk is not None else 4
    ok = int(ans.get("n_heads", -1)) == nhead and int(ans.get("n_kv_heads", -1)) == nkv
    return {"score": 10.0 if ok else 0.0, "pass": ok, "hint": "GQA: derivable de metadata attention.head_count(_kv) o de shapes attn_q/k"}


@judge("norms", None)
def j_norms(ans, spec):
    order = []
    for nm in spec["tensors"]:
        P, _ = RH.rownorms(nm, stride=32)
        order.append((nm, float(np.sum(P))))
    ref = [nm for nm, _ in sorted(order, key=lambda kv: -kv[1])]
    got = [str(x) for x in ans.get("orden", [])]
    pos_ref = {nm: i for i, nm in enumerate(ref)}
    if not got:
        return {"score": 0, "pass": False, "error": "falta orden"}
    lgt = [pos_ref.get(g, 99) for g in got]
    inv = sum(1 for i in range(1, len(lgt)) if lgt[i - 1] > lgt[i] if 99 not in (lgt[i - 1], lgt[i]))
    score = max(0.0, 5 - 2 * inv)
    return {"score": score, "pass": inv <= 1 and len(got) >= 4, "inversiones": inv}


@judge("outliers", None)
def j_outliers(ans, spec):
    nm = spec["tensor"]
    rd = rel_drift_rows(nm) if "delta" in nm or True else None
    D = RH.dec(RH.tensor(nm))
    P = np.linalg.norm(D.astype(np.float64).reshape(-1, D.shape[-1]), axis=1)
    ref = sorted(np.argsort(P)[::-1][:10].tolist())
    got = sorted(set(int(x) % D.shape[0] for x in ans.get("top_rows", [])))
    ref_s = set(ref)
    got_s = set(got[:20])
    inter = len(ref_s & got_s)
    jacc = inter / max(len(ref_s | got_s), 1)
    return {"score": 10 * jacc, "pass": jacc >= 0.3, "jaccard": round(jacc, 3)}


@judge("delta_magnitud", None)
def j_delta(ans, spec):
    fams = ["ffn_gate", "ffn_down", "ffn_up", "ssm_out", "attn_q", "ssm_alpha", "ssm_beta"]
    res = {}
    mx_f, mx_fv = None, 0
    layer_top, layer_val = None, 0
    for fam in fams:
        tot = []
        for b in range(0, 32, 4):
            nm = f"blk.{b}.{fam}.weight"
            try:
                rr = rel_drift_rows(nm)
            except Exception:
                continue
            tot.append(float(np.mean(rr)))
        if tot:
            m = float(np.mean(tot))
            res[fam] = round(m, 3)
            if m > mx_fv:
                mx_fv, mx_f = m, fam
    ref_fam = max(res, key=res.get)
    got_f = str(ans.get("familia", "").lower())
    fam_ok = ref_fam.split("_")[0] in got_f or got_f.split("_")[0] in ref_fam
    soc = 5.0 * fam_ok
    detail = {"ref_familia": ref_fam}
    return {"score": soc, "pass": soc >= 5.0, "ref_top_families": ref_fam, "ranking": res}


@judge("similarity", None)
def j_sim(ans, spec):
    cand = [f"blk.{b}.attn_q.weight" for b in (3, 7, 11, 15, 19, 23, 27, 31)]
    P = {nm: RH.rownorms(nm, stride=16)[0] for nm in cand}
    homunal = min(len(v) for v in P.values())
    P = {k: v[:homunal] for k, v in P.items()}
    refs = {}
    best_pairs = []
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            c = float(np.corrcoef(P[a], P[b])[0, 1])
            refs[(a, b)] = c
    top2 = sorted(refs.items(), key=lambda kv: -kv[1])[:2]
    ok = 0
    for (a, b), c in top2:
        gotpairs = ans.get("pares_similares", [])
        for pa in gotpairs:
            pa = [str(x) for x in pa] if isinstance(pa, list) else []
            if len(pa) == 2:
                gset = {pa[0], pa[1]}
                if a in str(gset) and b in str(gset):
                    ok += 1
                    break
    return {"score": 5 * ok, "pass": ok >= 1, "hint": "el par top del perfil normativo entre capas"}


@judge("emb_neighbors", None)
def j_embn(ans, spec):
    tok_embd = RH.dec(RH.tensor("token_embd.weight"))[: len(RH.tokens())]
    toks = RH.tokens()
    nrm = np.linalg.norm(tok_embd, axis=1)
    hits = 0
    detalle = {}
    for sid in spec["seed_ids"]:
        q = tok_embd[sid] / max(np.linalg.norm(tok_embd[sid]), 1e-9)
        sims = tok_embd @ q / np.maximum(nrm, 1e-9)
        top3 = np.argsort(sims)[::-1][1:4]
        ref = {toks[int(x)] for x in top3}
        got = ans.get("pred", {}).get(str(sid), [])
        inter = len({t for t in got} & ref)
        hits += (1 if inter >= 2 else (0.4 if inter == 1 else 0))
        detalle[sid] = {"seed": toks[sid], "match": int(inter)}
    return {"score": 3.3 * hits, "pass": hits >= 2.0, "detalle": str(detalle)[:240]}


@judge("estructura", None)
def j_struct(ans, spec):
    req = ["hipotesis", "evidencia", "contra_evidencia", "componente_candidato", "prediccion_falsable"]
    n = sum(1 for k in req if ans.get(k))
    return {"score": 2.0 * n, "pass": n >= 5, "falta": [k for k in req if not ans.get(k)]}


@judge("markers", None)
def j_markers(ans, spec):
    ids_markup = json.loads((ROOT / "data" / "markers.json").read_text(encoding="utf-8"))
    ref_n = len(ids_markup)
    got_n = int(ans.get("n_markers", -1))
    got_ids = [x for x in ans.get("ids", []) if isinstance(x, (int, float))]
    ref_ids = [int(x) for x in list(ids_markup.keys())]
    inter = len(set(ref_ids) & set(int(x) for x in got_ids[:120]))
    score = 0
    if got_n == ref_n:
        score += 5
    score += min(5, inter * 0.25)
    return {"score": score, "pass": score >= 7.5, "note": "n exacto + subconjunto válido"}


@judge("shared_rows", None)
def j_shared(ans, spec):
    cand = ["blk.17.ffn_down.weight", "blk.19.ffn_down.weight", "blk.22.ssm_beta.weight"]
    got = [int(x) for x in ans.get("filas_compartidas", [])[:20]]
    compute = 0
    detail = {}
    for a, b in ((cand[0], cand[1]), (cand[0], cand[2])) if len(cand) == 3 else []:
        try:
            ra = np.argsort(rel_drift_rows(a))[::-1][:12]
            rb = np.argsort(rel_drift_rows(b))[::-1][:12]
            ref = set(ra.tolist()) & set(rb.tolist())
        except Exception:
            continue
        inter = ref & set(got)
        compute += len(inter)
        detail[f"{a}~{b}"] = len(inter)
    return {"score": 2.5 * compute, "pass": compute >= 2, "detalle": detail}


@judge("toy", None)
def judge_toy(ans, spec):
    """Reto toy: el agente entrega {'task', 'program'}; corre mini_toy 200 ejemplos"""
    task = ans.get("task", spec.get("task", "parity"))
    program = ans.get("program") or None
    r = subprocess.run([sys.executable, str(ROOT / "toy_bench" / "mini_toy.py"),
                        "--task", task, *(["--program", program] if program else [])],
                       capture_output=True, text=True, timeout=1200,
                       cwd=str(ROOT), stdin=subprocess.DEVNULL, **_CREATION_FLAGS)
    out = r.stdout
    try:
        acc = float(out.split("ACC")[1].split()[0])
    except Exception:
        return {"score": -2, "pass": False, "error": out[-300:]}
    mn = float(spec.get("min_acc", 0.97))
    return {"score": 10 * acc, "pass": acc >= mn, "acc": acc}


@judge("lang_rows", None)
def j_lang(ans, spec):
    toks = RH.tokens()
    pat = [t for t in toks if any("\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff" or "\u0e00" <= c <= "\u0e7f" for c in t)]
    got = [x for x in ans.get("tokens", []) if isinstance(x, str)]
    inter = len(set(pat[:200]) & set(got))
    return {"score": min(10, 3.3 * inter), "pass": inter >= spec.get("min_match", 3)}


JCM_NULL = {"rms": None}


def _judget(kind, spec, ans):
    fn, sp = JCM.get(kind, (None, None))
    if fn is None:
        return {"score": -1, "pass": False, "error": f"juez '{kind}' no implementado (retó in ajunde)"}
    try:
        return fn(ans, spec or {})
    except Exception as e:
        return {"score": -2, "pass": False, "error": f"juez error: {repr(e)[:160]}"}


def _save_exp(id_, exp):
    exp["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (EXP / f"{id_.zfill(2)}.json").write_text(json.dumps(exp, ensure_ascii=False, indent=1), encoding="utf-8")


def submit(id_, ansfile):
    exp_p = EXP / f"{id_.zfill(2)}.json"
    exp = json.loads(exp_p.read_text(encoding="utf-8")) if exp_p.exists() else {"id": id_, "intentos": 0, "estado": "en_curso", "hits": []}
    try:
        ans = json.loads(Path(ansfile).read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"respuesta JSON invalida: {repr(e)[:120]}"}
    cat = catalogo()
    rr = next((x for x in cat["retos"] if x["id"] == id_.zfill(2)), None)
    out = _judget(rr["juez"]["kind"], rr["juez"], ans)
    exp["intentos"] = int(exp.get("intentos", 0)) + 1
    exp["score"] = max(float(exp.get("score", 0)), max(0.0, float(out.get("score", 0))))
    if out.get("pass"):
        exp["estado"] = "aprobado"
        exp.setdefault("aprendizaje", rr.get("aprendizaje"))
    else:
        exp["estado"] = "en_curso" if exp["estado"] != "fallido" else "fallido"
    exp.setdefault("hits", []).append({"ts": time.strftime("%H:%M:%S"), "score": round(float(out.get("score", 0)), 2), "pass": bool(out.get("pass")), "detalle": str(out.get("detalle", out.get("falta", "")))[:240]})
    _save_exp(id_, exp)
    _log("submit", reto=id_, score=out.get("score"), passed=bool(out.get("pass")))
    return {"juez": out, "expediente": exp}


def _rig_stats(alias):
    p = RIG / f"{alias}.json"
    if not p.exists():
        return None
    rs = json.loads(p.read_text(encoding="utf-8"))
    nn = [r for r in rs if "summary" not in r]
    return {"n": len(nn), "aciertos": sum(1 for r in nn if r.get("ok")),
            "loops": sum(1 for r in nn if r.get("loop")),
            "caps": sum(1 for r in nn if r.get("finish") == "length")}


def feedback(alias):
    refs = [f.name for f in RIG.iterdir() if alias in f.name]
    return {"archivos": refs, "stats": {f.name: _rig_stats(f.stem) for f in RIG.iterdir() if alias in f.name}}


def sellar(alias, dset, port):
    marca = f"{alias}__{dset}"
    lock = SELLADOS / f"{marca}.LOCKED"
    if lock.exists():
        return {"error": "holdout ya sellado para este candidato+set"}
    n = {"math500": 12, "diag": 8, "code": 6, "if": 6}.get(dset, 12)
    sset = dset if dset in ("math500", "diag") else "math500"
    oset = {"code": 24, "if": 40}.get(dset, 0)
    import subprocess
    args = ["python", str(ROOT / r"blog_prompts\eval_math.py"), alias, sset, "-p", str(port), "-n", str(n), "-o", str(oset)]
    rc = subprocess.run(args, capture_output=True, text=True, timeout=5400)
    p = RIG / f"{alias}-math.json"
    if not p.exists():
        return {"error": "eval fallo", "rc": rc.returncode, "tail": rc.stdout[-300:]}
    rs = json.loads(p.read_text(encoding="utf-8"))
    nn = [r for r in rs if "summary" not in r]
    ok = sum(1 for r in nn if r.get("ok"))
    caps = sum(1 for r in nn if r.get("finish") == "length")
    sellado = {"alias": alias, "set": dset, "n": len(nn), "ok": ok, "caps_length": caps,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (SELLADOS / f"{marca}.json").write_text(json.dumps(rs), encoding="utf-8")
    lock.write_text("sellado", encoding="utf-8")
    _log("holdout_sellado", **sellado)
    return {"ok": True, "resumen": sellado}


def verdicto(cand, vered, notas=""):
    vered = (vered or "").lower()
    if vered not in ("keep", "discard"):
        return {"error": "keep|discard"}
    _log("verdicto", cand=cand, vered=vered, notas=notas)
    return {"ok": True, "estado": _verdict_stats()}


def historial(n):
    rows = [json.loads(l) for l in HIST.read_text(encoding="utf-8").splitlines()] if HIST.exists() else []
    return {"eventos": rows[-int(n):] if n else rows, "verdictos": _verdict_stats(), "aprobados": [p.stem for p in EXP.glob("*.json") if json.loads(p.read_text(encoding="utf-8")).get("estado") == "aprobado"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("a", nargs="?", default=None)
    ap.add_argument("b", nargs="?", default=None)
    ap.add_argument("c", nargs="?", default=None)
    ap.add_argument("-p", "--port", type=int, default=8950)
    ap.add_argument("-n", type=int, default=40)
    args = ap.parse_args()
    w = {"submit": lambda: submit(args.a, args.b),
         "reto": lambda: reto(args.a),
         "expediente": lambda: expediente(args.a or ""),
         "feedback": lambda: feedback(args.a or ""),
         "sellar": lambda: sellar(args.a, args.b, args.port),
         "verdicto": lambda: verdicto(args.a, args.b, args.c),
         "historial": lambda: historial(args.n),
         "estado": estado}
    fn = w.get(args.cmd)
    out = fn() if fn else {"error": "cmd"}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
