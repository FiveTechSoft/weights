"""math500.py — runner de evaluación math-500 (estudio WSP, server OpenAI-compatible).

Levanta un subset determinista (seed+n) del dataset data/math500_test.jsonl, consulta
un endpoint tipo llama-server (/v1/chat/completions) con prompt user-only que pide la
respuesta final en \\boxed{...}, y compara con normalización determinista (stdlib only).

Sellado (--sell): al terminar el run escribe un lock en rig/sellados/ con SHA-256 de
{seed,indices} y hashes SHA-256 por gold. Si existe lock para ese seed/n reusa sus
índices; si el rig ya está sellado con OTRA combinación seed/n => error, nunca sobrescribe.

El schema de salida sigue rig_runs/r0-orig-math.json ({i, gold, pred, ok, finish, tail})
más un dict clave "meta" con {model, endpoint, seed, n, indices, temperature, max_tokens,
tag, ts, counts:{ok, total}}.

Durante el run NO se imprimen golds (ni ok) de los ítems evaluados: el resumen con los
fallos (i, gold, pred) sale solo al terminar.

Datos del dataset que maneja la normalización (500 ítems):
  311 enteros; fracs \\frac/{a}{b}, \\dfrac y sin llaves (\\frac43, \\frac 59, \\frac9{19});
  \\sqrt{a} y sin llaves (11\\sqrt2); \\pi; tuplas "(6,31,-1)" / listas "1,-2";
  grados "90^\\circ"; \\text{...} ("\\text{Evelyn}", "\\frac{270}7\\text{ degrees}");
  subscript "52_8"; miles con \\! ("10,\\!080", "\\$32,\\!348").
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[1]
DIR_DATA = ROOT / "data" / "math500_test.jsonl"
DIR_OUT = ROOT / "rig_runs"
DIR_LOCK = ROOT / "rig" / "sellados"
DIR_OUT.mkdir(parents=True, exist_ok=True)

REQ_TIMEOUT = 600  # segundos por petición al server


def _p(*a):
    print(*a, flush=True)


# ---------------- dataset ----------------

def load_items():
    items = []
    for line in DIR_DATA.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# ---------------- selección determinista del subset ----------------
# PRNG determinista propio (sha256 con contador): mismo seed+n => mismos índices SIEMPRE,
# estable entre versiones/plataformas (no depende del módulo random).

def _draw(seed, c):
    h = hashlib.sha256(f"wsp-select|{seed}|{c}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def select_indices(seed, n, total=500):
    if n <= 0 or n >= total:
        return list(range(total))
    idx = list(range(total))
    c = 0
    # Fisher-Yates completo con fuente sha256, subset = primeras n posiciones
    for k in range(total - 1, 0, -1):
        idx[k], idx[_draw(seed, c) % (k + 1)] = idx[_draw(seed, c) % (k + 1)], idx[k]
        c += 1
    order = sorted(idx[:n])
    return order


def sel_digest(seed, indices):
    blob = json.dumps({"seed": seed, "indices": [int(i) for i in indices]},
                      sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def gold_hashes(items, indices):
    return {str(i): hashlib.sha256(items[i]["answer"].encode()).hexdigest()
            for i in indices}


# ---------------- sellado (locks rig/sellados) ----------------

def _lock_path(seed, n):
    return DIR_LOCK / f"sellado_{seed}_{n}.json"


def sell_reuse_or_seal(seed, n, items, indices):
    """Devuelve los índices a usar; escribe el lock si es un run nuevo. Nunca sobrescribe."""
    DIR_LOCK.mkdir(parents=True, exist_ok=True)
    lp = _lock_path(seed, n)
    if lp.exists():
        lock = json.loads(lp.read_text(encoding="utf-8"))
        old = [int(i) for i in lock["indices"]]
        if sel_digest(lock["seed"], old) != lock["digest"]:
            _p(f"[sellado] corrupto {lp.name} (digest no coincide)"); sys.exit(1)
        cur = gold_hashes(items, old)
        if cur != lock["golds"]:
            _p(f"[sellado] golds cambiaron respecto al lock {lp.name} (dataset modificado)"); sys.exit(1)
        _p(f"[sellado] reutilizando lock {lp.name} ({len(old)} índices)")
        return old
    # rig sellado con otra combinación seed/n => error, jamás se sobrescribe
    for other in DIR_LOCK.glob("sellado_*.json"):
        _p(f"[sellado] ERROR: rig ya sellado con {other.name}; pedido {lp.name} distinto => no se sobrescribe")
        sys.exit(1)
    lock = {"seed": seed, "n": n, "indices": [int(i) for i in indices],
            "digest": sel_digest(seed, indices), "golds": gold_hashes(items, indices),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    lp.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")
    _p(f"[sellado] lock nuevo escrito: {lp}")
    return indices


# ---------------- prompt ----------------

def build_prompt(problem):
    return (
        "Resuelve el siguiente problema de matemáticas.\n\n"
        f"Problema:\n{problem}\n\n"
        "Piensa paso a paso y termina dando la respuesta final escrita exactamente "
        "dentro de \\boxed{...}."
    )


# ---------------- extracción de pred ----------------

def _grab(s, j):
    """Token LaTeX en s[j]: {'{...}' balanceado con escapes \\{ \\} -> contenido} o 1 char.
    Devuelve (token|None, pos_fin)."""
    while j < len(s) and s[j] == " ":
        j += 1
    if j >= len(s):
        return None, j
    if s[j] == "{":
        depth, k = 1, j + 1
        while k < len(s):
            ch = s[k]
            if ch == "\\":          # escapes: \{ \} no abren/cierran
                k += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[j + 1:k], k + 1
            k += 1
        return None, j             # sin cerrar: no válido
    return s[j], j + 1             # token sin llaves: un solo carácter


def extract_boxed(content):
    """Último \\boxed{...} con llaves balanceadas (soporta \\{ \\}); acepta \\boxed tok suelto.
    Devuelve (encontrado, pred); \\boxed{} vacío => (True, "")."""
    best = ("\x00", None)
    for m in re.finditer(r"\\boxed(?![a-zA-Z])", content):
        tok, end = _grab(content, m.end())
        if tok is not None:
            best = (tok, end)
    if best[1] is None:
        return False, ""
    return True, best[0]


def extract_fallback(content):
    """Fallback: último número o expresión final (a/b con enteros o decimal)."""
    t = content.replace("$", " ")
    pats = [r"[-+]?\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?",
            r"[-+]?(?:\d+\.?\d*|\.\d+)"]
    for p in pats:
        ms = list(re.finditer(p, t))
        if ms:
            return ms[-1].group().replace(" ", "")
    return ""


def extract_pred(content):
    ok_box, pred = extract_boxed(content)
    if ok_box:
        return pred               # puede ser "" (\\boxed{} vacío)
    # Respect the final-answer format used by the legacy math runner.
    finals = re.findall(r"(?:^|\n|\|)\s*Respuesta:\s*([^\n|]+)", content, re.IGNORECASE)
    if finals:
        return finals[-1].strip().strip("$")
    return extract_fallback(content)


# ---------------- normalización determinista (stdlib) ----------------

def _fracs(s):
    """\\frac{a}{b}/\\dfrac/\\tfrac y variantes sin llaves (\\frac43, \\frac 59, \\frac9{19})
    -> 'a/b'. Los args sin llaves son de UN solo carácter o \\macro."""
    def group(a):
        # Preserve operator precedence; simple scalar/symbol needs no parentheses.
        return a if re.fullmatch(r"[-+]?\d+(?:\.\d+)?|\\?[a-zA-Z]+", a) else "(" + a + ")"

    out, pos = [], 0
    pattern = re.compile(r"\\[dt]?frac(?![a-zA-Z])")
    while True:
        m = pattern.search(s, pos)
        if m is None:
            out.append(s[pos:])
            break
        a, j2 = _grab(s, m.end())
        b, j3 = _grab(s, j2) if a is not None else (None, j2)
        if a is None or b is None:
            out.append(s[pos:m.end()]); pos = m.end()
            continue
        a, b = _fracs(a), _fracs(b)
        out.extend((s[pos:m.start()], group(a) + "/" + group(b)))
        pos = j3
    return "".join(out)


def _sqrts(s):
    """Preserve explicit root indices; never equate cube and square roots."""
    out, pos = [], 0
    for m in re.finditer(r"\\sqrt(?![a-zA-Z])", s):
        j = m.end()
        index = "2"
        if j < len(s) and s[j] == "[":                     # salta índice opcional [n]
            depth, k = 1, j + 1
            while k < len(s) and depth:
                if s[k] == "[":
                    depth += 1
                elif s[k] == "]":
                    depth -= 1
                k += 1
            if depth:
                continue
            index = s[j + 1:k - 1].strip()
            j = k
        a, j2 = _grab(s, j)
        if a is None:
            continue
        out.append(s[pos:m.start()]); out.append(("sqrt(" + a + ")") if index == "2" else ("root(" + index + "," + a + ")")); pos = j2
    out.append(s[pos:])
    return "".join(out)


def _texts(s):
    """\\text{...} -> su contenido (permite comparar \\text{Evelyn} vs Evelyn)."""
    out, pos = [], 0
    for m in re.finditer(r"\\text(?![a-zA-Z])", s):
        a, j2 = _grab(s, m.end())
        if a is None:
            continue
        out.append(s[pos:m.start()]); out.append(a); pos = j2
    out.append(s[pos:])
    return "".join(out)


_THOUS = re.compile(r"(\d),(?=\d{3}(?:\D|$))")   # coma de miles SOLO entre grupos completos


def norm(s):
    s = s.strip().lower()
    s = s.replace("\\$", "").replace("$", "")          # \\$32,\\!348 -> 32,\\!348
    s = re.sub(r"\\[,;:!]", "", s)                     # espaciadores \\, \\! \\; (golds: 10,\\!080)
    s = re.sub(r"\\left|\\right", "", s)
    s = _texts(s)
    s = _fracs(s)
    s = _sqrts(s)
    s = s.replace("\\pi", "pi")
    s = s.replace("^{\\circ}", "^\\circ").replace("^\\circ", "")   # 90^\\circ -> 90
    s = re.sub(r"\b(degrees|cents|grados)\b", "", s)   # unidades vist en golds
    s = re.sub(r"\s+", "", s)
    # comas de miles (burbujea de derecha a izquierda en 1,234,567)
    while True:
        # A comma inside a tuple/list may separate coordinates, not thousands.
        s2 = s if any(c in s for c in "()[]{}") else _THOUS.sub(r"\1", s)
        if s2 == s:
            break
        s = s2
    # llaves simples en ^/_: 4210_{5} -> 4210_5, 2^{10} -> 2^10
    s = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", s)
    # fracción a/b con enteros -> forma reducida (equivale \\frac{14}{3} con 14/3)
    def _red(m):
        n, d = int(m.group(1)), int(m.group(2))
        if d == 0:
            return m.group(0)
        a, b = abs(n), abs(d)
        while b:
            a, b = b, a % b
        g = a or 1
        n, d = n // g, d // g
        return f"{n}/{d}"
    s = re.sub(r"(?<![\w./])(-?\d+)/(\d+)", _red, s)
    s = s.strip(",.") if s.count(".") > 1 or not re.fullmatch(r"-?\d*\.\d*", s) else s
    return s


def _numval(s):
    """Valor numérico si s es entero/decimal/fracción de enteros; None si no."""
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)", s):
        try:
            from fractions import Fraction
            return Fraction(s)
        except ZeroDivisionError:
            return None
    m = re.fullmatch(r"([-+]?\d+)/(\d+)", s)
    if m and int(m.group(2)) != 0:
        from fractions import Fraction
        return Fraction(int(m.group(1)), int(m.group(2)))
    return None


def equivalent(pred, gold):
    """pred == gold tras normalizar; equipara formas numéricas exactas (1 vs 1.0, 0.5 vs 1/2)."""
    pn, gn = norm(pred), norm(gold)
    if not pn:                      # \\boxed vacío / sin pred => nunca ok
        return False
    if pn == gn:
        return True
    vp, vg = _numval(pn), _numval(gn)
    return vp is not None and vg is not None and vp == vg


# ---------------- llamada al server (OpenAI-compatible, llama-server) ----------------

def chat(endpoint, model, prompt, temperature, max_tokens, retries):
    payload = json.dumps({"model": model, "stream": False, "temperature": temperature,
                          "max_tokens": max_tokens,
                          "messages": [{"role": "user", "content": prompt}]}).encode()
    last = ""
    for r in range(retries + 1):
        try:
            req = urllib.request.Request(endpoint, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            ch = data.get("choices") or [{}]
            finish = ch[0].get("finish_reason") or ""
            content = ((ch[0].get("message") or {}).get("content")) or ""
            return content, finish
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            ok_retry = e.code == 429 or 500 <= e.code < 600
        except (urllib.error.URLError, socket.timeout, OSError, KeyError, ValueError) as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            ok_retry = True
        if not ok_retry or r == retries:
            break
        back = 2 ** (r + 1)          # 2s, 4s, ... sobre 429/5xx/timeouts
        _p(f"    reintento {r + 1}/{retries} en {back}s ({last})")
        time.sleep(back)
    return "", f"error:{last}"


# ---------------- evaluación ----------------

def eval_one(item, args):
    t0 = time.time()
    content, finish = chat(args.endpoint, args.model, build_prompt(item["problem"]),
                           args.temperature, args.max_tokens, args.retries)
    pred = extract_pred(content)
    ok = bool(pred) and equivalent(pred, item["answer"])
    tail = content[-300:]
    return {"i": item["i"], "gold": item["answer"], "pred": pred, "ok": ok,
            "finish": finish, "tail": tail,
            "_seg": round(time.time() - t0, 1)}


def run_subset(indices, items, args):
    results = [None] * len(indices)
    t0 = time.time()
    if args.parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(eval_one, items[i], args): k for k, i in enumerate(indices)}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                k = futs[fut]
                results[k] = fut.result()
                done += 1
                r = results[k]
                _p(f"[{done}/{len(indices)}] i={r['i']} finish={r['finish']} seg={r['_seg']}")  # sin gold ni ok
    else:
        for done, i in enumerate(indices, 1):
            r = eval_one(items[i], args)
            results[done - 1] = r
            _p(f"[{done}/{len(indices)}] i={i} finish={r['finish']} seg={r['_seg']}")  # sin gold ni ok
    dur = round(time.time() - t0, 1)
    for r in results:
        r.pop("_seg")
    return results, dur


# ---------------- CLI ----------------

def make_parser():
    ap = argparse.ArgumentParser(prog="math500",
                                 description="runner de evaluación math-500 (subset determinista)")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8090/v1/chat/completions")
    ap.add_argument("--model", default="mimo")
    ap.add_argument("--n", type=int, default=0, help="tamaño del subset; 0 = todos (500)")
    ap.add_argument("--seed", type=int, default=42, help="seed del subset determinista")
    ap.add_argument("--sell", action="store_true",
                    help="sellado: lock en rig/sellados/ (dígesto de {seed,indices} + sha256 por gold)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=8192, dest="max_tokens")
    ap.add_argument("--out", default=None, help="ruta JSON de salida (default rig_runs/r0-math-<tag>.json)")
    ap.add_argument("--tag", default="orig")
    ap.add_argument("--retries", type=int, default=2, help="reintentos por 429/5xx/timeout")
    ap.add_argument("--parallel", type=int, default=1, help="hilos en paralelo (default 1 = serial)")
    ap.add_argument("--selftest", action="store_true")
    return ap


def cmd_run(args):
    items_all = load_items()
    total = len(items_all)
    for k, it in enumerate(items_all):
        it["i"] = k
    n = args.n if args.n else total
    indices = select_indices(args.seed, n, total)
    if args.sell:
        indices = sell_reuse_or_seal(args.seed, len(indices), items_all, indices)
    _p(f"[run] seed={args.seed} n={len(indices)} endpoint={args.endpoint} model={args.model} "
       f"temp={args.temperature} max_tokens={args.max_tokens} parallel={args.parallel} tag={args.tag}")
    mostrado = str(indices) if len(indices) <= 24 else str(indices[:24])[:-1] + ", ...]"
    _p(f"[run] índices: {mostrado}")
    results, dur = run_subset(indices, items_all, args)
    nok = sum(1 for r in results if r["ok"])
    meta = {"model": args.model, "endpoint": args.endpoint, "seed": args.seed,
            "n": len(indices), "indices": indices, "temperature": args.temperature,
            "max_tokens": args.max_tokens, "tag": args.tag,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "counts": {"ok": nok, "total": len(results)}}
    out = Path(args.out) if args.out else DIR_OUT / f"r0-math-{args.tag}.json"
    out.write_text(json.dumps({"meta": meta, "items": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    _p(f"[fin] {dur}s — resumen de fallos (i, gold, pred):")   # golds solo al terminar
    fails = [r for r in results if not r["ok"]]
    for r in fails:
        _p(f"  i={r['i']} gold={r['gold']!r} pred={r['pred']!r}")
    _p(f"[fin] sin fallos" if not fails else f"[fin] fallos: {len(fails)}/{len(results)}")
    _p(f"[fin] ok={nok}/{len(results)} ({100.0 * nok / len(results):.1f}%) -> {out}")


def cmd_selftest(_):
    """Selftest: asserts de extracción/normalización + determinismo del subset."""
    k = 0

    def chk(cond, msg):
        nonlocal k
        k += 1
        assert cond, f"selftest FAIL #{k}: {msg}"

    # --- extracción \\boxed (balanceado, escapes, vacío) ---
    chk(extract_pred("respueta \\boxed{42} fin") == "42", "boxed simple")
    chk(extract_pred("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}", "boxed con llaves anidadas")
    chk(extract_pred(r"conjunto \boxed{\{1,2,\}3}") == r"\{1,2,\}3", r"boxed con escapes \{ \}")
    chk(extract_pred(r"final: \boxed{\left( 3, \frac{\pi}{2} \right)}") == r"\left( 3, \frac{\pi}{2} \right)",
        "boxed con \\left/\\frac (formato gold real)")
    chk(extract_pred("\\boxed{x=1} \\boxed{\\frac{14}{3}}") == "\\frac{14}{3}", "último \boxed gana")
    chk(extract_pred("El resultado es \\boxed{} nada.") == "", r"\boxed{} vacío => pred=''")
    chk(extract_pred(r"asi que \boxed 7.") == "7", r"\boxed sin llaves (token suelto)")
    chk(equivalent(extract_pred("resp \\boxed{}"), "5") is False, r"\boxed{} vacío => ok=False")
    # --- fallback (último número / expresión) ---
    chk(extract_pred("El total es 3.21 unidades.") == "3.21", "fallback último decimal")
    chk(extract_pred("por tanto x = 14/3]") == "14/3", "fallback fracción final")
    chk(extract_pred("no hay nada") == "", "fallback sin número")

    # --- normalización: formatos reales del dataset ---
    chk(norm(r"\left( 3, \frac{\pi}{2} \right)") == norm("(3, pi/2)"), r"tupla \left + \frac{\pi}")
    chk(norm(r"\dfrac{17}{50}") == norm("17/50"), r"\dfrac == a/b")
    chk(norm(r"\tfrac{3}{2}") == norm(r"\frac{3}{2}"), r"\tfrac == \frac")
    chk(norm(r"\frac 59") == norm(r"\frac{5}{9}"), r"\frac 59 (sin llaves, espacio) == 5/9")
    chk(norm(r"\frac43") == norm(r"\frac 43") == "4/3", r"\frac43 sin llaves")
    chk(norm(r"\frac9{19}") == norm("9/19"), r"\frac9{19} llave parcial")
    chk(norm(r"\frac{14}{3}") == norm("14/3"), r"\frac{14}{3} == 14/3 (equivalencia)")
    chk(norm(r"\frac{2}{4}") == norm(r"\frac 12"), r"\frac reducido (2/4 == 1/2)")
    chk(norm(r"3\sqrt{13}") != norm(r"11\sqrt2"), "sqrt con coeficientes distintos difiere")
    chk(norm(r"11\sqrt2") == norm(r"11 \sqrt{2}"), r"\sqrt2 sin llaves == \sqrt{2}")
    chk(norm(r"3\sqrt{13}") == norm("3sqrt(13)"), r"\sqrt{a} -> sqrt(a)")
    chk(norm(r"- \frac{\pi}{6}") == norm("-pi/6"), r"\pi constante")
    chk(norm("(6,31,-1)") == norm("( 6, 31 , -1 )"), "tupla con espacios")
    chk(norm("1,-2") == norm("1, -2"), "lista/tupla sin espacios tras coma")
    chk(norm("1,234,567") == norm("1234567"), "miles completos se quitan")
    chk(norm("1,234") == norm("1234"), "miles con un grupo completo")
    chk(norm("P - Q") == norm("p - q"), "minúsculas/espacios")
    chk(norm(r"90^\circ") == norm(r"90^{\circ}") == "90", r"^\circ fuera (equivalente a entero)")
    chk(norm(r"\frac{270}7\text{ degrees}") == norm(r"\frac{270}{7}"), r"\text{ degrees} fuera")
    chk(norm(r"5.4 \text{ cents}") == norm("5.4"), r"\text{ cents} fuera")
    chk(norm(r"\text{Evelyn}") == norm("Evelyn"), r"\text{...} -> texto simple")
    chk(norm(r"52_8") == norm(r"52_{8}"), "subscript base con/sin llaves")
    chk(norm(r"10,\!080") == norm("10080"), r"miles con \! (gold real)")
    chk(norm(r"\$32,\!348") == norm("32348"), r"\$ fuera (gold real)")
    chk(equivalent("0.15", "3/20"), "equivalencia decimal/fracción (0.15 == 3/20)")
    chk(equivalent("1", "1.0"), "equivalencia entero/decimal (1 == 1.0)")
    chk(equivalent("", "17") is False, "pred vacío nunca ok")

    # --- determinismo del subset ---
    a1, a2 = select_indices(42, 20), select_indices(42, 20)
    chk(a1 == a2, "mismo seed+n => mismos índices (dos llamadas)")
    chk(len(a1) == 20 and len(set(a1)) == 20, "subset tamaño n sin repetidos")
    chk(all(0 <= i < 500 for i in a1), "índices en rango [0,500)")
    chk(select_indices(42, 0) == list(range(500)), "n=0 => todos los índices")
    chk(select_indices(42, 500) == list(range(500)), "n=total => todos")
    chk(select_indices(7, 5) != a1, "seed distinto => selección distinta")

    _p(f"selftest: {k} asserts OK")


def main():
    ap = make_parser()
    a = ap.parse_args()
    if a.selftest:
        cmd_selftest(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
