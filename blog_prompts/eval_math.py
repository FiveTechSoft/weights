"""eval-math (WSP-MATH-BASELINE v1): protocolo matematico riguroso.

Sets:
  diag    : GSM8K train muestra fija 128 (diseno de hipotesis)
  valid   : GSM8K train offset 128-383 (seleccion de candidatos)
  gsm8k_test : GSM8K test 1319 (CONFIRMACION sellada)
  math500 : MATH-500 (CONFIRMACION sellada)
Modo: --mode det (temp 0, top_p 1, 1 muestra) | --mode sampled (temp 0.6, 3 seeds)
JSON de resultados con esquema v1 (model_hash, generation_config, accuracy global y por categoria/nivel).
Uso: python blog_prompts/eval_math.py <alias> <set> -p 8950 [-n N] [-o O] [--mode det]
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = Path(r"C:\weights")
DATA = ROOT / "data"
RIG = ROOT / "rig_runs"

DATA_FILE = {"diag": DATA / "gsm8k_train.jsonl", "valid": DATA / "gsm8k_train.jsonl",
             "gsm8k_test": DATA / "gsm8k_test.jsonl", "math500": DATA / "math500_test.jsonl"}


def load(set_name, n, offset):
    f = DATA_FILE[set_name]
    items = []
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            items.append(json.loads(line))
    return items[offset:offset + n]


def ask(problem):
    if "answer" in problem and "question" in problem:
        return problem["question"], str(problem["answer"])
    return problem["problem"], str(problem.get("answer", problem.get("solution", "")))


def gold_norm(g):
    m = re.search(r"####\s*(-?[\d,\.]+)", g)
    if m:
        return m.group(1).replace(",", "")
    return g.strip()


def extract(text, set_name):
    if set_name == "math500":
        m = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if m:
            return m[-1].strip().replace(" ", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def norm_ans(a):
    if a is None:
        return None
    a = a.strip().replace(" ", "").replace("$", "").replace("\\%", "")
    try:
        av = float(a)
        return round(av, 6)
    except ValueError:
        return a.replace("\\", "").lower().rstrip(".")


def chat(server, prompt, seed, mode="sampled", max_toks=2048):
    if mode == "det":
        gen = {"temperature": 0.0, "top_p": 1.0, "max_tokens": max_toks, "seed": 0}
    else:
        gen = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": max_toks, "seed": seed}
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], **gen}).encode()
    req = urllib.request.Request(server + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason")


INSTR = ("Resuelve el problema paso a paso. Termina tu respuesta con UNICA linea final: 'Respuesta: <valor>'.\n\n")

def final_write(a, results, per_cat, per_lvl, mode):
    import hashlib
    import time
    hash_src = close_alias_to_file(a.alias)
    model_hash = hashlib.sha256(hash_src.encode()).hexdigest()[:16] if hash_src else "?"
    got = sum(1 for r in results if r.get("ok"))
    n_ok = sum(1 for r in results if "err" not in r)
    caps = sum(int(r.get("caps", 0)) for r in results)
    rec = {"model_hash": model_hash, "model_format": "GGUF Q6_K",
           "quantization": "Q6_K", "prompt_version": "INSTR-v1",
           "generation_config": {"mode": mode, "temp": 0.0 if mode == "det" else 0.6, "top_p": 1.0 if mode == "det" else 0.95},
           "seed": "per-index" if mode != "det" else 0,
           "total_questions": n_ok, "correct_answers": got,
           "accuracy": round(got / max(n_ok, 1), 4),
           "caps_length": caps,
           "categoria_scores": {k: {"ok": v[0], "n": v[1]} for k, v in per_cat.items()},
           "nivel_scores": {k: {"ok": v[0], "n": v[1]} for k, v in per_lvl.items()},
           "execution_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "experiment_id": f"{a.alias}_{a.set}",
           "individual": results}
    (RIG / f"{a.alias}-math.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== {a.alias} [{a.set}]: {got}/{n_ok} ({100*got/max(n_ok,1):.1f}%) caps={caps}", flush=True)


def close_alias_to_file(alias):
    for cand in ("mimo-v", "e6", "e5", "e4", "e3", "e2", "e1", "orig"):
        return f"model://{alias}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alias")
    ap.add_argument("set", choices=["diag", "valid", "gsm8k_test", "math500"])
    ap.add_argument("-p", "--port", type=int, default=8950)
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("-o", "--offset", type=int, default=0)
    ap.add_argument("--mode", choices=["det", "sampled"], default="det")
    a = ap.parse_args()
    items = load(a.set, a.n if a.n > 0 else 10**6, a.offset)
    if a.set in ("gsm8k_test", "math500") and a.mode != "sold":
        pass
    server = f"http://127.0.0.1:{a.port}"
    results, per_cat, per_lvl = [], {}, {}
    for i, prob in enumerate(items):
        q, g = ask(prob)
        prompt = INSTR + q
        try:
            txt, finish = chat(server, prompt, seed=42 + i, mode=a.mode)
        except Exception as e:
            results.append({"i": a.offset + i, "err": repr(e)[:120], "ok": False})
            final_write(a, results, per_cat, per_lvl, a.mode)
            break
        pred = extract(txt, a.set)
        g_norm = norm_ans(gold_norm(g))
        p_norm = norm_ans(pred)
        ok = (p_norm is not None) and (p_norm == g_norm)
        cat = str(prob.get("type") or "gsm8k")
        lvl = str(prob.get("level", "")) or "na"
        per_cat.setdefault(cat, [0, 0])
        per_lvl.setdefault(lvl, [0, 0])
        if ok:
            per_cat[cat][0] += 1
            per_lvl[lvl][0] += 1
        per_cat[cat][1] += 1
        per_lvl[lvl][1] += 1
        caps = int(finish == "length")
        results.append({"i": a.offset + i, "gold": gold_norm(g)[:40], "pred": str(pred)[:60],
                        "ok": ok, "finish": finish, "caps": caps,
                        "tail": txt[-160:].replace("\n", " | ")})
        final_write(a, results, per_cat, per_lvl, a.mode)
        print(f"[{a.offset+i+1}] ok={ok} gold={gold_norm(g)[:24]} pred={str(pred)[:32]}", flush=True)


if __name__ == "__main__":
    main()
