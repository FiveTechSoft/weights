"""eval-math: protocolo WSP-matematicas (blog exp 8 adaptado).

Sets:
  diag    : GSM8K train muestra fija (diseno de hipotesis, NUNCA holdout)
  math500 : MATH-500 muestra fija (HOLDOUT: solo eval final)
Sampling tipo blog: temp 0.6, top_p 0.95, top_k 20, max_tokens 2048, seed fija por indice.
Uso: python blog_prompts/eval_math.py <alias> <set> -p 8950 [-n 8] [-o 0]
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = Path(r"C:\weights")
DATA = ROOT / "data"
RIG = ROOT / "rig_runs"

DATA_FILE = {"diag": DATA / "gsm8k_train.jsonl", "math500": DATA / "math500_test.jsonl"}


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


def chat(server, prompt, seed, max_toks=2048):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                       "max_tokens": max_toks, "seed": seed}).encode()
    req = urllib.request.Request(server + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason")


INSTR = ("Resuelve el problema paso a paso. Termina tu respuesta con UNICA linea final: 'Respuesta: <valor>'.\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alias"); ap.add_argument("set", choices=["diag", "math500"])
    ap.add_argument("-p", "--port", type=int, default=8950)
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("-o", "--offset", type=int, default=0)
    a = ap.parse_args()
    items = load(a.set, a.n, a.offset)
    server = f"http://127.0.0.1:{a.port}"
    results = []
    for i, prob in enumerate(items):
        q, g = ask(prob)
        prompt = INSTR + q
        try:
            txt, finish = chat(server, prompt, seed=42 + i)
        except Exception as e:
            results.append({"i": a.offset + i, "err": repr(e)[:120], "ok": False})
            results_w = json.dumps(results, ensure_ascii=False, indent=1)
            (RIG / f"{a.alias}-math.json").write_text(results_w, encoding="utf-8")
            continue
        pred = extract(txt, a.set)
        g_norm = norm_ans(gold_norm(g))
        p_norm = norm_ans(pred)
        ok = (p_norm is not None) and (p_norm == g_norm)
        results.append({"i": a.offset + i, "gold": gold_norm(g)[:40], "pred": str(pred)[:60],
                        "ok": ok, "finish": finish,
                        "tail": txt[-160:].replace("\n", " | ")})
        (RIG / f"{a.alias}-math.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{a.offset+i+1}] ok={ok} gold={gold_norm(g)[:24]} pred={str(pred)[:32]}", flush=True)
    got = sum(1 for r in results if r.get("ok"))
    n_ok = sum(1 for r in results if "err" not in r)
    print(f"== {a.alias} [{a.set}]: {got}/{n_ok} ({100*got/max(n_ok,1):.1f}%)")
    (RIG / f"{a.alias}-math.json").write_text(json.dumps(results + [{"summary": {"ok": got, "n": n_ok}}], ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
