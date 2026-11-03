"""eval_code.py â€” HumanEval pass@1 para MiMo-V2.6 (medicion de programacion).
genera codigo (temp 0), extrae bloque python, ejecuta test en subprocess con timeout.
Uso: python blog_prompts/eval_code.py <alias> -p 8950 [-n 50] [-o 0]
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import shutil
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = Path(r"C:\weights")
RIG = ROOT / "rig_runs"

INSTR = """Completa la funciÃ³n Python exactamente como pide el docstring. Devuelve SOLO el cÃ³digo Python completo con la funciÃ³n requerida (sin explicaciones, sin tests), en un Ãºnico bloque ```python ... ```.

```python
{prompt}
```
"""


def chat(server, prompt, seed, temp=0.0, max_tokens=1200):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "temperature": temp, "top_p": 1.0, "max_tokens": max_tokens, "seed": seed}).encode()
    req = urllib.request.Request(server + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or "", d["choices"][0].get("finish_reason")


def extract_code(txt):
    import re
    m = re.findall(r"```python(.*?)```", txt, re.S)
    if m:
        return m[0]
    m = re.findall(r"```(.*?)```", txt, re.S)
    if m:
        return m[0]
    return txt.strip()


def run_test(code, test_code, entry_point, timeout=20):
    d = tempfile.mkdtemp(prefix="heval_")
    fn = Path(d) / "t.py"
    fn.write_text(code + "\n" + test_code + f"\ncheck({entry_point})\nprint('OK')", encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, str(fn)], capture_output=True, text=True, timeout=timeout)
        return (r.returncode == 0 and "OK" in r.stdout)
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alias")
    ap.add_argument("-p", "--port", type=int, default=8950)
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("-o", "--offset", type=int, default=0)
    a = ap.parse_args()
    maxtok = 600
    items = []
    with (ROOT / "data" / "humaneval.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(json.loads(line))
    items = items[a.offset:a.offset + a.n]
    server = f"http://127.0.0.1:{a.port}"
    results = []
    for i, p in enumerate(items):
        prompt = INSTR.format(prompt=p["prompt"])
        try:
            txt, fin = chat(server, prompt, seed=42 + i, max_tokens=600)
        except Exception as e:
            results.append({"i": a.offset + i, "err": repr(e)[:120], "pass": False})
            json.dump({"alias": a.alias, "total": len(results), "passed": sum(x.get("pass", False) for x in results), "individual": results}, (RIG / f"{a.alias}-code.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
            continue
        code = extract_code(txt)
        ok = run_test(code, p["test"], p["entry_point"])
        results.append({"i": a.offset + i, "task": p["entry_point"], "pass": ok, "finish": fin,
                        "tail": txt[-120:].replace("\n", " | ")})
        json.dump({"alias": a.alias, "total": len(results), "passed": sum(x.get("pass", False) for x in results), "individual": results}, (RIG / f"{a.alias}-code.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{a.offset+i+1}] {'PASS' if ok else 'FAIL'} {p['entry_point']}", flush=True)
    passes = sum(x.get("pass", False) for x in results)
    print(f"== {a.alias} code: {passes}/{len(results)} pass@1 ({100*passes/max(len(results),1):.1f}%)")


if __name__ == "__main__":
    main()

