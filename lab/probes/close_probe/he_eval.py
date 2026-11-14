"""Does the closing LoRA generalize outside math? HumanEval pass@1 on the first N problems, greedy, short cap.
Code = last ```python block of the answer (prompt prepended if it lacks `def <entry_point>`), run with the official
check() in a subprocess with a timeout, in a temp dir.

usage: python lab/probes/close_probe/he_eval.py <name> <port> [--n 20] [--cap 2048] [--lora "id:scale,..."]
"""
import argparse
import concurrent.futures
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def build_program(prompt, content, test, entry):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, re.S)
    if not blocks:
        return ""
    code = blocks[-1]
    if f"def {entry}" not in code:
        code = prompt + code
    elif "import" in prompt:   # keep the prompt's imports (typing etc.)
        code = "\n".join(l for l in prompt.splitlines() if l.startswith(("import", "from"))) + "\n" + code
    return f"{code}\n\n{test}\ncheck({entry})\n"


def run_program(src, timeout=10):
    if not src:
        return False
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "p.py"
        f.write_text(src, encoding="utf-8")
        try:
            return subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True,
                                  timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("port", type=int)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--lora", default="")
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()
    probs = [json.loads(l) for l in (ROOT / "data/humaneval.jsonl").read_text(encoding="utf-8").splitlines()][:a.n]

    def one(p):
        msg = ("Complete the following Python function. Return the full function in a ```python code block.\n\n"
               f"```python\n{p['prompt']}```")
        body = {"messages": [{"role": "user", "content": msg}], "temperature": 0, "seed": 0, "max_tokens": a.cap}
        if a.lora:
            body["lora"] = [{"id": int(x.split(":")[0]), "scale": float(x.split(":")[1])} for x in a.lora.split(",")]
        r = post(f"http://127.0.0.1:{a.port}/v1/chat/completions", body)
        ch = r["choices"][0]
        content = ch["message"].get("content") or ""
        return {"task": p["task_id"], "closed": ch["finish_reason"] == "stop",
                "ok": run_program(build_program(p["prompt"], content, p["test"], p["entry_point"])),
                "tokens": r["usage"]["completion_tokens"], "content_tail": content[-300:]}

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.parallel) as ex:
        rows = list(ex.map(one, probs))
    summ = {"name": a.name, "n": len(rows), "closed": sum(r["closed"] for r in rows), "ok": sum(r["ok"] for r in rows),
            "median_tokens": sorted(r["tokens"] for r in rows)[len(rows) // 2], "lora": a.lora,
            "seconds": round(time.time() - t0)}
    out = ROOT / "lab/runs/close_probe/fast"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"he_{a.name}.json").write_text(json.dumps({"summary": summ, "items": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
