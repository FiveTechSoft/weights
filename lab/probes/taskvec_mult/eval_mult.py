"""Exact-match eval of the multiplication task through llama-server. Output goes to lab/runs/taskvec_mult/fast/<name>.json
in the house layout (summary + items) so the scoring plugins can read it the same way as fast_eval runs.

usage: python lab/probes/taskvec_mult/eval_mult.py <name> <port> [--cap 512] [--parallel 4]
"""
import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "lab/runs/taskvec_mult/fast"
ANS_RE = re.compile(r"\d+")


def final_int(text: str):
    nums = ANS_RE.findall(text.replace(",", "").replace(" ", ""))
    return int(nums[-1]) if nums else None


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("port", type=int)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--parallel", type=int, default=12)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / "lab/runs/taskvec_mult/mult_test.jsonl", encoding="utf-8")]
    base = f"http://127.0.0.1:{a.port}/v1/chat/completions"

    def ask(r):
        body = {"messages": [{"role": "user", "content": r["prompt"]}], "temperature": 0, "max_tokens": a.cap}
        try:
            got = post(base, body)["choices"][0]["message"]["content"]
        except Exception as e:
            got = f"__error__ {e}"
        return {"grp": "A", "i": r["i"], "prompt": r["prompt"], "ref": r["answer"], "got": got,
                "ok": final_int(got) == r["answer"]}

    t0 = time.time()
    with ThreadPoolExecutor(a.parallel) as ex:
        items = list(ex.map(ask, rows))
    toks = []
    summ = {"name": a.name, "bench": "mult", "cap": a.cap, "seconds": round(time.time() - t0),
            "A": {"n": len(items), "closed": sum(x["ok"] for x in items), "ok": sum(x["ok"] for x in items),
                  "median_tokens": None}}
    FAST.mkdir(parents=True, exist_ok=True)
    (FAST / f"{a.name}.json").write_text(json.dumps({"summary": summ, "items": items}, indent=1),
                                         encoding="utf-8")
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
