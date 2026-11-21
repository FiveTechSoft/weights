"""Side by side: Qwen3.5-9B base vs the same base + the adapter WSPLang's delta_to_lora extracted from a fine-tune
(Jackrong's distill), on MATH-500 problems where the base reasons until it is cut off.

Needs a llama-server with the base GGUF and the adapter loaded as --lora <adapter.gguf> --lora-init-without-apply
(see README.md). Each problem is asked twice on the same server: adapter scale 0 (= the base) and adapter scale s.
Greedy, same prompt and grader as lab/probes/close_probe/fast_eval.py.

usage: python demos/math_transplant/demo.py <port> [--items 106,185,289] [--scale 0.5] [--cap 2048] [--out f.json]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rig"))
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def ask(port, problem, answer, scale, cap):
    import math500 as m5
    body = {"messages": [{"role": "user", "content": problem + SUFFIX}], "temperature": 0, "seed": 0,
            "max_tokens": cap, "lora": [{"id": 0, "scale": scale}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    ch = r["choices"][0]
    content = ch["message"].get("content") or ""
    pred = m5.extract_pred(content)
    return {"closed": ch["finish_reason"] == "stop", "ok": bool(pred) and m5.equivalent(pred, answer),
            "tokens": r["usage"]["completion_tokens"], "pred": pred, "tail": content[-400:]}


def _cell(s, cap):
    if not s["closed"]:
        return f"cut at {cap}, no answer" if not s["pred"] else f"cut at {cap}, last: {s['pred']}"
    return f"{'RIGHT' if s['ok'] else 'wrong'} {s['pred']} ({s['tokens']} tok)"


def render(rows, cap=2048):
    w = 34
    lines = [f"{'MATH-500 #':<11}{'expected':<10}{'base':<{w}}{'base + transplant':<{w}}", "-" * (21 + 2 * w)]
    for r in rows:
        lines.append(f"{r['i']:<11}{str(r['answer'])[:9]:<10}{_cell(r['base'], cap)[:w - 1]:<{w}}"
                     f"{_cell(r['lora'], cap)[:w - 1]:<{w}}")
    nb, nl = sum(r["base"]["ok"] for r in rows), sum(r["lora"]["ok"] for r in rows)
    lines.append(f"\nright: base {nb}/{len(rows)}   transplant {nl}/{len(rows)}")
    return "\n".join(lines)


def main():
    import math500 as m5
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--items", default="106,185,289")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--out")
    a = ap.parse_args()
    items = m5.load_items()
    rows = []
    for i in (int(x) for x in a.items.split(",")):
        it = items[i]
        print(f"#{i}: {it['problem'][:100]}...", flush=True)
        rows.append({"i": i, "answer": it["answer"], "base": ask(a.port, it["problem"], it["answer"], 0.0, a.cap),
                     "lora": ask(a.port, it["problem"], it["answer"], a.scale, a.cap)})
    print()
    print(render(rows, a.cap))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
