"""harness2 — rig agentic con metricas extendidas (fase confirmacion P4).

Igual protocolo que mock_harness (antml, MAX_TURNS 12, MOCK determinista) + por prompt:
  seed per-prompt (42+i para reproducibilidad A/B), historial de llamadas normalizadas,
  diversidad = len(set(hist))/max(n,1), y finish reasons acumulados.

Uso: python harness2.py <alias> <n> <offset> <port>
ESCRIBE: rig_runs/<alias>.json  (esquema compatible + campos nuevos: calls, diversity)
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

PROMPTS_FILE = Path(r"C:\temp\opencode\rl_oss_prompts.json")
OUT_DIR = Path(r"C:\weights\rig_runs")
MAX_TURNS = 12
MAX_TOKENS_TURN = 800
MOCK_RESULT = ("[mock harness] executed: ok. stdout captured: total 8\n"
               "drwxr-xr-x .  .. .git src config README.md")

RE_ABS = re.compile(
    r"<function=([a-zA-Z0-9_]+)>\s*<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>\s*</function>",
    re.S)
RE_ABS_TRUNC = re.compile(r"<function=([a-zA-Z0-9_]+)>\s*<parameter=([a-zA-Z0-9_]+)>(.*)", re.S)


def chat(server, messages, seed):
    body = json.dumps({"messages": messages, "temperature": 0.3,
                       "top_p": 0.95, "max_tokens": MAX_TOKENS_TURN, "seed": seed}).encode()
    req = urllib.request.Request(server + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("content") or "", d["choices"][0].get("finish_reason"))


def parse_calls(text):
    out = []
    for m in RE_ABS.finditer(text):
        out.append(f"{m.group(1).lower()}:{' '.join(m.group(3).split())}")
    if not out:
        for m in RE_ABS_TRUNC.finditer(text):
            out.append(f"{m.group(1).lower()}:{' '.join(m.group(3).split())[:80]}")
    return out


def run_task(server, prompt, seed):
    messages = [{"role": "user", "content": prompt}]
    hist, finishes, final_txt = [], [], None
    for _ in range(MAX_TURNS):
        try:
            txt, fin = chat(server, messages, seed)
        except Exception as e:
            return {"turns": len(finishes) + 1, "error": repr(e)[:150], "final": False,
                    "loop": False, "calls": hist, "diversity": None, "finishes": finishes}
        finishes.append(fin)
        calls = parse_calls(txt)
        if not calls:
            final_txt = txt[:400]
            break
        hist.extend(calls)
        resp = "\n".join(f"Result for `{c.split(':',1)[0]}`: {MOCK_RESULT}" for c in calls)
        messages.append({"role": "assistant", "content": txt})
        messages.append({"role": "user", "content": f"Tool results:\n{resp}\nContinue."})
    last3 = hist[-3:]
    cnt = {}
    for c in hist:
        cnt[c] = cnt.get(c, 0) + 1
    loop3 = len(last3) == 3 and len(set(last3)) == 1
    loop4 = any(v >= 4 for v in cnt.values())
    why = []
    if loop3:
        why.append("3-seguidas-iguales")
    if loop4:
        top = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))
        why.append(f"'{top[0][:50]}' x{top[1]}")
    div = round(len(set(hist)) / max(len(hist), 1), 2)
    return {"turns": len(messages) // 2, "final": final_txt is not None,
            "n_calls": len(hist), "calls": hist, "diversity": div,
            "loop": bool(loop3 or loop4), "why": why, "finishes": finishes}


def main():
    alias = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    port = sys.argv[4] if len(sys.argv) > 4 else "8950"
    server = f"http://127.0.0.1:{port}"
    prompts = json.load(PROMPTS_FILE.open(encoding="utf-8"))[offset:offset + n]
    results = []
    for i, p in enumerate(prompts):
        r = run_task(server, p["prompt"], seed=42 + offset + i)
        r.update({"id": p["id"], "config": p.get("config")})
        results.append(r)
        print(f"[{offset+i+1}] {p['id']}: loop={r['loop']} calls={r.get('n_calls')} div={r.get('diversity')}", flush=True)
        (OUT_DIR / f"{alias}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [r for r in results if "loop" in r]
    loops = sum(1 for r in ok if r["loop"])
    finals = sum(1 for r in ok if r.get("final"))
    divs = [r["diversity"] for r in ok if r.get("diversity") is not None]
    print(f"== {alias}: loops {loops}/{len(ok)} finales={finals} diversidad_med={sum(divs)/max(len(divs),1):.2f}", flush=True)


if __name__ == "__main__":
    main()
