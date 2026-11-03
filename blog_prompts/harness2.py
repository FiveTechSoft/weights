"""harness2 — rig agentic con metricas extendidas y soporte de remediaciones.

Igual protocolo que mock_harness (antml, MAX_TURNS 12) + por prompt:
  seed per-prompt (42+i para reproducibilidad A/B), historial de llamadas normalizadas,
  diversidad = len(set(hist))/max(n,1), y finish reasons acumulados.

Soporte de remediaciones anti-bucle:
  --sys-guard          : inyecta regla de parada/no-repeticion en system prompt.
  --presence-penalty P : penalizacion de presencia en /v1/chat/completions (default 0.0).
  --repeat-penalty R   : penalizacion de repeticion (default 1.0).
  --realistic-feedback : devuelve 'No such file or directory' si el comando pide archivos inexistentes.

Uso flexible:
  python harness2.py <alias> [n] [offset] [port]
  python harness2.py --alias <alias> -n 10 --offset 0 --port 8081 --sys-guard
ESCRIBE: rig_runs/<alias>.json
"""
import argparse
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
DEFAULT_MOCK = ("[mock harness] executed: ok. stdout captured: total 8\n"
                "drwxr-xr-x .  .. .git src config README.md")

SYS_GUARD_MSG = (
    "You are an expert autonomous software engineer.\n"
    "Strict rule: Never execute the exact same bash command more than once. "
    "If a file, path, or pattern is not found after checking, do not keep repeating ls or cat on that path. "
    "Conclude that it does not exist, switch exploration strategy, or produce your final repair explanation."
)

RE_ABS = re.compile(
    r"<function=([a-zA-Z0-9_]+)>\s*<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>\s*</function>",
    re.S)
RE_ABS_TRUNC = re.compile(r"<function=([a-zA-Z0-9_]+)>\s*<parameter=([a-zA-Z0-9_]+)>(.*)", re.S)


def chat(server, messages, seed, presence_penalty=0.0, repeat_penalty=1.0):
    gen_params = {
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": MAX_TOKENS_TURN,
        "seed": seed,
        "presence_penalty": presence_penalty,
        "repeat_penalty": repeat_penalty
    }
    body = json.dumps({"messages": messages, **gen_params}).encode()
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


def mock_response_for(call, realistic=False):
    if not realistic:
        return DEFAULT_MOCK
    cmd = call.split(":", 1)[1] if ":" in call else call
    # Si el comando busca algo especifico no existente en el root mock:
    if any(k in cmd for k in ["snakemake", "workflow_probe", "robotics", "IK.py", "ik.cpp", "bandit"]):
        return f"[mock harness] executed: exit code 1.\nbash: {cmd.split()[0] if cmd.split() else 'file'}: No such file or directory"
    if "EXIT:$?" in cmd or "echo $?" in cmd:
        return "[mock harness] executed: ok. stdout captured:\nEXIT:1"
    return DEFAULT_MOCK


def run_task(server, prompt, seed, sys_guard=False, presence_penalty=0.0, repeat_penalty=1.0, realistic=False):
    messages = []
    if sys_guard:
        messages.append({"role": "system", "content": SYS_GUARD_MSG})
    messages.append({"role": "user", "content": prompt})

    hist, finishes, final_txt = [], [], None
    for _ in range(MAX_TURNS):
        try:
            txt, fin = chat(server, messages, seed, presence_penalty=presence_penalty, repeat_penalty=repeat_penalty)
        except Exception as e:
            return {"turns": len(finishes) + 1, "error": repr(e)[:150], "final": False,
                    "loop": False, "calls": hist, "diversity": None, "finishes": finishes}
        finishes.append(fin)
        calls = parse_calls(txt)
        if not calls:
            final_txt = txt[:400]
            break
        hist.extend(calls)
        resp = "\n".join(f"Result for `{c.split(':',1)[0]}`: {mock_response_for(c, realistic)}" for c in calls)
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


def parse_args():
    # Detectar si se llama con sintaxis legacy: python harness2.py <alias> [n] [offset] [port]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        alias = sys.argv[1]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        port = sys.argv[4] if len(sys.argv) > 4 else "8081"
        return argparse.Namespace(alias=alias, n=n, offset=offset, port=port,
                                  sys_guard=False, presence_penalty=0.0, repeat_penalty=1.0,
                                  realistic_feedback=False)

    ap = argparse.ArgumentParser(description="harness2 — rig agentic multi-turno")
    ap.add_argument("alias", nargs="?", default="eval_run")
    ap.add_argument("-n", type=int, default=10, help="Numero de prompts")
    ap.add_argument("-o", "--offset", type=int, default=0, help="Offset en dataset")
    ap.add_argument("-p", "--port", type=str, default="8081", help="Puerto del server llama.cpp")
    ap.add_argument("--sys-guard", action="store_true", help="Inyecta regla anti-bucle en system prompt")
    ap.add_argument("--presence-penalty", type=float, default=0.0, help="Presencia penalty (0.0-2.0)")
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="Repeat penalty (1.0 = deshabilitado)")
    ap.add_argument("--realistic-feedback", action="store_true", help="Feedback con errores realistas en mock")
    return ap.parse_args()


def main():
    args = parse_args()
    server = f"http://127.0.0.1:{args.port}"
    prompts = json.load(PROMPTS_FILE.open(encoding="utf-8"))[args.offset:args.offset + args.n]
    results = []

    print(f"== Iniciando harness2 '{args.alias}' en {server} (n={args.n}, offset={args.offset}) ==", flush=True)
    if args.sys_guard:
        print("   [Remediacion] System Prompt Guard ACTIVO", flush=True)
    if args.presence_penalty > 0.0:
        print(f"   [Remediacion] Presence Penalty: {args.presence_penalty}", flush=True)
    if args.repeat_penalty != 1.0:
        print(f"   [Remediacion] Repeat Penalty: {args.repeat_penalty}", flush=True)
    if args.realistic_feedback:
        print("   [Remediacion] Feedback Realista ACTIVO", flush=True)

    for i, p in enumerate(prompts):
        r = run_task(
            server=server,
            prompt=p["prompt"],
            seed=42 + args.offset + i,
            sys_guard=args.sys_guard,
            presence_penalty=args.presence_penalty,
            repeat_penalty=args.repeat_penalty,
            realistic=args.realistic_feedback
        )
        r.update({"id": p["id"], "config": p.get("config")})
        results.append(r)
        print(f"[{args.offset+i+1}] {p['id']}: loop={r['loop']} calls={r.get('n_calls')} div={r.get('diversity')}", flush=True)

        # Guardar incremental
        (OUT_DIR / f"{args.alias}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    ok = [r for r in results if "loop" in r]
    loops = sum(1 for r in ok if r["loop"])
    finals = sum(1 for r in ok if r.get("final"))
    divs = [r["diversity"] for r in ok if r.get("diversity") is not None]
    print(f"== RESULTADO {args.alias}: loops {loops}/{len(ok)} ({loops/max(len(ok),1)*100:.1f}%) "
          f"finales={finals} div_med={sum(divs)/max(len(divs),1):.2f} ==", flush=True)


if __name__ == "__main__":
    main()
