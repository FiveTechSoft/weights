"""agent_loop_probe.py — Rig de evaluación para patología de bucles agénticos en MiMo-V2.6.

Evalúa llamadas repetitivas a herramientas (tool-call loops) bajo distintos samplers de inferencia
(temperatura, repetition_penalty, presence_penalty, frequency_penalty).
Soporta cálculo de intervalos de confianza de Wilson (95%) y métrica continua de turnos.
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR_DATA = ROOT / "data" / "rl_oss_prompts.json"
DIR_OUT = ROOT / "rig_runs"
DIR_OUT.mkdir(parents=True, exist_ok=True)

MOCK_RESULT = ("[mock harness] executed: ok. stdout captured: total 8\n"
               "drwxr-xr-x .  .. .git src config README.md\n")

RE_ABS = re.compile(
    r"<function=([a-zA-Z0-9_]+)>\s*<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>\s*</function>",
    re.S)
RE_ABS_TRUNC = re.compile(r"<function=([a-zA-Z0-9_]+)>\s*<parameter=([a-zA-Z0-9_]+)>(.*)", re.S)


def wilson_interval(k, n, z=1.96):
    """Calcula el intervalo de confianza de Wilson al 95% para proporciones binomiales."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def parse_calls(text):
    out = []
    for m in RE_ABS.finditer(text):
        fn = m.group(1).lower()
        body = m.group(3)
        out.append((fn, body))
    if not out:
        for m in RE_ABS_TRUNC.finditer(text):
            out.append((m.group(1).lower(), m.group(3)))
    return out


def norm_calls(calls):
    return [f"{fn}:{' '.join(args.split())}" for fn, args in calls]


def chat(endpoint, messages, sampling_params):
    payload = {
        "messages": messages,
        "temperature": sampling_params.get("temperature", 0.3),
        "top_p": sampling_params.get("top_p", 0.95),
        "max_tokens": sampling_params.get("max_tokens", 800),
        "seed": sampling_params.get("seed", 42),
    }
    # Pass repetition / penalty params if set
    if "repetition_penalty" in sampling_params and sampling_params["repetition_penalty"] != 1.0:
        payload["repetition_penalty"] = sampling_params["repetition_penalty"]
    if "presence_penalty" in sampling_params and sampling_params["presence_penalty"] != 0.0:
        payload["presence_penalty"] = sampling_params["presence_penalty"]
    if "frequency_penalty" in sampling_params and sampling_params["frequency_penalty"] != 0.0:
        payload["frequency_penalty"] = sampling_params["frequency_penalty"]

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    m = data["choices"][0]["message"]
    return (m.get("content") or "", data["choices"][0].get("finish_reason"))


def run_task(endpoint, prompt, sampling_params, max_turns=12):
    messages = [{"role": "user", "content": prompt}]
    hist, turns, final = [], 0, None
    loop_turn = None

    for t in range(1, max_turns + 1):
        turns = t
        try:
            txt, finish = chat(endpoint, messages, sampling_params)
        except Exception as e:
            return {
                "turns": turns,
                "error": repr(e)[:150],
                "final": False,
                "loop": False,
                "why": [],
                "n_calls": 0,
                "loop_turn": None
            }

        calls = norm_calls(parse_calls(txt))
        if not calls:
            final = txt[:400]
            break

        hist.extend(calls)
        resp = "\n".join(f"Result for `{c.split(':', 1)[0]}`: {MOCK_RESULT}" for c in calls)
        messages.append({"role": "assistant", "content": txt})
        messages.append({"role": "user", "content": f"Tool results:\n{resp}\nContinue."})

        # Check loop condition after each turn
        last3 = hist[-3:]
        loop3 = len(last3) == 3 and len(set(last3)) == 1
        cnt = {}
        for c in hist:
            cnt[c] = cnt.get(c, 0) + 1
        loop4 = any(v >= 4 for v in cnt.values())

        if (loop3 or loop4) and loop_turn is None:
            loop_turn = turns

    cnt = {}
    for c in hist:
        cnt[c] = cnt.get(c, 0) + 1
    last3 = hist[-3:]
    loop3 = len(last3) == 3 and len(set(last3)) == 1
    loop4 = any(v >= 4 for v in cnt.values())
    is_loop = bool(loop3 or loop4)

    why = []
    if loop3:
        why.append("3-consecutive-identical")
    if loop4:
        top = max(cnt.items(), key=lambda kv: kv[1])
        why.append(f"'{top[0][:50]}' x{top[1]}")

    return {
        "turns": turns,
        "final": final is not None,
        "n_calls": len(hist),
        "loop": is_loop,
        "loop_turn": loop_turn if is_loop else None,
        "why": why
    }


def main():
    ap = argparse.ArgumentParser(description="Probe de bucles agénticos con métricas continuas e inferencia estándar.")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--alias", default="baseline-vanilla")
    ap.add_argument("--n", type=int, default=10, help="Número de prompts a evaluar")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42, help="Semilla para muestreo estocástico")
    args = ap.parse_args()

    if not DIR_DATA.exists():
        sys.exit(f"Dataset de prompts no encontrado en {DIR_DATA}")

    all_prompts = json.load(DIR_DATA.open(encoding="utf-8"))
    prompts = all_prompts[args.offset:args.offset + args.n]

    sampling_params = {
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_tokens": 800,
        "seed": args.seed,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
    }

    out_file = DIR_OUT / f"{args.alias}.json"
    print(f"=== AGENT LOOP PROBE: {args.alias} ===")
    print(f"Endpoint: {args.endpoint}")
    print(f"Prompts: {len(prompts)} (offset={args.offset})")
    print(f"Sampling: temp={args.temperature}, rep_penalty={args.repetition_penalty}, pres_penalty={args.presence_penalty}, freq_penalty={args.frequency_penalty}\n")

    results = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        r = run_task(args.endpoint, p["prompt"], sampling_params)
        r.update({"id": p.get("id"), "config": p.get("config")})
        results.append(r)
        status_str = f"LOOP ({','.join(r['why'])}) at turn {r['loop_turn']}" if r["loop"] else "NO-LOOP (finished)"
        print(f"[{args.offset + i + 1}/{args.offset + len(prompts)}] {p.get('id', 'prompt')}: {status_str} | turns={r['turns']} calls={r['n_calls']}", flush=True)

        # Write intermediate
        out_file.write_text(json.dumps({
            "meta": {
                "alias": args.alias,
                "endpoint": args.endpoint,
                "n": args.n,
                "offset": args.offset,
                "sampling_params": sampling_params,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            },
            "results": results
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    ok_runs = [r for r in results if "error" not in r]
    loops = sum(1 for r in ok_runs if r["loop"])
    loop_turns = [r["loop_turn"] for r in ok_runs if r["loop_turn"] is not None]
    mean_loop_turn = sum(loop_turns) / len(loop_turns) if loop_turns else None
    low, high = wilson_interval(loops, len(ok_runs))

    print(f"\n==========================================")
    print(f"RESUMEN: {args.alias}")
    print(f"  Total evaluados: {len(ok_runs)} (errores: {len(results) - len(ok_runs)})")
    print(f"  Loops detectados: {loops}/{len(ok_runs)} ({loops / max(len(ok_runs), 1):.1%})")
    print(f"  Intervalo Wilson 95%: [{low:.1%}, {high:.1%}]")
    if mean_loop_turn:
        print(f"  Métrica continua (turnos medios hasta bucle): {mean_loop_turn:.2f}")
    print(f"  Tiempo total: {elapsed:.1f} s")
    print(f"  Resultados guardados en: {out_file}")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
