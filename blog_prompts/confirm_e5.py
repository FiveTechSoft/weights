"""Orquestador de confirmacion P4 para E5: baseline vs E5 en sets frescos y dev, con resets de server.

Secuencia (estimacion ~2.5-3h):
  fase 11: serve original -> conf-14a-orig (n=20, offset 10)
  fase 21: serve E5 -> conf-e5 (mismo set)
  fase 31: serve original -> conf-dev-orig (n=10 offset 0)
  fase 41: serve E5 -> conf-dev-e5 (n=10 offset 0)
Y final: comparacion against baseline original n=10 ya existente e informe.

Uso: python blog_prompts/confirm_e5.py -p 8950
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\weights")
RIG = ROOT / "rig_runs"
sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

M_ORIG = r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf"
M_E5 = r"C:\weights\servidos\mimo-e5.gguf"
PORT = 8950


def serve(model_path, alias):
    stop_service()
    cmd = [r"C:\Users\Anto\llamacpp\bin-b11962\llama-server.exe", "-m", model_path,
           "--port", str(PORT), "--host", "127.0.0.1", "-a", alias,
           "-c", "16384", "-ngl", "999", "--jinja"]
    log = open(ROOT / r"logs" / f"llama-{PORT}.log", "ab")
    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=0x00000008)
    for _ in range(60):
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


def stop_service():
    p = subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, text=True)
    time.sleep(2)


def eval_run(script, rest):
    r = subprocess.run(["python", str(ROOT / "blog_prompts" / script), *rest],
                       capture_output=False, text=True, timeout=7200)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", type=int, default=PORT)
    args = ap.parse_args()
    cfg = [
        ("mimo-orig", M_ORIG, "conf-fresca-orig", ["harness2.py", "conf-fresca-orig", "20", "10", str(PORT)]),
        ("mimo-e5", M_E5, "conf-fresca-e5", ["harness2.py", "conf-fresca-e5", "20", "10", str(PORT)]),
        ("mimo-orig", M_ORIG, "conf-dev-orig", ["harness2.py", "conf-dev-orig", "10", "0", str(PORT)]),
        ("mimo-e5", M_E5, "conf-dev-e5", ["harness2.py", "conf-dev-e5", "10", "0", str(PORT)]),
    ]
    informe = {}
    for server_alias, model_path, run_alias, harness_args in cfg:
        print(f"=== serviendo {server_alias} para {run_alias} ===", flush=True)
        if not serve(model_path, server_alias):
            print("servidor no levanto, abortando", flush=True)
            sys.exit(1)
        rc = eval_run(harness_args[0], harness_args[1:])
        informe[run_alias] = (server_alias, rc)
    stop_service()
    sett = {}
    for alias in ("conf-fresca-orig", "conf-fresca-e5", "conf-dev-orig", "conf-dev-e5"):
        rs = json.loads((RIG / f"{alias}.json").read_text(encoding="utf-8"))
        nn = [r for r in rs if "loop" in r]
        divs = [r.get("diversity") for r in nn if r.get("diversity") is not None]
        loops = sum(1 for r in nn if r.get("loop"))
        finals = sum(1 for r in nn if r.get("final"))
        sett[alias] = {"modelo": informe.get(alias, ("?",))[0], "loops": loops, "n": len(nn)}
        print(f"== {alias} ({informe.get(alias, ('?',))[0]}): loops={loops}/{len(nn)} finales={finals} "
              f"div_med={sum(divs)/max(len(divs),1):.2f}", flush=True)
    json.dump(sett, open(RIG / "confirm_summary.json", "w"), indent=1)
    print("LISTO: confirmacion P4", flush=True)


if __name__ == "__main__":
    main()
