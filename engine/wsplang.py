"""wsplang.py — parser + runtime MVP de WSP-Lang 0.1.

Convierte un programa declarativo (secciones experiment/model/objective/constraints/
baseline/observe/compare/hypothesize/localize/intervene/evaluate/accept_if) en un
PLAN estructurado, valida tipos/ops permitidas, y genera manifiesto + instrucciones
de ejecucion (mapea a las herramientas existentes del repo: wsp.py, capture_*, challenge).

CLI:  python wsplang.py programas/e2.wspl   -> JSON plan + manifiesto en retos/wsplang/
"""
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
ROOT = Path(r"C:\weights")
DIR_PL = ROOT / "retos" / "wsplang"
DIR_PL.mkdir(parents=True, exist_ok=True)

OPS_ALLOWED = {"inspect", "observe", "compare", "hypothesize", "localize",
               "intervene", "evaluate", "confirm", "rollback", "record", "select_candidate"}
TYPES = {"Model", "Tensor", "Activation", "Direction", "Behavior", "Hypothesis",
         "Intervention", "Benchmark", "Experiment", "Result"}
INT_OPS = {"project_out", "activation_projection", "scale", "ablate", "scale-rows", "scale-cols", "blend-rows"}


def parse_args_lines(text):
    """parser de secciones del estilo WSP-Lang (bloques y asignaciones simples)."""
    prog = {}
    current = None
    buf = prog.setdefault("__root__", [])
    lines = []
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()
        if indent == 0 and (s.endswith(":")):
            current = s[:-1].strip()
            buf = prog.setdefault(current, [])
        else:
            buf.append((indent, s))
    return prog


def validate(prog):
    errors = []
    steps = []
    # raiz
    root = prog.get("__root__", [])
    experiment = None
    for ind, s in root:
        if s.startswith("experiment"):
            experiment = s.split("experiment", 1)[1].strip().rstrip(":").strip()
    if not experiment:
        errors.append("falta experiment <nombre>")
    # secciones con ops
    for sec, buf in prog.items():
        if sec == "__root__":
            continue
        if sec not in OPS_ALLOWED and sec not in ("model", "objective", "constraints",
                                                  "types", "declared", "sources", "baseline"):
            errors.append(f"seccion desconocida: {sec}")
        for ind, s in buf:
            m = re.match(r"^(\w[\w_]+)\s*[=(]", s)
            if not m:
                errors.append(f"linea ignurada {sec}: {s[:50]}")
                continue
            fn = m.group(1)
            if sec in ("intervene",):
                if fn not in INT_OPS:
                    errors.append(f"op de intervencion no permitida: {fn}")
            steps.append({"sec": sec, "expr": s[:120]})
    return steps, errors


def plan_run(prog, steps):
    """map de secciones a comandos del runtime; manifest con huellas"""
    mapping = {
        "inspect": "wsp tensors/stats",
        "observe": "blog_prompts/capture_residual.py o capture_heads.py (capture=activations|residuos)",
        "compare": "wsp delta / score de activacion (posiciones rep/pro)",
        "hypothesize": "expediente retos/expedientes/<id>.json (campos obligatorios)",
        "localize": "wsp delta -k + head_scores/activation scores",
        "intervene": "wsp edit ediciones/<id>.json (POLICY enforced)",
        "evaluate": "wsp eval <alias> -n 20 (bg) + wsp vs",
        "confirm": "challenge.py sellar (LOCK) + veredicto keep/discard",
        "rollback": "no usar servidos/<id>.gguf (modelo original intacto)",
        "record": "journal + expediente",
        "select_candidate": "challenge.py historial",
    }
    sec_counts = {}
    for st in steps:
        sec_counts[st["sec"]] = sec_counts.get(st["sec"], 0) + 1
    plan = {"secciones": sec_counts, "mapeo": [mapping.get(s, None) for s in sec_counts]}
    return plan


def main():
    src = Path(sys.argv[1])
    text = src.read_text(encoding="utf-8")
    prog = parse_args_lines(text)
    steps, errors = validate(prog)
    experiment = None
    for ind, s in prog.get("__root__", []):
        if s.startswith("experiment"):
            experiment = s.split("experiment", 1)[1].strip().rstrip(":").strip()
    plan = plan_run(prog, steps)
    out = {
        "programa": experiment,
        "archivo": str(src),
        "pasos": len(steps),
        "steps": steps,
        "plan": plan,
        "errores": errors,
        "validado": len(errors) == 0,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manif = DIR_PL / f"{experiment or 'sin-nombre'}.manifest.json"
    manif.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
