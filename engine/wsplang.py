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

OPS_ALLOWED = {"inspect", "observe", "compare", "hypothesize", "localize", "peek",
               "intervene", "evaluate", "confirm", "poke", "poke-weight", "rollback",
               "record", "select_candidate"}
TYPES = {"Model", "Tensor", "Activation", "Direction", "Behavior", "Hypothesis",
         "Intervention", "Benchmark", "Experiment", "Result"}
INT_OPS = {"project_out", "activation_projection", "poke-weight", "scale", "ablate",
           "scale-rows", "scale-cols", "blend-rows"}


def parse_args_lines(text):
    """parser de secciones del estilo WSP-Lang (bloques y asignaciones simples).

    La cabecera `experiment <nombre>[: ]` va a raiz (no crea seccion), para que
    validate() pueda extraer el nombre del experiment. Las secciones (model:,
    objective:, steps:, ...) se reconocen por su nombre y van a su propia seccion,
    con su indent registrado para saber si estan anidadas.
    """
    prog = {}
    current = None
    current_indent = None
    buf = prog.setdefault("__root__", [])
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()
        if indent == 0 and s.startswith("experiment"):
            # cabecera del experiment: va a raiz, no es una seccion
            name = re.sub(r"^experiment\s+", "", s).strip().rstrip(":").strip()
            if name:
                prog.setdefault("__root__", []).append(("exp", name))
            continue
        m = re.match(r"^(\S+)\s*:\s*$", s)
        if m:
            sec = m.group(1)
            if sec == current and current_indent == indent:
                # cabecera duplicada: se ignora para no duplicar la seccion
                continue
            current = sec
            current_indent = indent
            prog.setdefault(sec, [])
            continue
        if current is None:
            # fuera de cualquier seccion: ignorado
            continue
        prog[current].append((indent, s))
    return prog


METADATA = {"model", "objective", "constraints", "types", "declared", "sources",
            "baseline", "discover", "accept_if", "evaluate", "record", "hypothesis",
            "note"}

def validate(prog):
    errors = []
    steps = []
    # raiz
    root = prog.get("__root__", [])
    experiment = None
    for ind, s in root:
        if isinstance(s, str) and s.startswith("experiment"):
            experiment = s.split("experiment", 1)[1].strip().rstrip(":").strip()
        elif ind == "exp":
            experiment = s
    if not experiment:
        errors.append("falta experiment <nombre>")
    # secciones con ops
    for sec, buf in prog.items():
        if sec == "__root__":
            continue
        if sec not in OPS_ALLOWED and sec not in METADATA and sec != "steps":
            errors.append(f"seccion desconocida: {sec}")
        for ind, s in buf:
            m = re.match(r"^(\w[\w_]+)\s*[=(]", s)
            if m:
                fn = m.group(1)
                if sec == "intervene":
                    if fn not in INT_OPS:
                        errors.append(f"op de intervencion no permitida: {fn}")
                steps.append({"sec": sec, "expr": s[:120]})
            elif s.startswith("step "):
                steps.append({"sec": sec, "expr": s[:120]})
            elif sec not in METADATA:
                errors.append(f"linea ignorada {sec}: {s[:50]}")
    return steps, errors


def plan_run(prog, steps):
    """map de secciones a comandos del runtime; manifest con huellas"""
    mapping = {
        "inspect": "wsp tensors/stats",
        "observe": "blog_prompts/capture_residual.py o capture_heads.py (capture=activations|residuos)",
        "compare": "wsp delta / score de activacion (posiciones rep/pro)",
        "hypothesize": "expediente retos/expedientes/<id>.json (campos obligatorios)",
        "localize": "wsp delta -k + head_scores/activation scores",
        "peek": "wsp peek <tensor> [row=] [col=] [k=]  (anatomia del tensor: normas fila/columna, rank, frac)",
        "poke": "wsp poke <tensor> <row> <col> <valor> [out=] (poke de un solo peso, requant solo bloque)",
        "poke-weight": "wsp poke <tensor> <row> <col> <valor> (poke de peso individual; requantiza solo el bloque)",
        "intervene": "wsp edit ediciones/<id>.json (POLICY enforced)",
        "evaluate": "wsp eval <alias> -n 20 (bg) + wsp vs",
        "confirm": "challenge.py sellar (LOCK) + veredicto keep/discard",
        "rollback": "no usar servidos/<id>.gguf (modelo original intacto)",
        "record": "journal + expediente",
        "select_candidate": "challenge.py historial",
    }

    # 1) pasos de la seccion "steps": cada uno es "step <op> <args...>"
    step_cmds = []
    for st in steps:
        if st["sec"] == "steps":
            m = re.match(r"^step\s+(\S+)\s*(.*)$", st["expr"].strip())
            if m:
                step_cmds.append({"op": m.group(1), "args": m.group(2).strip()})
    # 2) asignaciones de secciones de op (intervene, evaluate, ...)
    op_assigns = []
    for sec, buf in prog.items():
        if sec in OPS_ALLOWED and sec != "steps":
            for ind, s in buf:
                m = re.match(r"^(\w[\w_]+)\s*[=(]", s)
                if m and m.group(1) in mapping:
                    op_assigns.append({"op": m.group(1), "args": s[m.end():].strip()})
    # 3) conteo por seccion
    sec_counts = {}
    for st in steps:
        sec_counts[st["sec"]] = sec_counts.get(st["sec"], 0) + 1
    # 4) mapa de paso -> comando: cada paso "step <op> ..." usa el mapa de ops
    paso_mapeo = {p["op"]: mapping.get(p["op"]) for p in step_cmds}
    plan = {
        "secciones": sec_counts,
        "pasos": step_cmds,
        "pasos_mapeo": paso_mapeo,
        "ops": op_assigns,
        "mapeo": {s: mapping.get(s) for s in sec_counts if s in mapping},
    }
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
