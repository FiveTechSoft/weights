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
ROOT = Path(__file__).resolve().parents[1]
DIR_PL = ROOT / "challenges" / "wsplang"
DIR_PL.mkdir(parents=True, exist_ok=True)

OPS_ALLOWED = {"inspect", "observe", "compare", "hypothesize", "localize", "peek",
               "intervene", "evaluate", "confirm", "poke", "poke-weight", "rollback",
               "record", "select_candidate", "semantic_map", "semantic-map", "semantic_areas", "semantic-areas"}
TYPES = {"Model", "Tensor", "Activation", "Direction", "Behavior", "Hypothesis",
         "Intervention", "Benchmark", "Experiment", "Result", "SemanticArea", "SemanticCentroid"}
INT_OPS = {"project_out", "activation_projection", "poke-weight", "scale", "ablate",
           "scale-rows", "scale-cols", "blend-rows", "cov-steer"}


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
            # fuera de cualquier seccion: registrar en __root__ para reporte de validate()
            prog.setdefault("__root__", []).append((indent, s))
            continue
        prog[current].append((indent, s))
    return prog


METADATA = {"model", "objective", "constraints", "types", "declared", "sources",
            "baseline", "discover", "accept_if", "evaluate", "record", "hypothesis",
            "note", "semantic_areas", "semantic_target"}

def validate(prog):
    errors = []
    steps = []
    # raiz
    root = prog.get("__root__", [])
    experiment = None
    for ind, s in root:
        if ind == "exp":
            experiment = s
        elif isinstance(s, str) and s.startswith("experiment"):
            experiment = s.split("experiment", 1)[1].strip().rstrip(":").strip()
        else:
            errors.append(f"linea fuera de seccion: {s}")
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
                m_step = re.match(r"^step\s+(\S+)(?:\s+(.*))?$", s)
                if m_step:
                    step_op = m_step.group(1)
                    if step_op not in OPS_ALLOWED:
                        errors.append(f"operacion de paso no permitida: {step_op}")
                    else:
                        steps.append({"sec": sec, "expr": s[:120], "op": step_op, "args": (m_step.group(2) or "").strip()})
                else:
                    errors.append(f"formato de step invalido: {s[:50]}")
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
        "semantic_map": "wsp semantic-map <gguf> (10 areas semanticas y centroides diferenciales)",
        "semantic-map": "wsp semantic-map <gguf>",
        "semantic_areas": "wsp peek <tensor> -r <row> -c <col> (devuelve areas semanticas y afinidad lexica)",
        "semantic-areas": "wsp peek <tensor> -r <row> -c <col>",
        "peek": "wsp peek <tensor> [row=] [col=] [k=] (anatomia del tensor + areas semanticas dominantes y afinidades)",
        "poke": "wsp poke <tensor> <row> <col> <valor> [out=] (poke de un solo peso, requant solo bloque)",
        "poke-weight": "wsp poke <tensor> <row> <col> <valor> (poke de peso individual; requantiza solo el bloque)",
        "cov-steer": "wsp edit (steering alineado con covarianza diagonal por bloques)",
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


def execute_plan(plan, gguf_path=None):
    """Ejecuta los pasos interactivos del plan WSP-Lang usando wsp.py."""
    import sys
    sys.path.insert(0, str(ROOT))
    import wsp

    default_gguf = Path(r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf")
    target_gguf = Path(gguf_path) if gguf_path else default_gguf
    if not target_gguf.exists():
        return [{"op": "init", "status": "error", "args": "", "error": f"Modelo GGUF no encontrado en {target_gguf}"}]

    executed_steps = []
    for step in plan.get("pasos", []):
        op = step["op"]
        raw_args = step.get("args", "")
        tokens = raw_args.split()
        res = {"op": op, "args": raw_args, "status": "ok"}

        try:
            if op in ("semantic_map", "semantic-map"):
                centroids = wsp.get_semantic_centroids(target_gguf)
                top_k = None
                target_tensor = None
                for tok in tokens:
                    if "=" in tok:
                        k_name, val = tok.split("=", 1)
                        if k_name in ("top", "top_k", "k"):
                            top_k = int(val)
                        elif k_name in ("tensor", "t"):
                            target_tensor = val
                    elif tok.startswith("blk.") or "weight" in tok:
                        target_tensor = tok

                areas_info = {}
                for k, meta in wsp.SEMANTIC_AREAS.items():
                    areas_info[k] = {
                        "label": meta["label"],
                        "color": meta["color"],
                        "seeds_count": len(meta["seeds"]),
                        "seeds": meta["seeds"],
                        "has_centroid": k in centroids
                    }

                top_areas = None
                ranking_basis = None
                if target_tensor:
                    # Functional ranking: compute true cosine alignment with target tensor
                    peek_info = wsp.peek(target_gguf, target_tensor, k=top_k or 3)
                    if "semantic_areas" in peek_info and peek_info["semantic_areas"]:
                        ranked = peek_info["semantic_areas"].get("top_areas", [])
                        top_areas = {item["area"]: {"label": item["label"], "score": item["score"]} for item in ranked}
                        ranking_basis = f"tensor_projection:{target_tensor}"
                elif top_k:
                    # Model-wide catalog: rank transparently by seed representative density, not arbitrary dict order
                    sorted_by_seeds = sorted(areas_info.items(), key=lambda kv: len(kv[1]["seeds"]), reverse=True)[:top_k]
                    top_areas = {k: v for k, v in sorted_by_seeds}
                    ranking_basis = f"seed_density (unprojected catalog; specify tensor=... for functional ranking)"

                res["result"] = {
                    "total_areas": len(areas_info),
                    "all_areas": areas_info,
                    "top_areas": top_areas,
                    "ranking_basis": ranking_basis,
                }

            elif op in ("peek", "semantic_areas", "semantic-areas"):
                tensor = tokens[0] if tokens else "blk.19.attn_output.weight"
                row = None
                col = None
                k = 8
                top_k = 3
                for tok in tokens[1:]:
                    if "=" in tok:
                        k_name, val = tok.split("=", 1)
                        if k_name in ("row", "r"): row = int(val)
                        elif k_name in ("col", "c"): col = int(val)
                        elif k_name == "k": k = int(val)
                        elif k_name in ("top", "top_k"): top_k = int(val)
                    elif row is None and (tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit())):
                        row = int(tok)
                    elif col is None and (tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit())):
                        col = int(tok)

                peek_res = wsp.peek(target_gguf, tensor, row=row, col=col, k=k)
                if "semantic_areas" in peek_res and peek_res["semantic_areas"]:
                    all_a = peek_res["semantic_areas"].get("all_areas", [])
                    peek_res["semantic_areas"]["top_areas"] = all_a[:top_k]
                res["result"] = peek_res

            elif op in ("poke", "poke-weight"):
                tensor = tokens[0]
                # Enforce security boundary: forbid critical tensors (norm, output, embd)
                if tensor in wsp.FORBIDDEN_TENSORS or any(p in tensor for p in wsp.FORBIDDEN_PAT):
                    res["status"] = "security_rejection"
                    res["error"] = f"SECURITY POLICY REJECTION: Tensor '{tensor}' is forbidden from direct modification (head/norm protection)."
                    executed_steps.append(res)
                    continue
                row = int(tokens[1])
                col = int(tokens[2])
                val = float(tokens[3])
                out_file = tokens[4] if len(tokens) > 4 else None
                poke_res = wsp.poke(target_gguf, tensor, row, col, val, out=out_file)
                res["result"] = poke_res

            else:
                res["status"] = "skipped (no-interactive)"
        except Exception as e:
            res["status"] = "error"
            res["error"] = str(e)

        executed_steps.append(res)
    return executed_steps


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="wsplang", description="Parser y runtime para WSP-Lang 0.1")
    ap.add_argument("program", help="Ruta al archivo .wspl")
    ap.add_argument("--run", "--exec", action="store_true", help="Ejecutar pasos declarativos (peek, semantic_map, poke) y devolver areas semanticas")
    ap.add_argument("--gguf", default=None, help="Ruta al archivo GGUF (default: MiMo en llamacpp)")
    args = ap.parse_args()

    src = Path(args.program)
    text = src.read_text(encoding="utf-8")
    prog = parse_args_lines(text)
    steps, errors = validate(prog)
    experiment = None
    for ind, s in prog.get("__root__", []):
        if ind == "exp":
            experiment = s
        elif isinstance(s, str) and s.startswith("experiment"):
            experiment = s.split("experiment", 1)[1].strip().rstrip(":").strip()
    plan = plan_run(prog, steps)

    execution_results = None
    if args.run:
        if errors:
            print(f"=== WSP-Lang VALIDATION FAILED: {len(errors)} error(s) ===")
            for err in errors:
                print(f"  [ERROR] {err}")
            execution_results = [{
                "op": "validation",
                "status": "aborted",
                "error": f"Execution halted: {len(errors)} validation errors detected",
                "errors": errors
            }]
        else:
            execution_results = execute_plan(plan, gguf_path=args.gguf)

    out = {
        "programa": experiment,
        "archivo": str(src),
        "steps_count": len(steps),
        "steps": steps,
        "plan": plan,
        "errors": errors, "errores": errors,
        "validated": len(errors) == 0, "validado": len(errors) == 0,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if execution_results is not None:
        out["ejecucion"] = execution_results

    manif = DIR_PL / f"{experiment or 'sin-nombre'}.manifest.json"
    manif.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.run and execution_results:
        print(f"=== WSP-Lang EXECUTION RESULTS: {experiment} ===")
        for step in execution_results:
            op = step["op"]
            print(f"\n> Step: {op} {step['args']} [status: {step['status']}]")
            res = step.get("result", {})
            if "semantic_areas" in res and res["semantic_areas"]:
                sem = res["semantic_areas"]
                dom = sem.get("dominant", {})
                print(f"  DOMINANT SEMANTIC AREA: {dom.get('label')} ({dom.get('area')}) [score={dom.get('score'):+.4f}]")
                top_list = sem.get("top_areas") or sem.get("ranked", [])
                if top_list:
                    print(f"  TOP SEMANTIC AREAS (TOP {len(top_list)}):")
                    for rk in top_list:
                        print(f"    * {rk['label']:<28} ({rk['area']:<15}) score={rk['score']:+.4f}")
                all_list = sem.get("all_areas", [])
                if all_list:
                    print(f"  ALL SEMANTIC AREAS ({len(all_list)}):")
                    for rk in all_list:
                        print(f"    - {rk['label']:<28} ({rk['area']:<15}) score={rk['score']:+.4f}")
                seeds = sem.get("top_seeds", [])
                if seeds:
                    print("  Affine seeds:", ", ".join([f"{s['seed']}:{s['score']:+.3f}" for s in seeds[:5]]))
            elif "semantic_profile" in res:
                prof = res["semantic_profile"]
                print("  TENSOR SEMANTIC PROFILE:")
                top_p = prof.get("top_areas", [])
                if top_p:
                    print("  Top areas:")
                    for it in top_p:
                        print(f"    * {it['label']:<28} ({it['area']:<15}) peso={it['fraction']:.1%}")
                all_p = prof.get("all_areas", [])
                if all_p:
                    print("  All areas:")
                    for it in all_p:
                        print(f"    - {it['label']:<28} ({it['area']:<15}) peso={it['fraction']:.1%}")
            elif "total_areas" in res:
                all_map = res.get("all_areas", {})
                print(f"  ALL MODEL SEMANTIC AREAS ({res['total_areas']} áreas activas):")
                for k, info in all_map.items():
                    print(f"    - {info['label']:<28} ({k:<15}) [semillas: {info['seeds_count']}]")
                top_map = res.get("top_areas")
                if top_map:
                    print(f"\n  SELECTED TOP SEMANTIC AREAS (TOP {len(top_map)}):")
                    for k, info in top_map.items():
                        print(f"    * {info['label']:<28} ({k:<15})")
            elif "error" in step:
                print(f"  Error: {step['error']}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
