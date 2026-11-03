"""Servidor MCP para el WSP Challenge & Observability Engine (stdio; opencode o cualquier cliente MCP lo lanza)."""
import json
import platform
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wsp-challenge")
_ROOT = Path(__file__).resolve().parent.parent
ENGINE = _ROOT / "engine" / "challenge.py"
WSP_TOOL = _ROOT / "wsp.py"
_CREATION_FLAGS = {"creationflags": 0x08000000} if platform.system() == "Windows" else {}


def _wsp(fn, *args, **kw):
    """Llama a una función de wsp.py en proceso (peek|poke|semantic)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wsp", WSP_TOOL)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return {"error": str(e)[:600]}
    try:
        func = getattr(mod, fn)
        res = func(*args, **kw)
        if isinstance(res, dict):
            return res
        return {"salida": str(res)}
    except Exception as e:
        return {"error": str(e)[:600]}


def _run(*args):
    p = subprocess.run([sys.executable, str(ENGINE), *args], capture_output=True, text=True, timeout=1800,
                       cwd=str(_ROOT), stdin=subprocess.DEVNULL, **_CREATION_FLAGS)
    if p.returncode != 0:
        return {"error": p.stderr[-400:]}
    try:
        return json.loads(p.stdout)
    except Exception:
        return {"raw": p.stdout[-400:]}


@mcp.tool()
def run_challenge(action: str, reto_id: str = "", answer_file: str = "") -> dict:
    """Retos progresivos del currículo WSP-40.
    action: 'next' (siguiente reto), 'brief' (reto concreto), 'submit' (responder: respuesta JSON en answer_file), 'expediente'.
    El evaluador NUNCA revela golds antes del submit; holdouts se sellan aparte."""
    if action == "next":
        return _run("reto")
    if action == "brief":
        return _run("reto", reto_id)
    if action == "submit":
        return _run("submit", reto_id, answer_file)
    if action == "expediente":
        return _run("expediente", reto_id)
    return {"error": "action: next|brief|submit|expediente"}


@mcp.tool()
def inspect_feedback(alias: str) -> dict:
    """Agregados de un run (aciertos/loops/caps) SIN golds de holdout."""
    return _run("feedback", alias)


@mcp.tool()
def experiment_history(n: int = 40) -> dict:
    """Historial de eventos + veredictos + challenge aprobados: memoria acumulativa."""
    return _run("historial", str(n))


@mcp.tool()
def sell_holdout(alias: str, dataset: str, port: int = 8950) -> dict:
    """Sellado de holdout (1 ejecución por candidato+set; después LOCKED). dataset: math500|diag|code|if."""
    return _run("sellar", alias, dataset, "-p", str(port))


@mcp.tool()
def record_verdict(candidato: str, verdict: str, notas: str = "") -> dict:
    """keep|discard de un candidato con notas. Mantener keep-if-better."""
    return _run("verdicto", candidato, verdict, notas)


@mcp.tool()
def engine_status() -> dict:
    """Estado del nivel y estado general del currículo WSP."""
    return _run("estado")


@mcp.tool()
def peek_weight(gguf: str, tensor: str, row: str = "", col: str = "", top: int = 10) -> dict:
    """`peek()`: leer un peso o anatomía de un tensor SIN modificarlo.
    gguf: ruta del modelo. tensor: nombre del tensor.
    row/col: índice opcional de fila/columna. top: k elementos destacados."""
    return _wsp("peek", gguf, tensor,
                row=int(row) if str(row).isdigit() else None,
                col=int(col) if str(col).isdigit() else None,
                k=int(top))


@mcp.tool()
def poke_weight(gguf: str, tensor: str, row: int, col: int, value: float, out: str = "") -> dict:
    """`poke()`: perturbar UN peso individual (requantiza solo su bloque; protegido por FORBIDDEN_TENSORS).
    value: valor de destino. out: ruta de salida opcional."""
    # Strict security barrier: reject forbidden tensors
    forbidden = ("token_embd.weight", "output.weight", "output_norm.weight")
    if tensor in forbidden or any(p in tensor for p in ("norm.weight", ".norm", "lm_head")):
        return {
            "status": "security_rejection",
            "error": f"SECURITY POLICY REJECTION: Tensor '{tensor}' is protected and forbidden from direct edit."
        }
    return _wsp("poke", gguf, tensor, str(row), str(col), str(value),
                out=out if out else None)


@mcp.tool()
def get_semantic_areas(gguf: str, tensor: str = "", row: str = "", col: str = "") -> dict:
    """Obtiene áreas semánticas y rol funcional anatómico en WSP-Lang.
    gguf: ruta al modelo GGUF.
    tensor: nombre del tensor (opcional). Si no se proporciona, devuelve centroides generales."""
    if not tensor:
        return _wsp("get_semantic_centroids", gguf)
    return _wsp("peek", gguf, tensor,
                row=int(row) if str(row).isdigit() else None,
                col=int(col) if str(col).isdigit() else None)


@mcp.tool()
def run_wsplang(program_path: str, execute: bool = True, gguf: str = "") -> dict:
    """Ejecuta un programa declarativo WSP-Lang (.wspl) con Strict Validation Gate y sandbox de seguridad.
    program_path: ruta al archivo .wspl.
    execute: si es True, ejecuta interactivamente los pasos."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wsplang", str(_ROOT / "engine" / "wsplang.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prog_file = Path(program_path)
    if not prog_file.exists():
        return {"error": f"Archivo {program_path} no encontrado"}
    text = prog_file.read_text(encoding="utf-8")
    prog = mod.parse_args_lines(text)
    steps, errors = mod.validate(prog)
    plan = mod.plan_run(prog, steps)
    out = {"steps": steps, "plan": plan, "errors": errors, "valid": len(errors) == 0}
    if execute:
        if errors:
            out["ejecucion"] = [{
                "op": "validation",
                "status": "aborted",
                "error": f"Execution halted: {len(errors)} validation errors detected",
                "errors": errors
            }]
            return out
        out["ejecucion"] = mod.execute_plan(plan, gguf_path=gguf if gguf else None)
    return out


@mcp.tool()
def execute_wsplang_code(code: str, gguf: str = "") -> dict:
    """Ejecuta un bloque de código WSP-Lang en memoria (inline) con validación estricta y sandbox.
    code: código fuente en sintaxis WSP-Lang (.wspl).
    gguf: ruta opcional al modelo GGUF (default: MiMo-2.6-9B)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wsplang", str(_ROOT / "engine" / "wsplang.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prog = mod.parse_args_lines(code)
    steps, errors = mod.validate(prog)
    plan = mod.plan_run(prog, steps)
    out = {"steps": steps, "plan": plan, "errors": errors, "valid": len(errors) == 0}
    if errors:
        out["ejecucion"] = [{
            "op": "validation",
            "status": "aborted",
            "error": f"Execution halted: {len(errors)} validation errors detected",
            "errors": errors
        }]
        return out
    out["ejecucion"] = mod.execute_plan(plan, gguf_path=gguf if gguf else None)
    return out


@mcp.tool()
def run_agent_loop_probe(endpoint: str = "http://127.0.0.1:8081/v1/chat/completions",
                         alias: str = "mcp-probe", n: int = 5, offset: int = 0,
                         repetition_penalty: float = 1.0, temperature: float = 0.3) -> dict:
    """Ejecuta el probe de bucles agénticos en llamadas a herramientas sobre prompts reales de terminal.
    Mide la tasa de bucles, turnos medios hasta el primer bucle e intervalo de confianza de Wilson al 95%.
    endpoint: URL OpenAI-compatible (/v1/chat/completions).
    repetition_penalty: penalización de repetición (baseline barato de inferencia)."""
    probe_script = _ROOT / "rig" / "agent_loop_probe.py"
    cmd = [
        sys.executable, str(probe_script),
        "--endpoint", endpoint,
        "--alias", alias,
        "--n", str(n),
        "--offset", str(offset),
        "--temperature", str(temperature),
        "--repetition-penalty", str(repetition_penalty)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(_ROOT), **_CREATION_FLAGS)
    out_file = _ROOT / "rig_runs" / f"{alias}.json"
    if out_file.exists():
        try:
            return json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"stdout": p.stdout[-600:], "stderr": p.stderr[-400:], "returncode": p.returncode}


@mcp.tool()
def run_logic_probe(endpoint: str = "http://127.0.0.1:8081/v1/chat/completions",
                    n: int = 5, seed: int = 271828) -> dict:
    """Ejecuta el probe determinista de lógica de Horn (forward chaining sintético) sin dependencias.
    Devuelve la exactitud exacta YES/NO."""
    probe_script = _ROOT / "rig" / "logic_probe.py"
    cmd = [
        sys.executable, str(probe_script),
        "--endpoint", endpoint,
        "--n", str(n),
        "--seed", str(seed)
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(_ROOT), **_CREATION_FLAGS)
    out_file = _ROOT / "rig_runs" / "logic-mimo-orig.json"
    if out_file.exists():
        try:
            return json.loads(out_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"stdout": p.stdout[-600:], "stderr": p.stderr[-400:], "returncode": p.returncode}


if __name__ == "__main__":
    mcp.run(transport="stdio")
