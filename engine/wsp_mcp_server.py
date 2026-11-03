"""Servidor MCP para el WSP Challenge Engine (stdio; opencode lo lanza)."""
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
    """Llama a una funcion de wsp.py en proceso (peek|poke|semantic)."""
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
    action: 'next' (siguiente reto), 'brief' (reto concreto), 'submit' (responder: respuest JSON en answer_file), 'expediente'.
    El judador NUNCA revela golds antes del submit; holdouts se sellan aparte."""
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
    """Historial de eventos + verdictos + challenge aprobados: memoria acumulativa."""
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
    """Estado del nivel/estado general del currículo."""
    return _run("estado")


@mcp.tool()
def peek_weight(gguf: str, tensor: str, row: int = "", col: int = "", top: int = 10) -> dict:
    """`peek()`: leer un peso (o la anatomía de un tensor) SIN modificarlo.
    gguf: ruta del modelo (o de un candidato en servidos/). tensor: nombre del tensor.
    row/col: índice de la fila/columna a inspeccionar. top: top-k por norma de fila/columna."""
    return _wsp("peek", gguf, tensor,
                row=int(row) if row else None,
                col=int(col) if col else None,
                k=int(top))


@mcp.tool()
def poke_weight(gguf: str, tensor: str, row: int, col: int, value: float, out: str = "") -> dict:
    """`poke()`: perturbar UN peso individual (requantiza solo su bloque; sin colateral).
    value: valor de destino (en magnitud absoluta, no delta). out: ruta de salida (o '' = archivo .poke-*.gguf)."""
    return _wsp("poke", gguf, tensor, str(row), str(col), str(value),
                out=out if out else None)


@mcp.tool()
def get_semantic_areas(gguf: str, tensor: str = "", row: int = None, col: int = "") -> dict:
    """`get_semantic_areas()`: obtiene y devuelve las áreas semánticas y rol funcional en WSP-Lang.
    gguf: ruta al modelo GGUF.
    tensor: nombre del tensor (opcional). Si no se proporciona, devuelve el mapa completo de áreas.
    row/col: coordenadas del peso (opcional)."""
    if not tensor:
        return _wsp("get_semantic_centroids", gguf)
    return _wsp("peek", gguf, tensor,
                row=int(row) if row is not None and str(row).isdigit() else None,
                col=int(col) if col and str(col).isdigit() else None)


@mcp.tool()
def run_wsplang(program_path: str, execute: bool = True, gguf: str = "") -> dict:
    """Ejecuta un programa declarativo WSP-Lang (.wspl) y devuelve el plan y resultados semánticos.
    program_path: ruta al archivo .wspl.
    execute: si es True, ejecuta interactivamente los pasos y devuelve áreas semánticas."""
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
        out["ejecucion"] = mod.execute_plan(plan, gguf_path=gguf if gguf else None)
    return out


if __name__ == "__main__":
    mcp.run(transport="stdio")
