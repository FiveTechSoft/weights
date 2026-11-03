"""Servidor MCP para el WSP Challenge Engine (stdio; opencode lo lanza)."""
import json
import subprocess
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("wsp-challenge")
ENGINE = r"C:\weights\engine\challenge.py"


def _run(*args):
    p = subprocess.run([sys.executable, ENGINE, *args], capture_output=True, text=True, timeout=1800, cwd=r"C:\weights")
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
