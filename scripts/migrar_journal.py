"""Migrar las entradas del journal operativo ({"ts","kind","detail"}) al schema
ExperimentRecord v1 (roadmap #12bis).

Cada entrada del journal es un evento operativo (reserva / edit / read / eval),
no una experimentacion. Por eso todas las entradas migradas llevan
conclusion="INCONCLUSIVE" y hypothesis/target en blanco: el journal no afirma
nada sobre si una intervencion mejora o empeora algo.

El mapa de campos es:
  ts          -> timestamp
  kind        -> decide el rol del registro (verjeta / intervencion / observacion)
  detail      -> texto de evidencia, o los campos de intervencion si aplica

Se escribe a C:\weights\journal\migrado.jsonl (un registro por linea).
"""
import json
import re
import sys

sys.path.insert(0, r"C:\weights\engine")
from record import ExperimentRecord, validate  # noqa: E402

JOURNAL_FILES = [
    r"C:\weights\journal\journal.jsonl",
    r"C:\weights\journal\journal-2026-11-04.jsonl",
]
OUT_FILE = r"C:\weights\journal\migrado.jsonl"

# Expresiones para extraer el modelo resultante de los detalles de reserva/edit.
MODEL_RE = re.compile(r'"out"\s*:\s*"([^"]+)"')
OPS_RE = re.compile(r'"(?:ops|ops_count)"\s*:\s*(\d+)')

# kind -> rol del registro en el nuevo schema
# El journal usa "write" para reservas y "reserva" como sinonimo; ambos son
# intervenciones sobre un modelo.
KIND_ROLES = {
    "write": "intervencion",
    "reserva": "intervencion",
    "edit": "intervencion",     # se edito un modelo: es una intervencion
    "read": "observacion",      # se leio una tensora: es evidencia
    "eval": "observacion",      # se evaluo: es evidencia
}


def parse_model(path):
    """Devolver (modelo, n_ops) desde un detalle de reserva/edit."""
    m = MODEL_RE.search(path or "")
    if not m:
        return None, None
    model = m.group(1)
    ops = OPS_RE.search(path or "")
    return model, int(ops.group(1)) if ops else None


def migrate_entry(entry):
    """Convertir una entrada de journal a un dict listo para ExperimentRecord."""
    ts = entry.get("ts", "")
    kind = entry.get("kind", "")
    detail = entry.get("detail", "")
    role = KIND_ROLES.get(kind, "observacion")

    # El journal no contiene hipotesis ni objetivos declarados: van en blanco.
    hypothesis = ""
    target = ""
    benchmark_version = ""
    placebo = ""
    holdout = ""
    conclusion = "INCONCLUSIVE"

    evidence = ""
    if role == "intervencion":
        # reserva / edit: el modelo resultante es el candidato.
        model, n_ops = parse_model(detail)
        candidate = model or ""
        model_hash = model or ""
        intervention = detail
    else:
        # read / eval: el detalle es evidencia.
        evidence = detail
        candidate = ""
        model_hash = ""
        intervention = ""

    # Para las entradas que nombran una tensora leida, extraer el target.
    if role == "observacion":
        # Busca nombres de tensora como "blk.N.ssm_out.weight" o "attn_q" etc.
        tensor = re.search(r'(?:blk\.|attn\.|blk)[A-Za-z0-9_.]*[A-Za-z_]+', detail)
        target = tensor.group(1) if tensor else ""

    return {
        "ts": ts,
        "kind": kind,
        "hypothesis": hypothesis,
        "evidence": evidence,
        "target": target,
        "intervention": intervention,
        "model_hash": model_hash,
        "benchmark_version": benchmark_version,
        "candidate": candidate,
        "placebo": placebo,
        "holdout": holdout,
        "conclusion": conclusion,
    }


def main():
    records = []
    for fn in JOURNAL_FILES:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                rec = migrate_entry(entry)
                # Validar antes de escribir: el schema debe rechazar un veredicto
                # invalido y aceptar los valores permitidos.
                if not validate(rec):
                    print("  VALIDACION FALLIDA:", fn, "->", json.dumps(rec, ensure_ascii=False)[:160])
                    return 1
                records.append(rec)
        print("  migrado", fn, "->", len(records), "registros")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("  escribio", OUT_FILE, "->", len(records), "registros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
