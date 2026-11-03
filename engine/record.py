"""Registro de experimentos WSP-Lang.

Cada experimento queda como un registro JSON validado contra un esquema
compartido: objetivo, modelo, edicion, ejecucion, resultados y veredicto.
La version 1 separa dos capacidades que se miden de forma independiente:
  - capacidad funcional de edicion: la edicion se aplica y produce un modelo
    ejecutable con las operaciones del programa.
  - capacidad real de mejro del proces: el modelo editado supera al baseline
    en el objetivo declarado, medido sobre datos de comparacion.
"""

import json
import os
import re
import uuid

SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "experiment_id",
        "model_id",
        "objective",
        "baseline",
        "edicion",
        "ejecucion",
        "resultados",
    ],
    "properties": {
        "experiment_id": {"type": "string", "pattern": "^[e][0-9]{4,}$"},
        "model_id": {"type": "string"},
        "objective": {"type": "string"},
        "baseline": {"type": "object"},
        "edicion": {"type": "object"},
        "ejecucion": {"type": "object"},
        "resultados": {"type": "object"},
        "veredicto": {"type": "string", "enum": ["keep", "discard", "sin_verificar"]},
    },
}

BASELINE_KEYS = ["nombre", "modelo", "version", "pesos", "fecha"]
EDICION_KEYS = ["program", "operaciones", "tensor", "fila", "columna", "valor", "ruta"]
EJECUCION_KEYS = ["estado", "modelo_editado", "aplicada", "errores"]
RESULTADOS_KEYS = ["capacidad_edicion", "capacidad_mejro", "objetivo", "datos", "metricas"]


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def validate(data):
    """Devuelve las violaciones del esquema como lista de cadenas."""
    if not isinstance(data, dict):
        return ["registro no es un objeto"]
    errores = []
    for key in SCHEMA["required"]:
        if key not in data:
            errores.append(f"faltan el campo '{key}'")
    _TIPO = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
    for key, props in SCHEMA["properties"].items():
        if key in data:
            esperado = props["type"]
            if esperado == "object":
                if not isinstance(data[key], dict):
                    errores.append(f"'{key}' debe ser un objeto")
            elif esperado == "array":
                if not isinstance(data[key], list):
                    errores.append(f"'{key}' debe ser una lista")
            else:
                tipos = tuple(esperado) if isinstance(esperado, list) else (esperado,)
                tipos = tuple(_TIPO[t] for t in tipos)
                if not isinstance(data[key], tipos):
                    errores.append(f"'{key}' no es un {esperado}")
            if esperado not in ("object", "array"):
                if "pattern" in props and isinstance(data[key], str):
                    if not re.match(props["pattern"], data[key]):
                        errores.append(f"'{key}' no cumple el patron")
                if "enum" in props and data[key] not in props["enum"]:
                    errores.append(f"'{key}' no esta en el enum")
            if key in BASELINE_KEYS:
                if not isinstance(data[key], dict):
                    errores.append(f"'{key}' debe ser un objeto")
                elif not set(data[key]).issubset(BASELINE_KEYS):
                    errores.append(f"'{key}' tiene campos extra")
            if key == "edicion":
                if not isinstance(data[key], dict):
                    errores.append(f"'{key}' debe ser un objeto")
                elif not set(data[key]).issubset(EDICION_KEYS):
                    errores.append(f"'{key}' tiene campos extra")
            if key == "ejecucion":
                if not isinstance(data[key], dict):
                    errores.append(f"'{key}' debe ser un objeto")
                elif not set(data[key]).issubset(EJECUCION_KEYS):
                    errores.append(f"'{key}' tiene campos extra")
            if key == "resultados":
                if not isinstance(data[key], dict):
                    errores.append(f"'{key}' debe ser un objeto")
                elif not set(data[key]).issubset(RESULTADOS_KEYS):
                    errores.append(f"'{key}' tiene campos extra")
            if key == "objective" and not isinstance(data[key], str):
                errores.append(f"'{key}' debe ser una cadena")
    return errores


def _normalizar(data):
    out = dict(data)
    if isinstance(out.get("baseline"), dict):
        out["baseline"] = {k: _num(v) for k, v in out["baseline"].items()}
    if isinstance(out.get("resultados"), dict):
        out["resultados"] = {k: _num(v) for k, v in out["resultados"].items()}
    return out


class ExperimentRecord:
    """Registro de un experimento WSP-Lang en disco."""

    PATRON_ID = re.compile(r"^[e][0-9]{4,}$")

    def __init__(self, experiment_id, model_id, objective, baseline, edicion,
                 ejecucion, resultados, veredicto="sin_verificar", notas=""):
        self.experiment_id = experiment_id
        self.model_id = model_id
        self.objective = objective
        self.baseline = _normalizar(baseline)
        self.edicion = _normalizar(edicion)
        self.ejecucion = _normalizar(ejecucion)
        self.resultados = _normalizar(resultados)
        self.veredicto = veredicto
        self.notas = notas

    def to_dict(self):
        d = {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "objective": self.objective,
            "baseline": self.baseline,
            "edicion": self.edicion,
            "ejecucion": self.ejecucion,
            "resultados": self.resultados,
        }
        if self.veredicto != "sin_verificar":
            d["veredicto"] = self.veredicto
        if self.notas:
            d["notas"] = self.notas
        return d

    def save(self, path):
        datos = _normalizar(self.to_dict())
        errores = validate(datos)
        if errores:
            raise ValueError("registro invalido: " + "; ".join(errores))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(datos, fh, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            datos = json.load(fh)
        errores = validate(datos)
        if errores:
            raise ValueError("registro invalido: " + "; ".join(errores))
        return cls(
            experiment_id=str(datos["experiment_id"]),
            model_id=str(datos["model_id"]),
            objective=str(datos["objective"]),
            baseline=datos["baseline"],
            edicion=datos["edicion"],
            ejecucion=datos["ejecucion"],
            resultados=datos["resultados"],
            veredicto=str(datos.get("veredicto", "sin_verificar")),
            notas=str(datos.get("notas", "")),
        )

    def append(self, path):
        """Agrega este registro al archivo de historial sin borrar nada."""
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def from_jsonl(cls, path):
        """Lee todos los registros de un historial JSONL.

        Cada registro puede ocupar varias lineas (json.dump con indent),
        asi que se acumulan las lineas hasta completar un valor JSON valido.
        """
        registros = []
        if not os.path.exists(path):
            return registros
        with open(path, "r", encoding="utf-8") as fh:
            texto = fh.read()
        texto = texto.strip()
        if not texto:
            return registros
        decoder = json.JSONDecoder()
        idx = 0
        print("from_jsonl: reading", path)
        while idx < len(texto):
            try:
                datos, fin = decoder.raw_decode(texto, idx)
                print("from_jsonl: decoded", idx, "len", len(texto))
            except json.JSONDecodeError:
                break
            if fin == idx:
                break
        registros.append(cls(
                experiment_id=str(datos["experiment_id"]),
                    model_id=str(datos.get("model_id", "")),
                    objective=str(datos.get("objective", "")),
                    baseline=datos.get("baseline", {}),
                    edicion=datos.get("edicion", {}),
                    ejecucion=datos.get("ejecucion", {}),
                    resultados=datos.get("resultados", {}),
                    veredicto=str(datos.get("veredicto", "sin_verificar")),
                    notas=str(datos.get("notas", "")),
                ))
        return registros

    @classmethod
    def next_id(cls):
        return "e" + str(uuid.uuid4().hex[:8]).upper()


def index_by_objective(registros):
    """Agrupa los registros por objetivo, conservando el orden de aparicion."""
    por_objetivo = {}
    for r in registros:
        por_objetivo.setdefault(r.objective, []).append(r)
    return por_objetivo
