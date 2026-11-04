# -*- coding: utf-8 -*-
"""Seed ExperimentRecord schema with the real evidence already gathered.

Two records, both drawn from files that already exist on disk:
  1. P8 sampling hyperparameters -> loop_rate 0.70 -> 0.65 (SUPPORTED)
  2. Loop-direction RE (extract_loop_direction.py) -> layer 1 signal (INCONCLUSIVE)

Run:  python seed_experimentos.py
"""
import json, os, sys
from datetime import datetime, timezone

sys.path.insert(0, r"C:\weights\engine")
from record import ExperimentRecord, validate  # noqa: E402

JOURNAL = r"C:\weights\journal\experimentos.jsonl"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def write(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    recs = []

    # ---- Record 1: P8 sampling hyperparameters (SUPPORTED) ---------------
    # Evidence: rig_runs/eval_vanilla_n20.json (14/20 loops) vs
    #           rig_runs/eval_reppenalty115_n20.json (13/20 loops)
    r1 = ExperimentRecord(
        experiment_id="E0-p8-sampling",
        model_id="mimo-v2.6-distill-qwen-9b",
        objective="programacion general (tarea general, loop_rate)",
        baseline={"nombre": "sampling vanilla", "modelo": "mimo-v2.6-distill-qwen-9b",
                  "version": "temperature 0.3, top_p 0.95, max_tokens 800, seed 42, "
                             "repetition_penalty 1.0, presence_penalty 0.0"},
        edicion={"program": "rig_runs/eval_presence12_n20.json",
                 "operaciones": ["presence_penalty=1.2", "repetition_penalty=1.15"],
                 "tensor": None, "fila": None, "columna": None,
                 "ruta": "rig_runs/eval_presence12_n20.json"},
        ejecucion={"estado": "ok", "modelo_editado": "presence_penalty=1.2, "
                                                      "repetition_penalty=1.15",
                   "aplicada": True, "errores": []},
        resultados={
            "capacidad_mejro": "loop_rate",
            "loop_rate": {"vanilla": 0.70, "candidate": 0.65, "n": 20,
                          "delta": -0.05, "ruta": "rig_runs/eval_vanilla_n20.json"},
        },
        hypothesis="presence_penalty=1.2 reduce la tasa de bucles en la tarea general.",
        evidence={
            "metric": "loop_rate",
            "valor": 0.65,
            "baseline_valor": 0.70,
            "delta": -0.05,
            "n": 20,
            "ruta": "rig_runs/eval_vanilla_n20.json",
            "contextos": "general",
            "fecha": "2026-11-04T00:28:50",
        },
        target="loop_rate",
        intervention="presence_penalty=1.2, repetition_penalty=1.15",
        model_hash="mimo-v2.6-distill-qwen-9b",
        benchmark_version="rig_runs/eval_vanilla_n20.json",
        candidate="presence_penalty=1.2, repetition_penalty=1.15",
        placebo="vanilla (baseline)",
        holdout="rig_runs/eval_vanilla_n20.json",
        conclusion="SUPPORTED",
        veredicto="keep_sampling",
        notas="Mejora REAL y cuantificada: loop_rate 0.70 -> 0.65 (14/20 -> 13/20, n=20). "
              "AVISO honesto: es una mejora de HIPERPARAMETROS DE SAMPLING, NO de "
              "espacio de pesos (weight-space). No es un fix WSP.",
    )
    recs.append(r1.to_dict())

    # ---- Record 2: Loop-direction RE (INCONCLUSIVE) ----------------------
    # Evidence: rig_runs/loop_direction_report.json
    # layer 1 separation_ratio=1.83 p=0.01; layers 2-13 ~1.04-1.19 (ruido)
    r2 = ExperimentRecord(
        experiment_id="E1-re-loop-direction",
        model_id="mimo-v2.6-distill-qwen-9b",
        objective="diagnostico: localizar direccion de bucle en el residual stream",
        baseline={"nombre": "activaciones ocultas loop vs prod", "modelo":
                  "mimo-v2.6-distill-qwen-9b", "version": "rig/extract_loop_direction.py"},
        edicion={"program": "rig/extract_loop_direction.py",
                 "operaciones": ["RE contrastive hidden states: loop vs prod contexts"],
                 "tensor": None, "fila": None, "columna": None,
                 "ruta": "rig_runs/loop_direction_report.json"},
        ejecucion={"estado": "ok", "modelo_editado": "RE contrastive hidden states",
                   "aplicada": True, "errores": []},
        resultados={
            "capacidad_mejro": "separacion_ratio",
            "separacion_ratio": {"L1": 1.83, "L2": 1.52, "L6": 1.19, "L5": 1.16,
                                 "L0": 1.15, "L4": 1.14, "L7": 1.14, "L3": 1.13,
                                 "null_perm_mean": 0.0828, "p_L1": 0.01},
        },
        hypothesis="La capa 1 contiene un vector direccion de bucle-productivo "
                  "distinguible por contraste de activaciones ocultas.",
        evidence={
            "metric": "separation_ratio",
            "valor": 1.83,
            "baseline_valor": 1.0,
            "delta": 0.83,
            "n": 12,
            "ruta": "rig_runs/loop_direction_report.json",
            "contextos": "loop vs prod",
            "separacion": {"L1": 1.83, "L2": 1.52, "L6": 1.19, "L5": 1.16,
                           "L0": 1.15, "L4": 1.14, "L7": 1.14, "L3": 1.13},
            "p_valor": 0.01,
            "null_perm_mean_norm": 0.0828,
        },
        target="separacion_ratio",
        intervention="RE de contrastive hidden states, loop vs prod contexts",
        model_hash="mimo-v2.6-distill-qwen-9b",
        benchmark_version="rig_runs/loop_direction_report.json",
        candidate="L1",
        placebo="permutation null",
        holdout="permutation null",
        conclusion="INCONCLUSIVE",
        veredicto="hypothesis_only",
        notas="UNA sola capa (L1) se separa (1.83, p=0.01); las demas (L2-L13) quedan "
              "en ~1.04-1.19 (ruido). NO es una mejora demostrada: es una hipotesis "
              "con una senal debil y sin confirmacion de mejora de metrica.",
    )
    recs.append(r2.to_dict())

    # ---- validate + write ------------------------------------------------
    for r in recs:
        errs = validate(r)
        if errs:
            print("VALIDATION ERRORS:", errs)
            return 1
        print("OK:", r["experiment_id"], "->", r["conclusion"])

    write(JOURNAL, recs)
    print("WROTE", JOURNAL, len(recs), "records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
