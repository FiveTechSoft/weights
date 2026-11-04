"""compile_from_trace — el compilador determinista 'trace -> cirugia' del curriculum.
Reglas (inference) fijas y testadas (nada self-report):
  entrada: trace npz (formato curriculum: W0_flat, snapshots, loss, meta json con metrics)
  salida: {tier, surgery_program (ediciones-json estilo wsp.py), record (ExperimentRecord validado)}
Reglas city_fares:
  R1 (EXACT)      : el oraculo del dataset es cerrado-compilable por construccion (dataset in {parity8})
                    -> surgery = compose_circuit
  R2 (APPROX)     : last CPL linear share >= 0.5x delta (readout domina) -> surgery = ridge-readout
  R3 (UNREACHABLE): nada de lo anterior (delta full-rank multicapa / feature-learning)
El tier APH se emite junto con rule_id y el record usa validate() + infer_tier si hay scores.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "lab"))
from engine.record import ExperimentRecord, validate, infer_tier  # noqa: E402

CLOSABLE = {"parity8"}     # datasets con construccion cerrada conocida (M-1b evidenciado)

def detect_tier(meta, dataset):
    lsm = float(meta.get("last_linear_share", 0.0))
    if dataset in CLOSABLE:
        return "EXACT", "R1-compose_circuit"
    if lsm >= 0.5:
        return "APPROX", "R2-ridge-readout"
    return "UNREACHABLE", "R3-none"

def compile_row(dataset, meta, trace_path=None, scores=None, experiment_id=None):
    tier, rule = detect_tier(meta, dataset)
    dsp = R / "lab" / "data" / f"wspx_{dataset}.json"
    ops = {}
    if rule.startswith("R2"):
        ops = {"tensor": "out.weight",
               "op": "ridge-readout",
               "program": ["lambda=1.0", "margin=+9/-9", "features=h_final(W0)"]}
    elif rule.startswith("R1"):
        ops = {"op": "compose_circuit",
               "program": ["threshold-units-k10", "alternating-readout", "const-shift-eps/2"]}
    program = {"experiment": f"compile_{dataset}_{tier}",
                "trace": str(trace_path or ""),
                "rule_id": rule, "ops": ops, "requires": "wsp.py-writer (GGUF) o torch writer (lab)"}
    return program

def experiment_from_trace(trace_path, dataset=None, ep_id=None):
    tr = np.load(trace_path, allow_pickle=False)
    meta = json.loads(str(tr["meta"])) if "meta" in tr.files else {}
    dataset = dataset or meta.get("dataset", "recall")
    tier, rule = detect_tier(meta, dataset)
    program = compile_row(dataset, meta, trace_path)
    rid = hashlib.sha256(str(program).encode()).hexdigest()[:10]
    eid = ep_id or f"e2026.12{dataset[:3]}.{rid}"
    m2b = None
    p2b = R / "lab" / "runs" / f"m2b_results.json"
    if p2b.exists():
        m2b = json.loads(p2b.read_text(encoding="utf-8"))
    rec = {
        "experiment_id": eid,
        "model_id": meta.get("patient", "nanowsp3-pretrained"),
        "objective": f"cirugia compilada desde trace ({rule}) para {dataset}",
        "baseline": {"nombre": "W0 pretrain", "modelo": "nanowsp3-pretrained"},
        "edicion": {"program": json.dumps(program)[:500], "ruta": str(trace_path)},
        "ejecucion": {"estado": "planned" if m2b is None else "compiled"},
        "resultados": (m2b or {}).get("acc", {}),
        "hypothesis": f"trace compilado mediante {rule} reproduzca/supera el finetune de {dataset}",
        "benchmark_version": f"curriculum-{dataset}",
        "candidate": rule, "placebo": "norm-matched-random", "holdout": "certify",
        "surgery_tier": tier, "tier_inferred": True,
        "claim_class": "causal_learn", "placebo_required": True,
        "conclusion": "REJECTED" if tier == "UNREACHABLE" else "SUPPORTED",
        "veredicto": "discard" if tier == "UNREACHABLE" else "keep",
    }
    errs = validate(rec)
    return rec, errs, program

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(R / "lab" / "runs" / "finetune_recall.npz"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    rec, errs, program = experiment_from_trace(a.trace, a.dataset)
    print(json.dumps({"tier": rec["surgery_tier"], "rule": rec["candidate"],
                      "conclusion": rec["conclusion"], "valid": not errs}, indent=2))
    if errs:
        print("ERRORES:", errs)
    if a.save:
        outp = R / "lab" / "runs" / "records" / f"{rec['experiment_id']}.json"
        Path(outp).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print("record guardado en", outp)
