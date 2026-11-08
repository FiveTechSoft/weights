"""Write the protocol v6 outcome as an engine/record.py experiment record and validate it with the real schema.

    python lab/meta/record_v6.py        # writes lab/meta/records/e_v6_adaptive.json

Fit test, not a new op: it shows the existing record schema can hold a pre-registered result with its placebo
(shuffled prior), controls and verdict. Conclusions are derived from the criteria, not typed in.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from engine import record as rec  # noqa: E402
from lab.meta import evaluate_v6 as ev6  # noqa: E402


def build_record():
    rows = ev6.compute_rows()
    crit = ev6.criteria(rows)
    holds = {k: v[2] < 0 for k, v in crit.items()}                  # 95% CI entirely below 0
    mean = lambda p: statistics.fmean(r[p][0] for r in rows.values())
    claim = holds["k"] and holds["l"]                                # pre-registered claim in 6ba876c
    return {
        "experiment_id": "e_v6_adaptive",
        "model_id": "NanoWSP-3",
        "objective": "within-task adaptation helps beyond the fixed recipe (pre-registered in 6ba876c)",
        "baseline": {"nombre": "fixed_recipe", "modelo": "NanoWSP-3", "version": "lab/meta/interference_v6.py"},
        "edicion": {"program": "lab/meta/evaluate_v6.py", "operaciones": ["imprint_thematic_corpus"],
                    "ruta": "lab/meta/interference_v6_scores.json"},
        "ejecucion": {"estado": "executed", "aplicada": True, "errores": []},
        "resultados": {"capacidad_edicion": 1.0, "capacidad_mejro": 1.0 if claim else 0.0,
                       "objetivo": "mean trials-to-success, censored at 16",
                       "datos": {p: round(mean(p), 3) for p in
                                 ("random", "fixed_recipe", "prior_only", "flat_adaptive", "shuffled_adaptive", "adaptive", "oracle")},
                       "metricas": {k: {"mean_diff": round(v[0], 3), "ci95": [round(v[1], 3), round(v[2], 3)], "holds": holds[k]}
                                    for k, v in crit.items()}},
        "hypothesis": "Updating a meta-learned prior with the new task's own trials beats the fixed recipe.",
        "evidence": {"metric": "trials_to_success", "n": len(rows), "ruta": "lab/meta/interference_v6_results.txt",
                     "delta": round(crit["l"][0], 3)},
        "placebo": "shuffled-experience prior (wrong task labels), 100 shuffles",
        "holdout": "20 fresh tasks scored after the pre-registration commit 6ba876c",
        "conclusion": "SUPPORTED" if claim else "INCONCLUSIVE",
        "veredicto": "keep" if claim else "sin_verificar",
        "claim_class": "causal_learn",
        "placebo_required": True,
    }


if __name__ == "__main__":
    r = build_record()
    errs = rec.validate(r)
    assert not errs, errs
    out = HERE / "records" / "e_v6_adaptive.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: r[k] for k in ("conclusion", "veredicto")}), r["resultados"]["metricas"])
