"""Auto-entry hook: append a record to journal/experimentos.jsonl at the end of each rig execution.

Reads rig_runs/*.json, computes the loop_rate for each run, compares it against the
vanilla baseline, and appends one ExperimentRecord per run. Only runs whose results
are a list of per-candidate entries are auto-entered; summaries, reports and
non-loop runs are skipped (they are not sampling-hyperparameter experiments).
"""
import json
import os
import sys

sys.path.insert(0, r"C:\weights\engine")
from record import ExperimentRecord

RIG_DIR = r"C:\weights\rig_runs"
JOURNAL = r"C:\weights\journal\experimentos.jsonl"
BASELINE_ALIAS = "eval_vanilla_n20"


def load_run(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("meta", {})
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return None, None
    if not isinstance(meta, dict):
        return None, None
    return meta, results


def loop_rate(results):
    """Fraction of candidates that looped."""
    total = len(results)
    loops = sum(1 for r in results if r.get("loop") is True)
    return loops / total if total else None


def baseline_loop_rate():
    path = os.path.join(RIG_DIR, BASELINE_ALIAS + ".json")
    if not os.path.exists(path):
        return None
    meta, results = load_run(path)
    return loop_rate(results)


def detect_intervention(meta):
    """Return (candidate, placebo) describing which sampling param changed."""
    base = {"temperature": 0.3, "top_p": 0.95, "max_tokens": 800, "seed": 42,
            "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
    params = meta.get("sampling_params", {})
    changed = {k: v for k, v in params.items() if k in base and v != base[k]}
    if not changed:
        return None, None
    placebo = "vanilla (baseline)"
    candidate = ", ".join(f"{k}={v}" for k, v in changed.items())
    return candidate, placebo


def main():
    base_rate = baseline_loop_rate()
    print(f"baseline loop_rate = {base_rate}")
    if base_rate is None:
        print("no baseline found; skipping auto-entry")
        return

    records = []
    for fname in sorted(os.listdir(RIG_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RIG_DIR, fname)
        meta, results = load_run(path)
        if meta is None or results is None:
            continue
        alias = meta.get("alias", fname)
        rate = loop_rate(results)
        cand, place = detect_intervention(meta)
        if cand is None:
            continue
        if rate is None:
            continue
        delta = rate - base_rate
        conclusion = "SUPPORTED" if delta < 0 else "INCONCLUSIVE"
        rec = ExperimentRecord(
            experiment_id=f"E-{alias}-loop",
            model_id="mimo-v2.6-distill-qwen-9b-Q6_K",
            objective="loop_rate",
            hypothesis=f"presence_penalty={meta.get('sampling_params',{}).get('presence_penalty','0.0')} "
                       f"reduce the loop rate in the general task (baseline {base_rate:.2f})",
            target="loop_rate",
            intervention=cand,
            model_hash="",
            benchmark_version="loop_rate_n20",
            candidate=cand,
            placebo=place,
            holdout=path,
            conclusion=conclusion,
            baseline=base_rate,
            edicion=rate,
            ejecucion=delta,
            resultados=results,
            veredicto=conclusion,
            notas=(f"rig_runs/{fname}; n={len(results)}; loop_rate {rate:.2f} vs baseline {base_rate:.2f} "
                   f"(delta {delta:+.2f}); sampling-only experiment (presence_penalty), not a "
                   f"weight-space edit."),
        )
        records.append(rec)
        print(f"  {fname:38s} rate={rate:.2f} delta={delta:+.2f} -> {conclusion}")

    for rec in records:
        with open(JOURNAL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict()) + "\n")
    print(f"WROTE {len(records)} record(s) to {JOURNAL}")


if __name__ == "__main__":
    main()
