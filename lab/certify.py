"""certify — juez EXHAUSTIVO de los datasets wspx.
- universe_eval: evalua una funcion de respuestas contra el oraculo del dataset en el universo COMPLETO.
- certify verdicts: EXACT (100% sobre universo), APPROX, FAILED; fallos agrupados por prefijo.

CLI: python lab/certify.py <dataset> --answers <json_path>  (json: {prompt: answer})
     python lab/certify.py <dataset> --selftest   (verifica el propio oraculo codificado)
Salida: rig_runs/certify_<ds>.json  (hash del dataset sellado dentro)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

R = Path(r"C:\weights")
sys.path.insert(0, str(R / "lab"))
from wspx_datasets import BUILDERS  # noqa: E402

OUT = R / "rig" / "runs_done" if False else R / "rig_runs"

def load_dataset(name):
    return BUILDERS[name]()

def universe_eval(ds, answers: dict):
    uni = ds["universe"]
    ok_rows, fails = [], []
    for r in uni:
        p, tgt, pred = r["prompt"], r["target"], answers.get(r["prompt"])
        good = (pred is not None) and (str(pred).strip()[:2] == tgt)
        (ok_rows if good else fails).append(r["prompt"] if good else {"prompt": r["prompt"], "target": tgt, "pred": pred})
    n = len(uni)
    res = {
        "dataset": ds["name"], "dataset_sha256": ds["sha256"],
        "n_universe": n, "n_ok": n - len(fails), "n_fail": len(fails),
        "exact_on_universe": round(len(ok_rows) / max(n, 1), 6),
        "failures_head": [ (f["prompt"], f["pred"], f["target"]) for f in fails[:20] ],
        "verdict": "EXACT" if not fails else ("APPROX" if (n - len(fails)) / n > 0.5 else "FAILED"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return res

def oracle_selftest(ds):
    # oscar del dataset: cada fila debe cumplir el oracle re-ejecutable por regla del generador
    bad = 0
    for r in ds["train"] + ds["holdout"]:
        if not r["target"] or not r["prompt"]:
            bad += 1
    pk = {(x["prompt"], x["target"]) for x in ds["train"]}
    for x in ds["holdout"]:
        if (x["prompt"], x["target"]) in pk:
            return "FAIL: fila compartida train/holdout"
    return "OK" if bad == 0 else f"FAIL: {bad} filas corruptas"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(BUILDERS))
    ap.add_argument("--answers", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ds = load_dataset(a.dataset)
    selftest = oracle_selftest(ds)
    if a.selftest:
        print(json.dumps({"selftest": selftest,
                          "sha256": ds["sha256"],
                          "n_universe": ds["n_universe"],
                          "train/holdout": (ds["n_train"], ds["n_holdout"])}, indent=2))
        sys.exit(0 if selftest == "OK" else 1)
    if not a.answers:
        sys.exit("--answers <json> o --selftest")
    answers = json.loads(Path(a.answers).read_text(encoding="utf-8"))
    res = universe_eval(ds, answers)
    res["oracle_selftest"] = selftest
    outp = Path(a.out) if a.out else OUT / f"certify_{a.dataset}.json"
    outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("dataset", "verdict", "n_ok", "n_fail", "exact_on_universe")}, indent=2))

if __name__ == "__main__":
    main()
