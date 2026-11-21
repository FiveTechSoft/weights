"""Validate engine/model_report.py on real models whose changes are known.

Cases (expected from documentation or from the full delta statistics of lab/probes/delta_patterns):
  dsv4 heretic vs dsv4 distill   directional_ablation, layers 9-23 (card: start 9, end 24 exclusive)
  huihui abliterated vs base     directional_ablation, layers 1-31 (findings F9, full statistics)
  Nyx-RP vs base                 lora_like (findings F15)
  tmax vs base                   full_fine_tune (RL, findings F11)
  MiMo, NeoHorse vs base         full_fine_tune; profile must match the full statistics (second method)
Writes lab/runs/model_report/<case>.md/.json and validation.json.
"""
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import model_report as mr  # noqa: E402
from engine import weight_diff as wd  # noqa: E402

OUT = ROOT / "lab" / "runs" / "model_report"
BASE = ROOT / "hf_cache" / "qwen35-9b-hf"


def reader(spec):
    if Path(spec).exists() or (ROOT / spec).exists():
        p = Path(spec) if Path(spec).exists() else ROOT / spec
        return wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
    return wd.Renamed(wd.RangeSafetensors(wd.hf_files(spec), wd.hf_fetch(spec)), wd.text_model_name)


CASES = [
    ("dsv4_heretic", "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic", "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B",
     {"method": "directional_ablation", "layers": [9, 23]}),  # card 9-24; Heretic end index is exclusive (ara.py: range(start, end))
    ("abliterated", "huihui-ai/Huihui-Qwen3.5-9B-abliterated", str(BASE), {"method": "directional_ablation", "layers": [1, 31]}),
    ("nyx_rp", "Indexnusrefather/Nyx-RP-9B-Instruct-2608-v1", str(BASE), {"method": "lora_like"}),
    ("tmax", "allenai/tmax-9b", str(BASE), {"method": "full_fine_tune"}),
    ("mimo", "hf_cache/mimo-distill-hf", str(BASE), {"method": "full_fine_tune", "stats": "mimo"}),
    ("neohorse", "hf_cache/neohorse-1-9b-hf", str(BASE), {"method": "full_fine_tune", "stats": "neohorse"}),
]


def spearman(x, y):
    ks = sorted(set(x) & set(y))
    n = len(ks)
    rk = lambda d: {k: i for i, k in enumerate(sorted(ks, key=lambda k: d[k]))}
    a, b = rk(x), rk(y)
    return 1 - 6 * sum((a[k] - b[k]) ** 2 for k in ks) / (n * (n * n - 1))


def main(only=None, rows=128):
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads((OUT / "validation.json").read_text()) if (OUT / "validation.json").exists() else {}
    for name, model, base, expect in CASES:
        if only and name not in only:
            continue
        t0 = time.time()
        rep = mr.build_report(reader(base), reader(model), rows=rows, model_name=model, base_name=base)
        secs = round(time.time() - t0)
        (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        (OUT / f"{name}.md").write_text(mr.render_markdown(rep), encoding="utf-8")
        c = rep["classification"]
        ok = c["method"] == expect["method"] and ("layers" not in expect or c.get("layers") == expect["layers"])
        row = {"expected": expect, "got": {k: c.get(k) for k in ("method", "layers", "rank_median", "rank_estimate",
                                                                   "direction_consistency")},
               "pass": ok, "seconds": secs}
        if "stats" in expect:   # second method: the full statistics of the same pair
            full = [json.loads(l) for l in (ROOT / "lab/runs/delta_patterns" / f"stats_{expect['stats']}.jsonl").read_text().splitlines()]
            g = {}
            for r in full:
                g.setdefault(re.sub(r"^model\.layers\.\d+\.", "", r["tensor"]), []).append(r)
            full_prof = {k: st.median(r["rel_delta"] for r in v) for k, v in g.items()}
            fast_prof = {k: p["rel_median"] for k, p in rep["profile"].items()}
            full_untouched = sorted(k for k, v in g.items() if all(r["changed"] == 0 for r in v))
            row["profile_spearman_vs_full"] = round(spearman(full_prof, fast_prof), 3)
            row["untouched_full"], row["untouched_fast"] = full_untouched, c["untouched"]
            row["pass"] = ok and row["profile_spearman_vs_full"] >= 0.9
        results[name] = row
        (OUT / "validation.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(name, json.dumps(row), flush=True)


if __name__ == "__main__":
    main(only=sys.argv[1:] or None)
