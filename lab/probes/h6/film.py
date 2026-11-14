"""H6 stage 0 film: the frames of each seed side by side, and what is shared across seeds.

Reads lab/runs/h6/seed<s>/chunk<t>.json and steps.jsonl; for the kept checkpoints (chunks 1, 2, 10) also the
cross-seed overlap of the cumulative change (fingerprint.cross_overlap, top-16, smaller side).
Writes lab/runs/h6/film.json and prints one line per frame.

usage: python lab/probes/h6/film.py [seeds...]   (default: every finished seed)
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "lab" / "runs" / "h6"
KEEP = ROOT / "hf_cache" / "h6"
sys.path.insert(0, str(HERE))


def frames(seed):
    out = []
    for t in range(0, 11):
        p = RUNS / f"seed{seed}" / f"chunk{t}.json"
        if not p.exists():
            break
        c = json.loads(p.read_text(encoding="utf-8"))
        e = c["eval"]
        row = {"chunk": t, "test_id": e["test_id"]["acc"], "ood_easy": e.get("ood_easy", {}).get("acc"),
               "ood_hard": e.get("ood_hard", {}).get("acc"), "retention_nll": e.get("retention_nll")}
        if "fingerprint" in c:
            ts = c["fingerprint"]["tensors"]
            row |= {"inc_norm": c["fingerprint"]["total_sq_norm"] ** 0.5,
                    "eff_rank_med": st.median(x["eff_rank"] for x in ts),
                    "top5_med": st.median(x["top5"] for x in ts),
                    "overlap_prev_inc_med": (st.median(x["overlap_prev_inc"] for x in ts)
                                             if ts[0]["overlap_prev_inc"] is not None else None),
                    "overlap_cum_med": (st.median(x["overlap_cum"] for x in ts)
                                        if ts[0]["overlap_cum"] is not None else None),
                    "share_by_kind": {k: v["share"] for k, v in c["summary"]["kind"].items()}}
        out.append(row)
    return out


def steps(seed):
    p = RUNS / f"seed{seed}" / "steps.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ts = r["tensors"].values()
        rows.append({"step": r["step"], "chunk": r["chunk"], "lr": r["lr"], "loss": r["loss"],
                     "cos_gu_med": st.median(v["cos_gu"] for v in ts),
                     "cos_uu_med": st.median(v["cos_uu"] for v in ts) if "cos_uu" in next(iter(ts)) else None})
    return rows


def cross(seeds, chunk):
    import torch  # noqa: F401
    from safetensors.torch import load_file
    import fingerprint as fp
    paths = [KEEP / f"seed{s}" / f"chunk{chunk}.safetensors" for s in seeds]
    if not all(p.exists() for p in paths):
        return None
    base_p = ROOT / "hf_cache" / "qwen35-08b-hf" / "model.safetensors-00001-of-00001.safetensors"
    raw = load_file(str(base_p))
    ws = [load_file(str(p)) for p in paths]
    base = {n: raw[n.replace("model.", "model.language_model.", 1)].float() for n in ws[0]}
    out = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ov = fp.cross_overlap(base, ws[i], ws[j], r=16)
            out[f"{seeds[i]}-{seeds[j]}"] = {"median": st.median(ov.values()), "min": min(ov.values()),
                                             "max": max(ov.values()), "chance_med": st.median(
                                                 16 / min(base[n].shape) for n in base)}
    return out


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [s for s in (1, 2, 3) if (RUNS / f"seed{s}" / "chunk10.json").exists()]
    rep = {"seeds": seeds, "frames": {s: frames(s) for s in seeds}, "steps": {s: steps(s) for s in seeds}}
    if len(seeds) >= 2:
        rep["cross_seed"] = {c: cross(seeds, c) for c in (1, 2, 10)}
    (RUNS / "film.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for s in seeds:
        print(f"seed {s}")
        for f in rep["frames"][s]:
            print("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in f.items() if k != "share_by_kind"))
    if "cross_seed" in rep:
        print("cross-seed overlap of the cumulative change (top-16):", json.dumps(rep["cross_seed"], indent=1))


if __name__ == "__main__":
    main()
