"""One streaming pass over Qwen3.8-27B, its Heretic-ARA stage and the TURBO model (plan_turbo27b.md).

usage: python lab/probes/delta_patterns/turbo27b.py [--wait-for <file>] [--limit N] [--only <glob>]
Per tensor: fetch the three versions (in parallel, HTTP ranges, nothing saved), then statistics and signatures of
D_total = turbo - base (-> turbo27b/total/), D_heretic = heretic - base and D_rest = turbo - heretic
(-> turbo27b/sealed/, not to be opened before the blind prediction is committed). Resumable.
"""
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import weight_diff as wd  # noqa: E402

REPOS = {"base": "Qwen/Qwen3.8-27B", "heretic": "trohrbaugh/Qwen3.8-27B-heretic-ara",
         "turbo": "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"}
OUT = ROOT / "lab" / "runs" / "delta_patterns" / "turbo27b"
DELTAS = {"total": ("base", "turbo", OUT / "total"), "heretic": ("base", "heretic", OUT / "sealed"),
          "rest": ("heretic", "turbo", OUT / "sealed")}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def readers():
    return {k: wd.Renamed(wd.RangeSafetensors(wd.hf_files(r), wd.hf_fetch(r)), wd.text_model_name)
            for k, r in REPOS.items()}


def main(limit=None, only=None):
    for _, _, d in DELTAS.values():
        d.mkdir(parents=True, exist_ok=True)
    R = readers()
    names = sorted(set.intersection(*(set(r.keys()) for r in R.values())))
    sig_paths = {k: d / f"sig_{k}.pt" for k, (_, _, d) in DELTAS.items()}
    sigs = {k: (torch.load(p) if p.exists() else {}) for k, p in sig_paths.items()}
    done = {k: {json.loads(l)["tensor"] for l in (d / f"stats_{k}.jsonl").read_text().splitlines()}
            if (d / f"stats_{k}.jsonl").exists() else set() for k, (_, _, d) in DELTAS.items()}
    todo = [n for n in names if not all(n in done[k] and (n in sigs[k]) for k in DELTAS)]
    if only:
        import fnmatch
        todo = [n for n in todo if fnmatch.fnmatch(n, only)]
    if limit:
        todo = todo[:limit]
    print(f"[turbo27b] {len(names)} tensors, {len(todo)} to do", flush=True)
    t0, pool = time.time(), concurrent.futures.ThreadPoolExecutor(3)
    for i, n in enumerate(todo, 1):
        got = dict(zip(R, pool.map(lambda k: R[k].get(n), R)))
        for k, (a, b, d) in DELTAS.items():
            x, y = got[a], got[b]
            if tuple(x.shape) != tuple(y.shape):
                continue
            if n not in done[k]:
                row = wd.tensor_delta_stats(n, x, y, device=DEVICE)
                with (d / f"stats_{k}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            if n not in sigs[k]:
                s = wd.delta_signature(n, x, y, device=DEVICE)
                sigs[k][n] = {key: (v.cpu() if torch.is_tensor(v) else v) for key, v in s.items()}
        if i % 25 == 0 or i == len(todo):
            for k, p in sig_paths.items():
                torch.save(sigs[k], p)
            print(f"[turbo27b] {i}/{len(todo)} {time.time() - t0:.0f}s {n}", flush=True)
    for k, p in sig_paths.items():
        torch.save(sigs[k], p)
    for k, (_, _, d) in DELTAS.items():
        rows = [json.loads(l) for l in (d / f"stats_{k}.jsonl").read_text().splitlines()]
        rep = wd.summarize(rows)
        rep["tensors"] = sorted(rep["tensors"], key=lambda r: -r["rel_delta"])
        (d / f"stats_{k}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (OUT / "done").write_text("ok")
    print("[turbo27b] done", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--wait-for" in args:
        p = Path(args[args.index("--wait-for") + 1])
        while not p.exists():
            time.sleep(120)
    main(limit=int(args[args.index("--limit") + 1]) if "--limit" in args else None,
         only=args[args.index("--only") + 1] if "--only" in args else None)
