"""What post-training changed in fine-tunes of one base model, and whether two fine-tunes changed the same things.

usage:
  python lab/probes/delta_patterns/run.py stats <target_name> <target_dir>   per-tensor stats of target - base
  python lab/probes/delta_patterns/run.py overlap <name1> <dir1> <name2> <dir2>   delta_overlap per tensor

Base: hf_cache/qwen35-9b-hf (Qwen/Qwen3.5-9B). Names lined up with weight_diff.text_model_name (text model only).
Each tensor's row is appended to lab/runs/delta_patterns/<job>.jsonl as soon as it is done; a rerun skips rows
already there. A summary JSON is written at the end.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import weight_diff as wd  # noqa: E402

BASE = ROOT / "hf_cache" / "qwen35-9b-hf"
OUT = ROOT / "lab" / "runs" / "delta_patterns"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("DP_THREADS", "4")))  # less CPU parallelism after the low-memory kill


def reader(path):
    return wd.Renamed(wd.LocalSafetensors(path), wd.text_model_name)


def done_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(name, target_dir):
    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / f"stats_{name}.jsonl"
    rows = done_rows(part)
    base, tgt = reader(BASE), reader(target_dir)
    t0 = time.time()

    def save(row):
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def progress(i, n, tensor):
        if i % 20 == 0:
            print(f"[{name}] {i}/{n} {time.time() - t0:.0f}s {tensor}", flush=True)

    kb, kt = set(base.keys()), set(tgt.keys())
    names = sorted(kb & kt)
    skip = {r["tensor"] for r in rows}
    mismatch = []
    for i, n in enumerate(names):
        if n in skip:
            continue
        a, b = base.get(n), tgt.get(n)
        if tuple(a.shape) != tuple(b.shape):
            mismatch.append(n)
            continue
        row = wd.tensor_delta_stats(n, a, b, device=DEVICE)
        rows.append(row)
        save(row)
        progress(i + 1, len(names), n)
    rep = wd.summarize(rows, only_in_base=sorted(kb - kt), only_in_target=sorted(kt - kb), shape_mismatch=mismatch)
    rep["tensors"] = sorted(rep["tensors"], key=lambda r: -r["rel_delta"])
    (OUT / f"stats_{name}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(name, json.dumps({"overall": rep["overall"], "by_family": rep["by_family"]}, indent=1), flush=True)


def overlap(n1, d1, n2, d2, k=16):
    OUT.mkdir(parents=True, exist_ok=True)
    job = f"overlap_{n1}_{n2}"
    part = OUT / f"{job}.jsonl"
    rows = done_rows(part)
    skip = {r["tensor"] for r in rows}
    base, t1, t2 = reader(BASE), reader(d1), reader(d2)
    names = sorted(set(base.keys()) & set(t1.keys()) & set(t2.keys()))
    t0 = time.time()
    for i, n in enumerate(names):
        if n in skip:
            continue
        a, b1, b2 = base.get(n), t1.get(n), t2.get(n)
        if not (a.shape == b1.shape == b2.shape):
            continue
        row = {"tensor": n, "family": wd.family(n), "layer": wd.layer_of(n), "shape": list(a.shape),
               **wd.delta_overlap(a, b1, b2, k=k, device=DEVICE)}
        rows.append(row)
        with part.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if (i + 1) % 20 == 0:
            print(f"[{job}] {i + 1}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    fam = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)
    summary = {f: {"tensors": len(v),
                   "mean_cosine": sum(r["cosine"] for r in v) / len(v),
                   "mean_subspace_overlap": (sum(r["subspace_overlap"] for r in v if "subspace_overlap" in r)
                                             / max(1, sum("subspace_overlap" in r for r in v))),
                   "mean_random_overlap": (sum(r["random_overlap"] for r in v if "random_overlap" in r)
                                           / max(1, sum("random_overlap" in r for r in v)))}
               for f, v in sorted(fam.items())}
    (OUT / f"{job}.json").write_text(json.dumps({"by_family": summary, "tensors": rows}, indent=1), encoding="utf-8")
    print(job, json.dumps(summary, indent=1), flush=True)


def signatures(name, target_dir, base_dir=BASE, k=16, sketch_dim=4096):
    """delta_signature for every tensor, saved to sig_<name>.pt (resumable, every 20 tensors)."""
    path = OUT / f"sig_{name}.pt"
    sigs = torch.load(path) if path.exists() else {}
    base, tgt = reader(base_dir), reader(target_dir)
    names = sorted(set(base.keys()) & set(tgt.keys()))
    t0 = time.time()
    for i, n in enumerate(names):
        if n in sigs:
            continue
        a, b = base.get(n), tgt.get(n)
        if tuple(a.shape) != tuple(b.shape):
            continue
        s = wd.delta_signature(n, a, b, k=k, sketch_dim=sketch_dim, device=DEVICE)
        sigs[n] = {key: (v.cpu() if torch.is_tensor(v) else v) for key, v in s.items()}
        if (i + 1) % 20 == 0:
            torch.save(sigs, path)
            print(f"[sig {name}] {i + 1}/{len(names)} {time.time() - t0:.0f}s", flush=True)
    torch.save(sigs, path)
    print(f"[sig {name}] done {len(sigs)} tensors", flush=True)


def check_second_method(name, target_dir, base_dir=BASE, n=3):
    """Recompute rel_delta and stable rank of n tensors with numpy float64 and a full SVD; must match the stats."""
    import numpy as np
    rows = {r["tensor"]: r for r in done_rows(OUT / f"stats_{name}.jsonl")}
    pick = [t for t in ("model.layers.10.linear_attn.in_proj_b.weight", "model.layers.17.mlp.gate_proj.weight",
                        "model.layers.3.self_attn.v_proj.weight") if t in rows][:n]
    base, tgt = reader(base_dir), reader(target_dir)
    out = []
    for t in pick:
        a = base.get(t).float().numpy().astype(np.float64)
        d = tgt.get(t).float().numpy().astype(np.float64) - a
        nd = np.linalg.norm(d)
        s = np.linalg.svd(d, compute_uv=False) if nd > 0 else np.zeros(1)
        rel, sr = nd / np.linalg.norm(a), ((s ** 2).sum() / s[0] ** 2 if nd > 0 else 0.0)
        ok = abs(rel - rows[t]["rel_delta"]) <= 1e-6 + 1e-4 * rel and abs(sr - rows[t]["stable_rank"]) <= 1e-3 * max(sr, 1)
        out.append({"tensor": t, "rel_numpy": rel, "rel_tool": rows[t]["rel_delta"], "stable_rank_numpy": sr,
                    "stable_rank_tool": rows[t]["stable_rank"], "agree": bool(ok)})
    (OUT / f"check_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[check {name}]", [(o["tensor"].split(".")[2], o["agree"]) for o in out], flush=True)
    if not all(o["agree"] for o in out):
        raise SystemExit(f"second method disagrees for {name}")


def download(repo, local):
    import subprocess
    cmd = ["hf", "download", repo, "--include", "*.safetensors", "--include", "*.json", "--local-dir", str(local),
           "--max-workers", os.environ.get("DP_DOWNLOAD_WORKERS", "2")]  # default 8 ran out of RAM next to the GPU job
    for attempt in range(3):
        if subprocess.run(cmd).returncode == 0 and list(Path(local).glob("*.safetensors")):
            return
        time.sleep(30)
    raise SystemExit(f"download failed: {repo}")


def wait_for(path, every=60):
    while not Path(path).exists():
        time.sleep(every)


# name, repo to download (None = already local), local dir, base dir
QUEUE = [
    ("mimo", None, "hf_cache/mimo-distill-hf", BASE),
    ("neohorse", None, "hf_cache/neohorse-1-9b-hf", BASE),
    ("abliterated", "huihui-ai/Huihui-Qwen3.5-9B-abliterated", "hf_cache/_dp_abliterated", BASE),
    ("tmax", "allenai/tmax-9b", "hf_cache/_dp_tmax", BASE),
    ("fara", "microsoft/Fara1.5-9B", "hf_cache/_dp_fara", BASE),
    ("nyx_rp", "Indexnusrefather/Nyx-RP-9B-Instruct-2608-v1", "hf_cache/_dp_nyx_rp", BASE),
    ("official_post", "Qwen/Qwen3.5-9B-Base", "hf_cache/_dp_qwen35_9b_base", None),  # base = the download
    # Controls added after F15 (findings.md F16): full fine-tune with another goal; a deliberate degradation.
    ("wemm_embed", "tencent/WeMM-Embedding-9B", "hf_cache/_dp_wemm_embed", BASE),
    ("angry_claudius", "axiomofmind/Angry-Claudius-9B", "hf_cache/_dp_angry_claudius", BASE),
    # DeepSeek-V4-Pro distill (LoRA-like) and its Heretic ARA (layers 9-24 documented): findings F17.
    # The distill is kept (no _dp_ prefix) because it is the base of the Heretic diff; delete it by hand afterwards.
    ("dsv4_distill", "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B", "hf_cache/dp_keep_dsv4_distill", BASE),
    ("dsv4_heretic", "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic", "hf_cache/_dp_dsv4_heretic",
     ROOT / "hf_cache" / "dp_keep_dsv4_distill"),
]


def pipeline():
    """Each model: download (if remote), stats, signatures, second-method check, delete the download."""
    import shutil
    wait_for(OUT / "overlap_mimo_neohorse.json")  # the GPU is busy until the first overlap job ends
    for name, repo, local, base_dir in QUEUE:
        local = ROOT / local
        if (OUT / f"sig_{name}.done").exists():
            continue
        if repo:
            download(repo, local)
        if base_dir is None:  # official post-training: Base -> Qwen3.5-9B
            base_dir, target = local, BASE
        else:
            target = local
        if not (OUT / f"stats_{name}.json").exists():
            stats_with_base(name, target, base_dir)
        check_second_method(name, target, base_dir)
        signatures(name, target, base_dir)
        (OUT / f"sig_{name}.done").write_text("ok")
        if repo and local.name.startswith("_dp_"):
            shutil.rmtree(local)
            print(f"[pipeline] deleted {local}", flush=True)


def stats_with_base(name, target_dir, base_dir):
    global BASE
    saved, BASE = BASE, Path(base_dir)
    try:
        stats(name, target_dir)
    finally:
        BASE = saved


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "stats":
        stats(a[0], ROOT / a[1])
    elif cmd == "overlap":
        overlap(a[0], ROOT / a[1], a[2], ROOT / a[3])
    elif cmd == "pipeline":
        pipeline()
    else:
        raise SystemExit(__doc__)
