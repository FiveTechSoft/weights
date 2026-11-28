"""Adam / weight-decay footprint on a full fine-tune pair read over HTTP range requests (no download).

Runs the preregistered classifier of lab/probes/adam_sig/PLAN.md (classify() in adam_sig.py, unmodified,
thresholds DECAY_TOL 0.05, STEP_CV 0.15) on DeepSeek-R1-Distill-Qwen-7B vs its public base Qwen2.5-Math-7B,
streaming the two bf16 tensors row-chunk by row-chunk from the Hub instead of downloading ~30 GB.
Result: results.md section 45.38, predictions.md row 54 (2026-11-28).

    python lab/probes/adam_sig/adam_stream.py embed|lm_head
CPU only; needs numpy and curl.
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adam_sig import classify, _summary, _levels, zero_rows_a, zero_rows_b  # noqa: E402

MODELS = {
    "donor": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/",
    "base": "https://huggingface.co/Qwen/Qwen2.5-Math-7B/resolve/main/",
}
KEYS = {"embed": "model.embed_tokens.weight", "lm_head": "lm_head.weight"}
ROWS, COLS, BROW, CH = 152064, 3584, 7168, 8192


def _get(url, off=None, nbytes=None):
    cmd = ["curl", "-sL"]
    if off is not None:
        cmd += ["-r", f"{off}-{off + nbytes - 1}"]
    cmd.append(url)
    out = subprocess.run(cmd, capture_output=True).stdout
    if off is not None:
        assert len(out) == nbytes, (len(out), nbytes)
    return out


def _locate(root, tensor_key):
    idx = json.loads(_get(root + "model.safetensors.index.json"))["weight_map"]
    url = root + idx[tensor_key]
    n = struct.unpack("<Q", _get(url, 0, 8))[0]
    head = json.loads(_get(url, 0, 8 + n)[8:])
    e = head[tensor_key]
    assert e["dtype"] == "BF16" and e["shape"] == [ROWS, COLS], e
    return url, 8 + n + e["data_offsets"][0]


def _bf16_f32(u8):
    a = u8.reshape(-1, 2)
    b = np.zeros((len(a), 4), dtype=np.uint8)
    b[:, 2:] = a
    return b.view(np.float32).reshape(-1)


def main(key):
    locs = {tag: _locate(root, KEYS[key]) for tag, root in MODELS.items()}
    parts, za, zb = [], 0, 0
    for i in range(0, ROWS, CH):
        n = min(CH, ROWS - i)
        nb = n * BROW
        Wb = _bf16_f32(_get(locs["base"][0], locs["base"][1] + i * BROW, nb)).reshape(n, COLS)
        Wd = _bf16_f32(_get(locs["donor"][0], locs["donor"][1] + i * BROW, nb)).reshape(n, COLS)
        D = Wd - Wb
        parts.append(classify(Wb.astype(np.float32), D.astype(np.float32)))
        za += zero_rows_a(D)
        zb += zero_rows_b(D)
        print(f"rows {i}-{i + n}", flush=True)
    r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cls = r["cls"]
    counts = {k: int(np.sum(cls == k)) for k in ("zero", "decay", "step", "other")}
    assert za == zb == counts["zero"], (za, zb, counts)
    order = np.argsort(-r["norm"])
    med = float(np.median(r["norm"][r["norm"] > 0])) if np.any(r["norm"] > 0) else 0.0
    rep = {"tensor": KEYS[key], "rows": ROWS, "counts": counts,
           "decay_c": _summary(r["decay_c"][cls == "decay"]),
           "step_mag": _summary(r["step_mag"][cls == "step"]),
           "step_mag_levels": _levels(r["step_mag"][cls == "step"]),
           "norm_median_nonzero": med,
           "top": [{"id": int(i), "norm": float(r["norm"][i]),
                    "x_median": float(r["norm"][i] / med) if med else None,
                    "cls": str(cls[i])} for i in order[:20]],
           "zero_sample_ids": [int(i) for i in np.flatnonzero(cls == "zero")[:: max(1, counts["zero"] // 20 or 1)][:20]]}
    print("REPORT", json.dumps({k: rep[k] for k in ("tensor", "counts", "decay_c", "step_mag", "norm_median_nonzero")}))
    return rep


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "embed")
