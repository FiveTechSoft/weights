"""A control-vector GGUF times a scale, as a new GGUF (same layers, same model_hint). The compile op writes the vector
at x1.0 (lab/probes/cc_weights/PLAN.md); Phi-4-mini-reasoning's chosen scale is 0.5 (results §45.15), so the op is
given the scaled file and its self-generated placebo (seed 2026) has the scaled norm.

  python lab/probes/cc_phi/scale_vector.py <src.gguf> <dst.gguf> <scale>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def scale_vector(src, dst, scale):
    import gguf
    from engine import contrast_vector as cvx
    from engine.cc_compile import read_vector
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; vectors are never overwritten")
    vecs = {l: v.numpy() * float(scale) for l, v in read_vector(src).items()}
    r = gguf.GGUFReader(str(src))
    hints = [f for f in r.fields.values() if f.name == "controlvector.model_hint"]
    hint = bytes(hints[0].parts[-1]).decode() if hints else "qwen35"
    del r, hints
    import gc
    gc.collect()                       # the reader memory-maps the file; Windows keeps it locked while mapped
    cvx.write_gguf(vecs, dst, hint)
    meta = {"src": str(src), "dst": str(dst), "scale": float(scale), "layers": len(vecs), "model_hint": hint,
            "norms": {str(l): round(float(np.linalg.norm(v)), 4) for l, v in sorted(vecs.items())}}
    Path(str(dst) + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(json.dumps(scale_vector(sys.argv[1], sys.argv[2], float(sys.argv[3]))))
