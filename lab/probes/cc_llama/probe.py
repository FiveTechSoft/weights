"""Stage-0 premise probe for lab/probes/cc_llama/PLAN.md (DeepSeek-R1-Distill-Llama-8B): the Phi probe's functions with
this model's runs dir, HF dir and width (d = 4,096, 32 layers).

  python lab/probes/cc_llama/probe.py items <port> <max_tokens> [suffix]  -> lab/runs/cc_llama/probe_<max_tokens><suffix>.json
  python lab/probes/cc_llama/probe.py randvec <out.gguf>                  -> same-norm random control vector, layers 1..31
  python lab/probes/cc_llama/probe.py compare <base_json> <vec_json>      -> P3: outputs differ?
  python lab/probes/cc_llama/probe.py offsets                             -> P4 (tokenizer offsets)
Nothing here decides; the plan reads the numbers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab.probes.cc_phi import probe as phi  # noqa: E402

RUNS = ROOT / "lab" / "runs" / "cc_llama"
HF = ROOT / "hf_cache" / "r1llama8b-hf"
D, LAYERS = 4096, 32
phi.RUNS, phi.HF = RUNS, HF


def items(port, max_tokens, suffix=""):
    return phi.items(port, max_tokens, suffix)


def randvec(out, d=D, layers=LAYERS, norm=1.0, seed=2026):
    import numpy as np
    from engine.contrast_vector import write_gguf
    g = np.random.default_rng(seed)
    vecs = {}
    for k in range(1, layers):
        v = g.normal(size=d).astype(np.float32)
        vecs[k] = v * (norm / np.linalg.norm(v))
    write_gguf(vecs, out, "llama")
    print({"out": str(out), "layers": len(vecs), "norm": norm, "d": d})


compare = phi.compare
offsets = phi.offsets

if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"items": items, "randvec": randvec, "compare": compare, "offsets": offsets}[cmd](*a)
