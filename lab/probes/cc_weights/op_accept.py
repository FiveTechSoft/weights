"""Acceptance of the WSP-Lang op compile_control_vector (OP_PLAN.md, criteria fixed before the run): the op's output
in hf_cache/cc_op against the §43.19 run (hf_cache/cc_weights, lab/runs/cc_weights). -> lab/runs/cc_weights/op_accept.json"""
import gc
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OP, OLD = ROOT / "hf_cache" / "cc_op", ROOT / "hf_cache" / "cc_weights"


def ffn_down(path, layers):
    import gguf
    r = gguf.GGUFReader(str(path))
    out = {t.name: (t.tensor_type, np.array(t.data).copy()) for t in r.tensors
           if t.name in {f"blk.{l}.ffn_down.weight" for l in layers}}
    del r
    gc.collect()
    return out


def main():
    import gguf
    rep = json.loads((OP / "report.json").read_text())
    g, gp = rep["G0"], rep["placebo"]["G0"]
    m, mp = rep["vector"]["merge"], rep["placebo"]["merge"]
    new, old = torch.load(OP / "vector" / "fit.pt"), torch.load(OLD / "vec" / "fit.pt")
    cos_u = {l: float(torch.nn.functional.cosine_similarity(new["u"][l].double(), old["u"][l].double(), dim=0))
             for l in old["u"]}
    layers = sorted(old["u"])
    a, b = ffn_down(OP / "merged-Q6_K.gguf", layers), ffn_down(OLD / "cc_merged-Q6_K.gguf", layers)
    ident, cos_w = {}, {}
    for name in a:
        ident[name] = bool(np.array_equal(a[name][1], b[name][1]))
        if not ident[name]:
            x = gguf.dequantize(a[name][1], a[name][0]).astype(np.float64).ravel()
            y = gguf.dequantize(b[name][1], b[name][0]).astype(np.float64).ravel()
            cos_w[name] = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    checks = {
        "g0_vector_median": abs(g["median"] - 0.032) <= 0.003,
        "g0_vector_worst": abs(g["worst"] - 0.043) <= 0.005,
        "g0_placebo_median": abs(gp["median"] - 0.032) <= 0.003,
        "g0_placebo_worst": abs(gp["worst"] - 0.047) <= 0.005,
        "merge_cos_min": abs(m["cos_written_dW_min"] - 0.83) <= 0.02,
        "merge_rel_err_median": abs(m["rel_err_median"] - 0.47) <= 0.02,
        "u_cos_per_layer": min(cos_u.values()) >= 0.999,
        "merged_ffn_down": all(ident.values()) or min(cos_w.values()) >= 0.999,
    }
    res = {"G0_vector": g, "G0_placebo": gp, "merge_vector": {k: m[k] for k in ("cos_written_dW_min",
           "cos_written_dW_median", "rel_err_median")}, "merge_placebo": {k: mp[k] for k in ("cos_written_dW_min",
           "cos_written_dW_median", "rel_err_median")}, "u_cos_min": min(cos_u.values()),
           "ffn_down_byte_identical": sum(ident.values()), "ffn_down_layers": len(ident),
           "ffn_down_cos_min_if_not_identical": min(cos_w.values()) if cos_w else None,
           "checks": checks, "accepted": all(checks.values())}
    (ROOT / "lab" / "runs" / "cc_weights" / "op_accept.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
