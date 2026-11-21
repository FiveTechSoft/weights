"""Mechanical check of the recompiled weights (PLAN.md, "Repeat with the corrected vector"): the op's output in
hf_cache/cc_op_v2 against the first compilation (hf_cache/cc_weights). Layers 1-30 of cc_v2 equal cc.gguf, so their
u_N and their written ffn_down must be identical; only layer 31 may differ. -> lab/runs/cc_v2/weights/v2_check.json"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from op_accept import ffn_down  # noqa: E402

NEW, OLD = ROOT / "hf_cache" / "cc_op_v2", ROOT / "hf_cache" / "cc_weights"
OUT = ROOT / "lab" / "runs" / "cc_v2" / "weights"


def main():
    import gguf
    rep = json.loads((NEW / "report.json").read_text())
    g, gp = rep["G0"], rep.get("placebo", {}).get("G0")
    new, old = torch.load(NEW / "vector" / "fit.pt"), torch.load(OLD / "vec" / "fit.pt")
    cos_u = {int(l): float(torch.nn.functional.cosine_similarity(new["u"][l].double(), old["u"][l].double(), dim=0))
             for l in old["u"]}
    layers = sorted(cos_u)
    last = max(layers)
    a, b = ffn_down(NEW / "merged-Q6_K.gguf", layers), ffn_down(OLD / "cc_merged-Q6_K.gguf", layers)
    ident, cos_w = {}, {}
    for name in a:
        l = int(name.split(".")[1])
        ident[l] = bool(np.array_equal(a[name][1], b[name][1]))
        if not ident[l]:
            x = gguf.dequantize(a[name][1], a[name][0]).astype(np.float64).ravel()
            y = gguf.dequantize(b[name][1], b[name][0]).astype(np.float64).ravel()
            cos_w[l] = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    m = rep["vector"].get("merge", {})
    checks = {
        "G0_pass": bool(g["pass"]),
        "layers_1_to_30_identical": all(ident[l] for l in layers if l != last),
        "u_cos_1_to_30": min(cos_u[l] for l in layers if l != last) >= 0.999,
    }
    res = {"G0_vector": g, "G0_placebo": gp,
           "merge_vector": {k: m.get(k) for k in ("cos_written_dW_min", "cos_written_dW_median", "rel_err_median")},
           "merge_layer_last": (m.get("layers") or {}).get(str(last)),
           "u_cos_last": cos_u[last], "ffn_down_identical": sum(ident.values()), "ffn_down_layers": len(ident),
           "ffn_down_cos_last": cos_w.get(last), "checks": checks, "ok": all(checks.values())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v2_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
