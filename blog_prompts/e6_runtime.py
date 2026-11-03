"""e6_runtime.py — computa u_resid = Wout @ diff_head para los heads top, y escribe e6.json"""
import importlib.util
import json
import sys

import numpy as np
import gguf

sys.argv = ["wsp", "x"]
spec = importlib.util.spec_from_file_location("wspf", r"C:\weights\wsp.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


P = r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf"
r = m.reader_of(P)
tm = m.tensor_map(r)
t = tm["blk.19.attn_output.weight"]
print("attn_output 19:", [int(x) for x in t.shape], t.tensor_type.name)
W = m.row_decode(t)  # (rows 4096 resid, cols 4096 headspace) según layout gguf

zs = np.load(r"C:\weights\data\head_vecs_e6.npz")
scores = json.load(open(r"C:\weights\data\head_scores_e6.json", encoding="utf-8"))
top_head = scores["19"]["rank"][0]
print("top head:", top_head)
HS = W.shape[1] // 16
diff = zs[f"u_l19_h{top_head}"].astype(np.float64)  # (256,)
u_resid = np.asarray(W[:, top_head * HS:(top_head + 1) * HS], dtype=np.float64) @ diff
u_resid = u_resid / (np.linalg.norm(u_resid) + 1e-30)
np.savez_compressed(r"C:\weights\data\e6_u_resid.npz", **{f"u_l19_h{top_head}": u_resid})
print("u_resid norm:", float(np.linalg.norm(u_resid)), "|med|:", round(float(np.abs(u_resid).mean()), 5))

prog = {
    "base": r"C:\Users\Anto\llamacpp\qwen35base\Qwen3.5_9B_Base.Q6_K.gguf",
    "ops": [
        {
            "type": "subtract-direction",
            "tensor": "blk.19.attn_output.weight",
            "u_from": f"npz:C:\\weights\\data\\e6_u_resid.npz#u_l19_h{top_head}",
            "axis": "rows",
            "cols_mask": list(range(top_head * HS, (top_head + 1) * HS)),
            "target_frac": 0.004,
            "touch_tol": 0.001
        }
    ],
    "out": r"C:\weights\servidos\mimo-e6.gguf",
}
json.dump(prog, open(r"C:\weights\ediciones\e6.json", "w", encoding="utf-8"), indent=1)
print("e6.json listo: head", top_head, "cols", top_head * HS, "-", (top_head + 1) * HS)
