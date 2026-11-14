import numpy as np
from safetensors.numpy import load_file
W = load_file(r"C:\weights\models\nanogpt_shakespeare_char\model.safetensors")
E = W["transformer.wte.weight"]; U = E * W["transformer.ln_f.weight"]
np.set_printoptions(precision=2, suppress=True)
cs = np.zeros((6, 6)); fr = np.zeros((6, 6)); nrm = np.zeros((6, 6))
for L in range(6):
    Wa = W[f"transformer.h.{L}.attn.c_attn.weight"]; Wp = W[f"transformer.h.{L}.attn.c_proj.weight"]
    X = E * W[f"transformer.h.{L}.ln_1.weight"]
    for h in range(6):
        sl = slice(64 * h, 64 * h + 64)
        OV = Wp[:, sl] @ Wa[768:][sl]  # 384x384
        Mh = U @ OV @ X.T  # rows: output logit char, cols: attended char
        Mh = Mh - Mh.mean(0, keepdims=True)
        diag = np.diag(Mh); cs[L, h] = (diag / (Mh.std(0) + 1e-9)).mean()
        fr[L, h] = (Mh.argmax(0) == np.arange(65)).mean()
        nrm[L, h] = np.linalg.norm(OV)
print("copy z-score\n", cs); print("frac chars where attended char is top output\n", fr); print("OV norm\n", nrm)
