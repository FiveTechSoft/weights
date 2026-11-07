import json, numpy as np
from safetensors.numpy import load_file
W = load_file(r"C:\weights\models\nanogpt_shakespeare_char\model.safetensors")
d = np.load("probe1.npz"); M = json.load(open("probe1_texts.json"))["meta"]
ids = d["ids"]; B, T = ids.shape
logits = d["logits"].astype(np.float32)
y = ids[:, 1:]
pred = logits[:, :-1].argmax(-1)
for kind in ["dev+loop", "rep", "nat"]:
    rows = [i for i, m in enumerate(M) if m["kind"] == kind]
    accs = []
    for i in rows:
        m = M[i]; lo = m["rep_start"] + m["period"] if m["period"] else 128
        accs.append((pred[i, lo:] == y[i, lo:]).mean())
    print(kind, "acc", np.mean(accs))
wte = W["transformer.wte.weight"]; lnf = W["transformer.ln_f.weight"]
mask = np.tril(np.ones((T, T), bool))
ind = np.zeros((6, 6)); prv = np.zeros((6, 6)); dla_rep = np.zeros((6, 6)); dla_nat = np.zeros((6, 6))
for L in range(6):
    A = d[f"transformer.h.{L}.attn.c_attn"].astype(np.float32)
    Wa = W[f"transformer.h.{L}.attn.c_attn.weight"]; Wp = W[f"transformer.h.{L}.attn.c_proj.weight"]
    q = A @ Wa[:384].T; k = A @ Wa[384:768].T
    Y = d[f"transformer.h.{L}.attn.c_proj"].astype(np.float32)
    for h in range(6):
        sl = slice(64 * h, 64 * h + 64)
        s = np.einsum("btd,bsd->bts", q[..., sl], k[..., sl]) / 8.0
        s = np.where(mask, s, -1e9); s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        iv, pv = [], []
        for i, m in enumerate(M):
            if not m["period"]: continue
            P = m["period"]; lo = m["rep_start"] + P
            t = np.arange(lo, T); iv.append(p[i, t, t - P + 1].mean()); pv.append(p[i, t, t - 1].mean())
        ind[L, h] = np.mean(iv); prv[L, h] = np.mean(pv)
        # direct logit attribution of this head toward correct next char (relative to mean logit)
        c = Y[..., sl] @ Wp[:, sl].T  # B,T,384
        U = (wte * lnf)  # 65,384
        lg = c @ U.T; lg = lg - lg.mean(-1, keepdims=True)
        corr = np.take_along_axis(lg[:, :-1], y[..., None], -1)[..., 0]
        r, n = [], []
        for i, m in enumerate(M):
            lo = m["rep_start"] + m["period"] if m["period"] else 128
            (r if m["period"] else n).append(corr[i, lo:].mean())
        dla_rep[L, h] = np.mean(r); dla_nat[L, h] = np.mean(n)
np.set_printoptions(precision=3, suppress=True)
print("induction attn (to t-P+1)\n", ind); print("prev-token attn\n", prv)
print("DLA correct, repeated\n", dla_rep); print("DLA correct, natural\n", dla_nat)
