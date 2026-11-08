"""Does choosing the rank-k subspace with real activations beat choosing it from the weights alone? (layer 0, mlp.down_proj, cache only)

down_proj: out = W k, W (4096 x 12288), k = the activation entering it. delta = W_mimo - W_base. A rank-r patch is U U^T delta with U (4096 x r).
 weights-only U : top-r left singular vectors of delta           (what layer_probe / delta_pack use)
 data-aware  U  : top-r left singular vectors of delta @ K_dev^T (K_dev = keys of the 4 dev prompts, 95 tokens; base trajectory)
 random      U  : random orthonormal
Metric on the 8 held-out prompts (217 tokens): captured = <patch K, delta K> / <delta K, delta K>, rel_err = |(delta - patch) K| / |delta K|.
Dev has 95 tokens, so the data-aware subspace can be at most rank 95; ranks tested: 8, 16, 32, 64.
"""
import json, sys
import torch
from safetensors import safe_open
from engine import weight_diff as wd
from lab.probes.delta_patterns import layer_probe as lp

cache = sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe_cache"
pr = lp.Probe(cache, 0)
for n in pr.names:  # base weights into the layer
    pr.params[n[len(pr.pre):]].data.copy_(pr.fb.get_tensor(n).float())
keys = []
hook = pr.layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: keys.append(a[0].reshape(-1, a[0].shape[-1]).clone()))
pr.forward()
hook.remove()
lens = [s.shape[1] for s in pr.seqs]
edge = [0]
for l in lens:
    edge.append(edge[-1] + l)
K = torch.cat(keys)
Kd, Kh = K[:edge[lp.N_DEV]], K[edge[lp.N_DEV]:]
n = pr.pre + "mlp.down_proj.weight"
a, b = pr.fb.get_tensor(n).float(), pr.fm.get_tensor(n).float()
d = b - a
del a, b
def ev(U):
    P = U @ (U.T @ d)
    y, t = (P @ Kh.T).reshape(-1), (d @ Kh.T).reshape(-1)
    return {"captured": float(y @ t / (t @ t)), "rel_err": float((t - y).norm() / t.norm())}
Ud, _, _ = torch.linalg.svd(d @ Kd.T, full_matrices=False)
side, v = torch.load(f"{cache}/basis_{n}.pt")
assert side == "left"
g = torch.Generator().manual_seed(5)
res = {"tokens_dev": int(Kd.shape[0]), "tokens_heldout": int(Kh.shape[0]), "tensor": n, "rows": {}}
for r in (8, 16, 32, 64):
    res["rows"][r] = {"weights_only": ev(v[:, :r]), "data_aware": ev(Ud[:, :r]),
                      "random": ev(torch.linalg.qr(torch.randn(d.shape[0], r, generator=g)).Q)}
    print(r, {k: {m: round(x, 3) for m, x in q.items()} for k, q in res["rows"][r].items()}, flush=True)
json.dump(res, open("/tmp/data_aware_rank.json", "w"), indent=1)
