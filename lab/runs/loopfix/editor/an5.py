import numpy as np
d = np.load("probe2.npz"); ids = d["ids"]
lg = d["logits"].astype(np.float32)
lp = lg - lg.max(-1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
lpc = np.take_along_axis(lp[:, :-1], ids[:, 1:, None], -1)[..., 0]  # predicts t+1 at t
j = np.arange(15, 59)
p1 = 70 + j; p2 = 196 + j
print("logp S1 %.3f  S2 %.3f  acc S1 %.3f S2 %.3f" % (lpc[:, p1].mean(), lpc[:, p2].mean(),
      (lg[:, p1].argmax(-1) == ids[:, p1 + 1]).mean(), (lg[:, p2].argmax(-1) == ids[:, p2 + 1]).mean()))
rng = np.random.default_rng(0); tr = rng.permutation(64)[:40]; te = np.setdiff1d(np.arange(64), tr)
dirs = {}
for k in list(range(6)) + ["f"]:
    key = f"transformer.h.{k}.mlp.c_fc" if k != "f" else "lm_head"
    H = d[key].astype(np.float32)
    D = H[:, p2] - H[:, p1]  # B,J,384
    u = D[tr].reshape(-1, 384).mean(0); 
    s1 = (H[te][:, p1] @ u).ravel(); s2 = (H[te][:, p2] @ u).ravel()
    auc = (s2[:, None] > s1[None, :]).mean()
    # consistency: fraction of per-pair differences aligned with u
    cosv = (D[te].reshape(-1, 384) @ u) / (np.linalg.norm(D[te].reshape(-1, 384), axis=1) * np.linalg.norm(u))
    print(k, "|u|=%.3f  heldout AUC %.3f  mean cos %.3f  |D| %.3f" % (np.linalg.norm(u), auc, cosv.mean(), np.linalg.norm(D, axis=-1).mean()))
    dirs[str(k)] = u
np.savez("rep_dirs_probe2.npz", **dirs)
