"""engine/closed_form_write.py - independent closed-form write of key->value associations into one linear layer.

Written from the textbook least-squares formulation (ridge with a protected-key penalty); no code from any external repo.

Given a layer W (d_out x d_in), keys K (d_in x n) seen at its input, and desired output changes R (d_out x n):
    minimise  ||dW K - R||^2  +  ||dW P||^2 * 1  +  lam * ||dW||^2     (P = protected keys, d_in x m)
    solution  dW = R K^T (K K^T + P P^T + lam I)^-1
Protected keys are inputs whose output must not move (ordinary text). The solve is in float64 on a d_in x d_in matrix, so it needs
d_in^2 * 8 bytes (Qwen3.5-0.8B down_proj d_in=3584: ~100 MB). The result is a dense update; nothing is claimed about its size or benefit.

  write_closed_form(W, K, R, P=None, lam=1e-3)  -> dW
  fit_report(W, dW, K, R, P, K_new=None)         -> dict(fit_resid, protected_drift, held_out_shift)
fit_resid is ||dW K - R|| / ||R|| (0 = exact write). protected_drift is ||dW P|| / ||W P|| (output movement on protected keys, relative).
held_out_shift is ||dW K_new|| / ||W K_new|| on keys that were not written: a write that memorises moves them little or randomly; whether
that is a skill is decided only by a task test, never by these numbers.
"""
from typing import Optional

import torch


def write_closed_form(W: torch.Tensor, K: torch.Tensor, R: torch.Tensor, P: Optional[torch.Tensor] = None,
                      lam: float = 1e-3) -> torch.Tensor:
    d_in = K.shape[0]
    if W.shape[1] != d_in or R.shape != (W.shape[0], K.shape[1]):
        raise ValueError("shape mismatch: W %s, K %s, R %s" % (tuple(W.shape), tuple(K.shape), tuple(R.shape)))
    Kd, Rd = K.double(), R.double()
    C = Kd @ Kd.T
    if P is not None:
        if P.shape[0] != d_in:
            raise ValueError("protected keys need d_in rows")
        Pd = P.double()
        C = C + Pd @ Pd.T
    scale = C.diagonal().mean().clamp_min(1e-12)
    C = C + lam * scale * torch.eye(d_in, dtype=torch.float64)
    dW = torch.linalg.solve(C, (Rd @ Kd.T).T).T      # C is symmetric: (R K^T C^-1) = solve(C, K R^T)^T
    if not torch.isfinite(dW).all():
        raise ValueError("non-finite update")
    return dW.to(W.dtype)


def fit_report(W, dW, K, R, P=None, K_new=None) -> dict:
    n = lambda t: float(t.double().norm())
    out = {"fit_resid": n(dW @ K - R) / max(n(R), 1e-30)}
    if P is not None:
        out["protected_drift"] = n(dW @ P) / max(n(W @ P), 1e-30)
    if K_new is not None:
        out["held_out_shift"] = n(dW @ K_new) / max(n(W @ K_new), 1e-30)
    return out
