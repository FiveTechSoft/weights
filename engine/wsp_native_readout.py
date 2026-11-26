"""Constrained native output-head surgery on frozen final hidden states.

A finite calibration certificate, NOT a guarantee on unseen text. Use only
with an untied output head, or explicitly untie before applying. No hooks
or additional inference branch are required by the resulting linear head.
"""
import math
import torch


@torch.no_grad()
def compile_rows(base, skill, protected, desired, *, max_delta, atol=1e-7, rtol=1e-10):
    """Return new selected readout rows and a certificate, or reject.

    base: (k,d) selected output rows; skill: (n,d) final hidden states;
    protected: (m,d) final hidden states; desired: (n,k) logit increments.
    The certificate uses the actual representable, dtype-rounded update.
    All other head rows must remain unchanged when the result is installed.
    """
    if not all(math.isfinite(v) and v > 0 for v in (max_delta, atol, rtol)) or rtol >= 1:
        raise ValueError('invalid tolerances or budget')
    if any(t.ndim != 2 or not t.is_floating_point() or not torch.isfinite(t).all()
           for t in (base, skill, protected, desired)):
        raise ValueError('expected finite floating matrices')
    k, d = base.shape
    n, sd = skill.shape
    if not k or not d or not n or not len(protected) or sd != d or protected.shape[1] != d or desired.shape != (n,k):
        raise ValueError('shape mismatch or empty calibration')
    w, x, p, y = (t.detach().to(device=base.device, dtype=torch.float64)
                  for t in (base, skill, protected, desired))
    _, s, vh = torch.linalg.svd(p, full_matrices=False)
    q = vh[s > rtol*s[0]]
    if len(q) == d:
        raise ValueError('protected states span the complete readout space')
    z = x - (x @ q.T) @ q
    u, singular, v = torch.linalg.svd(z, full_matrices=False)
    keep = singular > rtol * torch.linalg.matrix_norm(x, ord=2)
    if not keep.any():
        raise ValueError('no identifiable skill directions in protected nullspace')
    delta_t = v[keep].T @ ((u[:,keep].T @ y) / singular[keep,None])
    delta_t = delta_t - q.T @ (q @ delta_t)
    result = (w + delta_t.T).to(base.dtype)
    actual = result.double() - w
    skill_error = float((x @ actual.T - y).abs().max())
    protected_error = float((p @ actual.T).abs().max())
    norm = float(actual.norm())
    if not all(math.isfinite(v) for v in (skill_error, protected_error, norm)):
        raise ValueError('nonfinite candidate')
    if max(skill_error, protected_error) > atol or norm > max_delta:
        raise ValueError('incompatible constraints or delta budget exceeded')
    return result, dict(skill_error=skill_error, protected_error=protected_error,
                        delta_frobenius=norm, protected_rank=len(q),
                        skill_rank=int(keep.sum()), backward_calls=0)
