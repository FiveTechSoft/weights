"""Constrained native output-head surgery on frozen final hidden states.

A finite calibration certificate, NOT a guarantee on unseen text. Use only
with an untied output head, or explicitly untie before applying. No hooks
or additional inference branch are required by the resulting linear head.
"""
import math
import torch


class ReadoutRejected(ValueError):
    def __init__(self, reason, certificate):
        super().__init__(reason)
        self.certificate = certificate


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
    continuous_skill_error = float((x @ delta_t - y).abs().max())
    continuous_protected_error = float((p @ delta_t).abs().max())
    continuous_norm = float(delta_t.norm())
    result = (w + delta_t.T).to(base.dtype)
    actual = result.double() - w
    skill_error = float((x @ actual.T - y).abs().max())
    protected_error = float((p @ actual.T).abs().max())
    norm = float(actual.norm())
    if not all(math.isfinite(v) for v in (skill_error, protected_error, norm)):
        raise ValueError('nonfinite candidate')
    diagnostics = dict(
        protected_rank=len(q), skill_rank=int(keep.sum()),
        skill_constraints=n, nullspace_dimension=d-len(q),
        projected_singular_min=float(singular[keep].min()),
        projected_singular_max=float(singular[keep].max()),
        continuous_skill_error=continuous_skill_error,
        continuous_protected_error=continuous_protected_error,
        continuous_delta_frobenius=continuous_norm,
        skill_error=skill_error, protected_error=protected_error,
        delta_frobenius=norm, max_delta=max_delta, atol=atol)
    failures = []
    if max(continuous_skill_error, continuous_protected_error) > atol:
        failures.append('continuous constraint residual exceeds tolerance')
    if continuous_norm > max_delta:
        failures.append('minimum-norm continuous solution exceeds delta budget')
    if max(skill_error, protected_error) > atol:
        failures.append('representable update violates constraint tolerance')
    if norm > max_delta:
        failures.append('representable update exceeds delta budget')
    if failures:
        raise ReadoutRejected('; '.join(failures), diagnostics)
    return result, dict(skill_error=skill_error, protected_error=protected_error,
                        delta_frobenius=norm, protected_rank=len(q),
                        skill_rank=int(keep.sum()), backward_calls=0)
