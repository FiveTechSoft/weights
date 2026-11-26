"""Exact two-layer gradient flow on the balanced, aligned spectral manifold.
Loss = ||B A - C||_F^2 / 2. This is not general Transformer training.
"""
import math
import torch


@torch.no_grad()
def matrix_flow(a, b, target, time, *, tolerance=1e-9):
    """Return new A,B without iterations; reject unsupported initial states.

    A (r,n), B (m,r), C (m,n), r <= min(m,n). The initial product
    must use the first r singular modes of C, with positive balanced
    factors and a shared orthogonal hidden basis.
    """
    if not math.isfinite(time) or time < 0 or not 0 < tolerance <= 1e-6:
        raise ValueError('invalid time or tolerance')
    if any(x.ndim != 2 or not x.is_floating_point() or not torch.isfinite(x).all() for x in (a,b,target)):
        raise ValueError('expected finite floating matrices')
    r,n = a.shape
    m,k = b.shape
    if not 0 < r <= min(m,n) or k != r or target.shape != (m,n):
        raise ValueError('shape mismatch')
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError('factor dtype/device mismatch')
    x,y,c = (z.to(device=a.device,dtype=torch.float64) for z in (a,b,target))
    u,s,vh = torch.linalg.svd(c, full_matrices=False)
    u,s,vh = u[:,:r],s[:r],vh[:r]
    if (s <= 0).any():
        raise ValueError('target modes must be positive')
    av = x @ vh.T
    bu = y.T @ u
    scale = av.norm()
    if scale == 0:
        raise ValueError('zero initialization')
    def require(error, reference):
        if error.norm() > tolerance * reference.norm():
            raise ValueError('initial factors not balanced and aligned')
    require(x-av@vh,x)
    require(y-u@bu.T,y)
    require(av-bu,av)
    p = av.square().sum(0)
    if (p <= 0).any():
        raise ValueError('zero initial mode')
    q = av / p.sqrt()
    require(q.T@q-torch.eye(r,device=q.device,dtype=q.dtype),torch.eye(r,device=q.device,dtype=q.dtype))
    if time == 0:
        return a.clone(), b.clone()
    decay = 2*s*time
    log_den = torch.logaddexp(torch.log(-torch.expm1(-decay)),torch.log(s)-torch.log(p)-decay)
    w = torch.exp((torch.log(s)-log_den)/2)
    aa,bb = ((q*w)@vh).to(a.dtype), ((u*w)@q.T).to(b.dtype)
    if not torch.isfinite(aa).all() or not torch.isfinite(bb).all():
        raise ValueError('nonfinite output')
    return aa,bb
