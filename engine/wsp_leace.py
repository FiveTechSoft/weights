"""Certified rank-one linear erasure under a supplied output covariance."""
import torch


def erase_direction(weight, direction, covariance, *, atol=1e-6):
    """Return erased weights and a certificate, or reject.

    Implements covariance-directed erasure for centered outputs with
    covariance Sigma, subject to u^T P = 0. This is a rank-one linear
    eraser, not a general affine/multiclass LEACE implementation.
    """
    w = weight.double()
    u = direction.to(device=w.device, dtype=w.dtype)
    s = covariance.to(device=w.device, dtype=w.dtype)
    if w.ndim != 2 or u.shape != (w.shape[0],) or s.shape != (w.shape[0],)*2:
        raise ValueError("incompatible erasure dimensions")
    if not all(torch.isfinite(x).all() for x in (w,u,s)):
        raise ValueError("nonfinite erasure inputs")
    if not torch.allclose(s,s.T,rtol=1e-7,atol=1e-10):
        raise ValueError("covariance must be symmetric")
    s = (s+s.T)/2
    eig = torch.linalg.eigvalsh(s)
    scale = float(eig.abs().max())
    if scale == 0 or float(eig.min()) < -1e-8*scale:
        raise ValueError("covariance must be nonzero positive semidefinite")
    norm = u.norm()
    if float(norm) == 0:
        raise ValueError("zero concept direction")
    u = u/norm
    denom = u @ s @ u
    if float(denom) <= 1e-12*scale:
        raise ValueError("degenerate concept covariance")
    v = s @ u / denom
    erased = (w-v[:,None]*(u @ w)[None,:]).to(weight.dtype)
    leak = float((u @ erased.double()).norm())
    before = float((u @ w).norm())
    if not torch.isfinite(erased).all() or leak > atol*max(1.,before):
        raise ValueError("rounded weights fail erasure certificate")
    # For centered h: E||v(u^T h)||² = ||v||² u^T Sigma u.
    # Orthogonal projection is the minimum Euclidean movement per sample;
    # this covariance-based direction is not generally less distortive.
    return erased, {"leak_before":before,"leak_after":leak,
        "covariance_rank":int(torch.linalg.matrix_rank(s)),
        "expected_displacement":float(v.square().sum()*denom),
        "orthogonal_displacement":float(denom),
        "certificate":"rank_one_linear_erasure"}
