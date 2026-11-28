"""Convex minimum-norm readout surgery via the nonnegative QP dual.

The dual supplies a lower bound in the chosen protected nullspace even if
the numerical optimizer stops early. No gradient through the LLM is used.
"""
import math

import numpy as np
import torch

from engine.wsp_native_readout import ReadoutRejected


@torch.no_grad()
def compile_qp(base, skill, protected, logits, labels, outside_max, *,
               max_delta=10., margin=2., atol=1e-3, rtol=1e-8, maxiter=4000):
    from scipy.optimize import minimize

    if not all(math.isfinite(v) and v > 0 for v in (max_delta, margin, atol, rtol)) or rtol >= 1:
        raise ValueError("invalid tolerances or budget")
    if type(maxiter) is not int or not 1 <= maxiter <= 10000:
        raise ValueError("invalid iteration limit")
    if any(not t.is_floating_point() or not torch.isfinite(t).all()
           for t in (base, skill, protected, logits, outside_max)):
        raise ValueError("finite floating matrices required")
    if base.ndim != 2 or skill.ndim != 2 or protected.ndim != 2:
        raise ValueError("matrix inputs required")
    k, d = base.shape
    n = len(skill)
    if (k < 2 or d < 1 or n < 1 or not len(protected) or n*k > 4096
            or skill.shape != (n,d) or protected.shape[1] != d
            or logits.shape != (n,k) or outside_max.shape != (n,)
            or labels.shape != (n,) or labels.dtype != torch.long
            or not ((labels >= 0) & (labels < k)).all()):
        raise ValueError("invalid shapes, labels or constraint count")
    w,x,p,l,b = (t.detach().cpu().double() for t in
                 (base, skill, protected, logits, outside_max))
    y = labels.cpu().numpy()
    _,s,vh = torch.linalg.svd(p, full_matrices=False)
    q = vh[s > rtol*s[0]]
    z = (x - (x @ q.T) @ q).numpy()
    lnp, bnp = l.numpy(), b.numpy()
    # Each constraint is <B_i outer z_sample, delta> >= rhs_i.
    coeff, sample, rhs = [], [], []
    for i,c in enumerate(y):
        a = np.zeros(k)
        a[c] = 1.
        coeff.append(a)
        sample.append(i)
        rhs.append(margin + bnp[i] - lnp[i,c])
        for j in range(k):
            if j != c:
                a = np.zeros(k)
                a[c], a[j] = 1., -1.
                coeff.append(a)
                sample.append(i)
                rhs.append(margin + lnp[i,j] - lnp[i,c])
    B, ids, rhs = np.asarray(coeff), np.asarray(sample), np.asarray(rhs)
    norms = np.linalg.norm(B,axis=1)*np.linalg.norm(z[ids],axis=1)
    if np.any((norms <= 1e-12) & (rhs > atol)):
        raise ReadoutRejected("zero projected constraint cannot meet margin",
                              {"protected_rank":len(q),"backward_calls":0})
    keep = norms > 1e-12
    B, ids, rhs, norms = B[keep], ids[keep], rhs[keep], norms[keep]
    C = B/norms[:,None]
    G = (C @ C.T)*(z[ids] @ z[ids].T)
    r = (rhs + 2*atol)/norms

    def objective(lam):
        g = G @ lam
        return float(.5*lam @ g-r @ lam), g-r

    if len(r):
        opt = minimize(objective,np.zeros(len(r)),jac=True,method="L-BFGS-B",
                       bounds=[(0.,None)]*len(r),
                       options={"maxiter":maxiter,"ftol":1e-14,"gtol":1e-10,"maxls":50})
        lam = np.maximum(opt.x,0.)
        iterations, status = int(opt.nit), str(opt.message)
    else:
        lam = np.zeros(0)
        iterations, status = 0, "all constraints already satisfied"
    delta = torch.from_numpy((C.T*lam) @ z[ids])
    delta -= (delta @ q.T) @ q
    # Bound for nominal margins relaxed by the actual acceptance tolerance,
    # with exact nullspace protection (not arbitrary protected-logit errors).
    lower = float(((rhs-atol)/norms) @ lam - .5*delta.square().sum())
    lower_norm = math.sqrt(max(0.,2*lower))
    candidate = (w+delta).to(dtype=base.dtype,device=base.device)
    actual = candidate.cpu().double()-w
    scores = l+x@actual.T
    row = torch.arange(n)
    correct = scores[row,labels.cpu()]
    scores[row,labels.cpu()] = -torch.inf
    violation = float((margin+torch.maximum(scores.max(1).values,b)-correct).clamp_min(0).max())
    cert = dict(solver="convex_dual_lbfgsb", iterations=iterations, solver_status=status,
                protected_rank=len(q), nullspace_dimension=d-len(q),
                margin=margin, margin_violation=violation,
                protected_error=float((p@actual.T).abs().max()),
                delta_frobenius=float(actual.norm()), max_delta=max_delta,
                dual_lower_norm=lower_norm, continuous_delta_frobenius=float(delta.norm()),
                bound_scope="exact chosen protected nullspace; margin minus atol",
                backward_calls=0, optimizer_steps=0)
    keys = ("margin_violation","protected_error","delta_frobenius","dual_lower_norm")
    if (not all(math.isfinite(cert[key]) for key in keys)
            or violation > atol or cert["protected_error"] > atol
            or cert["delta_frobenius"] > max_delta):
        raise ReadoutRejected("QP did not certify a rounded update within budget",cert)
    return candidate,cert
