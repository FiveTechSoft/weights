"""Native readout feasibility using linear margin constraints.

Cyclic halfspace projection in a protected nullspace. This is a numerical
constraint solver, not gradient training or a proof of infeasibility when
the iteration budget expires. The checkpoint contains only ordinary weights.
"""
import math
import torch

from engine.wsp_native_readout import ReadoutRejected


@torch.no_grad()
def compile_margins(base, skill, protected, logits, labels, outside_max, *,
                    max_delta=10., margin=2., atol=1e-3, rtol=1e-8,
                    max_sweeps=200):
    """Change selected rows only; beat other selected and all frozen logits."""
    if not all(math.isfinite(v) and v>0 for v in (max_delta,margin,atol,rtol)) or rtol>=1:
        raise ValueError("invalid budget or tolerance")
    if type(max_sweeps) is not int or not 1<=max_sweeps<=10000:
        raise ValueError("invalid sweep budget")
    if any(not t.is_floating_point() or not torch.isfinite(t).all()
           for t in (base,skill,protected,logits,outside_max)):
        raise ValueError("finite floating inputs required")
    if base.ndim!=2 or skill.ndim!=2 or protected.ndim!=2:
        raise ValueError("matrix inputs required")
    k,d=base.shape
    n=len(skill)
    if k<2 or d<1 or n<1 or not len(protected) or skill.shape[1]!=d or protected.shape[1]!=d or logits.shape!=(n,k) or outside_max.shape!=(n,) or labels.shape!=(n,):
        raise ValueError("shape mismatch")
    if labels.dtype!=torch.long or not ((labels>=0)&(labels<k)).all():
        raise ValueError("invalid labels")
    w,x,p,l,b=(t.detach().to(device=base.device,dtype=torch.float64)
               for t in (base,skill,protected,logits,outside_max))
    labels=labels.to(w.device)
    _,s,vh=torch.linalg.svd(p,full_matrices=False)
    q=vh[s>rtol*s[0]]
    z=x-(x@q.T)@q
    norms=z.square().sum(1)
    delta=torch.zeros_like(w)
    # Reserve slack for conversion to the checkpoint dtype.
    m=margin+2*atol
    rows=torch.arange(n,device=w.device)
    def residual(update):
        current=l+x@update.T
        correct=current[rows,labels]
        wrong=current.clone()
        wrong[rows,labels]=-torch.inf
        return float((margin+torch.maximum(wrong.max(1).values,b)-correct).clamp_min(0).max())
    used=0
    for sweep in range(max_sweeps):
        used=sweep+1
        for i in range(n):
            c=int(labels[i])
            a=z[i]
            a2=float(norms[i])
            if a2<=1e-24:
                continue
            violation=float(b[i]+m-l[i,c]-delta[c]@a)
            if violation>0:
                delta[c].add_(a,alpha=violation/a2)
            for j in range(k):
                if j==c:
                    continue
                violation=float(m+l[i,j]-l[i,c]-(delta[c]-delta[j])@a)
                if violation>0:
                    change=a*(violation/(2*a2))
                    delta[c]+=change
                    delta[j]-=change
        # Orthogonal projection onto the norm budget and protected nullspace.
        delta-= (delta@q.T)@q
        norm=float(delta.norm())
        if norm>max_delta:
            delta*=max_delta/norm
        if residual(delta)<=atol/4:
            break
    result=(w+delta).to(base.dtype)
    actual=result.double()-w
    cert=dict(solver="cyclic_halfspace_projection",sweeps=used,
              protected_rank=len(q),nullspace_dimension=d-len(q),
              margin=margin,margin_violation=residual(actual),
              protected_error=float((p@actual.T).abs().max()),
              delta_frobenius=float(actual.norm()),max_delta=max_delta,
              backward_calls=0,optimizer_steps=0)
    if (not torch.isfinite(result).all() or
        not all(math.isfinite(cert[key]) for key in ("margin_violation","protected_error","delta_frobenius")) or
        cert["margin_violation"]>atol or cert["protected_error"]>atol or
        cert["delta_frobenius"]>max_delta):
        raise ReadoutRejected("margin solver did not certify a feasible rounded update within budget",cert)
    return result,cert
