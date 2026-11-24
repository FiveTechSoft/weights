"""Training-to-surgery: offline donor-assisted analysis, not donor-free synthesis."""
from __future__ import annotations
import torch

@torch.no_grad()
def factorize(base, donor):
    if set(base)!=set(donor):
        raise ValueError("checkpoint keys differ")
    if not all(base[k].shape==donor[k].shape for k in base):
        raise ValueError("checkpoint shapes differ")
    for state in (base,donor):
        if not torch.equal(state["lm_head.weight"],state["transformer.wte.weight"]):
            raise ValueError("tied embedding mismatch")
    factors={}
    for name,w in base.items():
        if name=="lm_head.weight":
            continue
        delta=donor[name].double()-w.double()
        if not torch.isfinite(delta).all():
            raise ValueError(f"nonfinite delta: {name}")
        if delta.ndim==2:
            u,s,vh=torch.linalg.svd(delta,full_matrices=False)
            factors[name]=(u,s,vh)
        elif delta.ndim==1:
            factors[name]=delta
        else:
            raise ValueError(f"unsupported tensor: {name}")
    return factors

def group(name):
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "embedding_norm"

@torch.no_grad()
def compile_delta(base,factors,rank=None,groups=None,placebo=False,seed=1337):
    """Rank cap per matrix; vector deltas exact (including placebo)."""
    result={}
    stats={}
    rng=torch.Generator().manual_seed(seed)
    for name,f in factors.items():
        selected=groups is None or group(name) in groups
        if isinstance(f,tuple):
            u,s,vh=f
            r=len(s) if rank is None else min(rank,len(s))
            if not selected:
                r=0
            if placebo and r:
                u=torch.linalg.qr(torch.randn(u.shape[0],r,generator=rng,dtype=u.dtype),mode="reduced")[0]
                vh=torch.linalg.qr(torch.randn(vh.shape[1],r,generator=rng,dtype=vh.dtype),mode="reduced")[0].T
            delta=(u[:,:r]*s[:r])@vh[:r]
            energy=float(s[:r].square().sum())
            total=float(s.square().sum())
            stats[name]={"group":group(name),"rank":r,"max_rank":len(s),
                         "energy":energy,"total_energy":total,
                         "factor_values":r*(u.shape[0]+vh.shape[1]+1),
                         "dense_values":base[name].numel()}
        else:
            delta=f if selected else torch.zeros_like(f)
            stats[name]={"group":group(name),"rank":None,"max_rank":None,
                         "energy":float(delta.square().sum()),"total_energy":float(f.square().sum()),
                         "factor_values":f.numel() if selected else 0,"dense_values":f.numel()}
        result[name]=(base[name].double()+delta).to(base[name].dtype).contiguous()
    result["lm_head.weight"]=result["transformer.wte.weight"].clone()
    return result,stats
