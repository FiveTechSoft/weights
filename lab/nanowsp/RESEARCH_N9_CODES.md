# N9 relation identities: independent ideal-code audit
2026-11-25. Native N9 run 45595709146 queued at initial inspection; this is a separate locally executed PyTorch 2.10.0 CPU test, not native checkpoint evaluation. Source: natural_n9.py at e425ed4c5276de58cda5193f974cdb0c5b2cb7c7.
All 1078 parent codes are distinct, zero-sum to FP64 error 8.8818e-16, squared norm 10/3 to error 4.4409e-16, minimum dot-product match margin .9999999999999996. The 8 class codes add norm 7/8. 512 seeded synthetic distinct parent/class detector keys, evaluated against all 8624 parent/class combinations: 4415488 comparisons, zero incorrect threshold activations (>1e-4), max FP32 activation error 4.57763671875e-5, unavailable-position max 0.0. 100000 seeded arbitrary identity replacements via old+(new-old): maximum coordinate error 1.4901161193847656e-8, squared norm error 4.76837158203125e-7, max absolute coordinate sum 4.76837158203125e-7. These are arithmetic checks, not independent statistical examples or semantic abstractions.
Reproduction:
```python
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
n=1078;i=torch.arange(n)
code=torch.cat([F.one_hot((i//6**k)%6,6).double()-1/6 for k in range(4)],1)
cls=torch.eye(8,dtype=torch.double)-1/8
G=code@code.T;G.fill_diagonal_(-float('inf'))
print(len(code.unique(dim=0)),code.sum(-1).abs().max().item(),
      (code.square().sum(-1)-10/3).abs().max().item(),10/3-G.max().item())
keys=torch.randperm(n*8,generator=torch.Generator().manual_seed(909))[:512]
p=torch.arange(n*8)//8;c=torch.arange(n*8)%8
x=torch.cat([code[p],cls[c]],1).float();w=x[keys]
z=32*(x@w.T-(10/3+7/8)+.5);a=F.gelu(z)
target=torch.arange(n*8)[:,None]==keys[None,:]
print(a.numel(),((a>1e-4)!=target).sum().item(),
      (a-16*target.float()).abs().max().item(),F.gelu(z-32).abs().max().item())
g=torch.Generator().manual_seed(910)
old=torch.randint(n,(100000,),generator=g);new=torch.randint(n,(100000,),generator=g)
q=code.float();r=q[old]+(q[new]-q[old])
print(len(old),(r-q[new]).abs().max().item(),
      (r.square().sum(-1)-10/3).abs().max().item(),r.sum(-1).abs().max().item())
```
Native attention, LayerNorm, parent routing, learned count distributions, and long-context prediction remain unverified. No N9 threshold or source code changed. The relation IDs are assigned, not learned semantic categories. No claim of RSI.
