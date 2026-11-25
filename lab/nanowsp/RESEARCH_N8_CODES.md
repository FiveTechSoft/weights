# N8 compact code audit (2026-11-25)
Independent local NumPy/SciPy test of the ideal compact scratch code, not native attention, corpus selection, or checkpoint evaluation. N8 source pinned to ae04b03a79b62ba8e45528abad3835299bcf92b5, native run 45589442210 pending at time of audit.
For i=0..64 define c(i)=concat(onehot8(i//9)-1/8, onehot9(i%9)-1/9).
FP64: 65 distinct rows; max absolute row sum 1.6653345369377348e-16; max squared norm error relative to 2-1/8-1/9 = 2.220446049250313e-16; minimum squared distance between distinct codes 1.9999999999999991. Gram diagonal 1.7638888888888884; max off-diagonal .7638888888888888; match margin ~1. FP32 max absolute row sum 6.7055225e-8.
For three concatenated codes the true match has sum of three diagonal dot products; every nonmatch loses at least 1. The position marker adds +.5 if available and -.5 otherwise. A gain of 32 yields approximately 16 for the true match and <=-16 for all nonmatches or unavailable positions. GELU saturates numerically.
Executable reproduction (requires numpy and scipy):
```python
import numpy as np
from scipy.special import ndtr
i=np.arange(65)
X=np.concatenate([np.eye(8)[i//9]-1/8,np.eye(9)[i%9]-1/9],axis=1).astype(np.float32)
G=X@X.T
cn=np.float32(2-1/8-1/9)
a,b,c=np.indices((65,65,65))
a,b,c=a.ravel(),b.ravel(),c.ravel()
keys=np.random.default_rng(808).choice(65**3,512,replace=False)
maxerr=0.;wrong=0;early=0.;n=0
for k in keys:
    j,h,l=k//4225,(k//65)%65,k%65
    z=32*(G[a,j]+G[b,h]+G[c,l]-3*cn+.5)
    y=z*ndtr(z)
    target=(a==j)&(b==h)&(c==l)
    maxerr=max(maxerr,float(np.max(abs(y-16*target))))
    wrong+=int(np.sum((y>1e-4)!=target))
    early=max(early,float(np.max(abs((z-32)*ndtr(z-32)))))
    n+=len(z)
print(n,maxerr,wrong,early)
```
Observed: 140608000 detector/context comparisons, maximum activation error 0.0, wrong activation count 0, unavailable-position maximum 0.0. These are ideal-code arithmetic checks, not independent statistical samples. 512 detector keys are seeded synthetic distinct triples, not N8 selected corpus keys. In real arithmetic GELU is not a hard gate; zeros here are numerical saturation. Algebraic margin applies to all possible keys. Native LayerNorm, attention, learned count distributions, parent preservation and predictive gate remain to be verified. No change to N8 protocol or code.
