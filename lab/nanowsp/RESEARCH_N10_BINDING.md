# N10 supplementary synthetic binding check
Local NumPy experiment, 4096 independently sampled episode-specific random permutations of 16 symbols, one uniform query per episode. Fixed alpha=8 and uniform prior. Each episode contains k observations of every key, shuffled in order. Retrieve values whose key equals query, update distribution using N10 count mixture. Negative control independently permutes keys within each episode, retaining values and key frequencies. This is an external algorithm with explicitly supplied key/value observations, not a native Transformer or natural-language ICL result. The test was exploratory, not preregistered.
| Observations per key | NLL | Accuracy | Shuffled NLL | Shuffled accuracy |
|---:|---:|---:|---:|---:|
|0|2.772588722|.059082031|2.772588722|.059082031|
|1|1.791759469|1|2.820367410|.063720703|
|2|1.386294361|1|2.858911403|.066650391|
|4|.980829253|1|2.911246832|.061523438|
All probability sums exactly 1 in this FP64 test. Same seeded episodes reused across k; not independent repeated samples. Uniform prior argmax uses symbol 0 tie break. Mapping is deterministic within each episode, with complete clean observations, so perfect accuracy with one example per key is expected. No generalization beyond this task established.
Reproduce:
```python
import numpy as np
rng=np.random.default_rng(1010); N=4096; V=16; alpha=8
perm=np.argsort(rng.random((N,V)),axis=1)
q=rng.integers(V,size=N); y=perm[np.arange(N),q]
for k in [0,1,2,4]:
    keys=np.tile(np.arange(V),k)[None,:].repeat(N,0)
    keys=np.take_along_axis(keys,np.argsort(rng.random(keys.shape),axis=1),axis=1)
    vals=np.take_along_axis(perm,keys,axis=1)
    def predict(keys):
        mask=keys==q[:,None]
        counts=np.zeros((N,V))
        np.add.at(counts,(np.broadcast_to(np.arange(N)[:,None],keys.shape),vals),mask)
        return (alpha/V+counts)/(alpha+mask.sum(1)[:,None])
    p=predict(keys)
    shuffled=np.take_along_axis(keys,np.argsort(rng.random(keys.shape),axis=1),axis=1)
    c=predict(shuffled)
    print(k,-np.log(p[np.arange(N),y]).mean(),(p.argmax(1)==y).mean(),
          -np.log(c[np.arange(N),y]).mean(),(c.argmax(1)==y).mean(),
          np.max(abs(p.sum(1)-1)))
```
Native N10 is not proposed until the statistical natural-language feasibility gate in PLAN_N10.md passes. This sanity check does not override that gate. N10 run 45599656542 queued at inspection.
