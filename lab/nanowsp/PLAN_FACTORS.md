# Distributed factor test (last synthetic bridge before natural text)
Previous joint run 45552935023 passed, but explicitly interpolates 16 contexts.
Keep the 6800 parameter runtime, two heads, known fixed lags (2,7) to
isolate representation learning, not repeat lag discovery. Model class:
centered logits L(a,b,:) = sum_r U[a,r] V[b,r] B[:,r], r=3, fixed
orthonormal Walsh output contrasts B. Fit factors only from observed
conditional counts in raw recurrent streams, excluding all targets whose
context lies in {2,3}x{2,3}. No hidden-context labels in the fit, no
gradient, no donor. This is a restricted rank-one-per-contrast model,
not universal tensor decomposition. Recover each slice by rank-one CUR
from first two rows and columns, via SVD of 2x2 anchor intersection.
Compare to additive logit fit on the same observed cells and uniform
backoff for unseen cells. Eight fixed seeded fixtures, factors randomly
signed magnitudes .6..1.4, independent train 4096 and test 512 length64
streams. Raw sequences include held-out contexts; their target counts
are masked before fit, no other use of train data. Test all 16 context
pairs on arbitrary histories for compiler fidelity and score only
unseen pairs on independent natural recurrence test. Report observed
count coverage, factor vs oracle NLL, additive and uniform NLL.
Require every fixture native probability max error <=.002 vs factored
conditional, held-out mean excess NLL vs true <=.06, mean improvement
>=.05 over additive and uniform, strict reload, zero-product ablation
near uniform (max deviation .002). Explicit 21 effective factor
degrees of freedom vs 48 centered table logits (fixed B); dense model
size unchanged. Native multiplication via paired GELU even differences:
g(e(u+v))+g(-e(u+v))-g(e(u-v))-g(-e(u-v)) ~ 8 phi(0)e²uv.
Fixed e=.02, output anchor +/-256 approximates constant final LN scale;
report numerical approximation, not exact multiplication. No hidden
context-specific neurons in second block. Include counterexample:
perturb only unseen true logits, same observed statistics cannot
identify it; no universal completion claim. Save all weights/reports.
Next step natural text on reference split, not further synthetic suites.
