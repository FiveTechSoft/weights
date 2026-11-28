# N5: construct linguistic features from corpus tensors
N4 45560449819 numerical gate passed (relative residuals .003742,
.002804,.003496), predictive gate failed: NLL 2.189383/2.194681
vs N3 2.207301/2.175514. Do not tune solver further.
Test representation rather than solver. Same 65-char corpus/hash, native
3230208-parameter architecture, teacher split and held-out slices as N1.
Reuse N1 causal retrieval: current token, lags 1,2,3, uniform prefix.
For each of six local lag pairs compute train-only 65x65x65 counts.
Smooth against the average of the two single-lag conditionals (strength
8); center output log probabilities and double-center context axes.
SVD of flattened interaction tensor yields five output contrast
directions; SVD of each associated context matrix yields four paired
factors. Thus 120 products. Eight additional current/prefix products
from pooled lags 8,16,32,48 (two contrasts x four factors).
Normalize factor maximum magnitude to one; project factors to existing
retrieval subspaces. Prefix products represent averaged retrieved
features, not exact distant token retrieval.
Compile signed linear features (512 GELUs) and 128 bilinear features
(512 GELUs) using four-GELU product identity, with epsilon .1. Keep
paired output coefficients tied so the algebraic products remain explicit.
Fit a single normalized ridge to teacher logits on the same 2048 windows
as N2_1. Compare additive-only, seeded random products with identical
lag pairs and widths, and corpus-derived products. No random features
in the corpus arm. Same fitting objective and regularization for all.
Save native checkpoints, exact reload, NLL/KL/RMSE on both development
slices, zero-product ablation, ideal product vs native activation error
on sampled training windows. No external lookup at inference.
Primary gate: corpus arm beats both additive and random by >=.05 NLL
on both slices; also report teacher KL and legacy >=50% gap recovery.
Full language gates unchanged, final reference tail only if both pass.
The teacher is still local and cannot establish long-context knowledge;
prefix factors are an exploratory representation, not proof of context.
Earlier slices repeatedly used; no claim of fresh independent evidence.
No optimizer/autograd/backprop; statistical SVD and ridge fitting only.
