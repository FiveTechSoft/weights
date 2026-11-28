# Local Jacobian law
Prior law (45482139927) failed both splits; unembedding surrogate and excessive
norm are plausible causes, not established causes. Replace surrogate by the
actual derivative through all downstream layers and every sequence position.
Freeze step100. Use top four right singular directions of each failed
corpus-derived correction (computed again from training only) as fixed input
basis; output directions are solved, not copied from step500. Parameterize
additive effective VO map per head as C_h = D_h R_h, D_h 256x4.
To keep exact native rank64, instead choose R_h within row space of native
V_h: project candidate input directions into that space, orthonormalize,
and compute coefficients T_h such that R_h=T_h V_h. Then change only O_h
by D_h T_h. This makes the intervention exact without SVD refactor error.
Use all positions, not merely final position. Fit on 512 fixed training
windows (length64), held-out training audit 128 disjoint sampled windows.
Use centered-logit linearization and empirical Fisher/Gauss-Newton:
gradient g=J^T(p-y), curvature J^T(diag(p)-pp^T)J; solve damped linear
system for D using matrix-free conjugate gradients with at most 20 iterations.
Alternatively use full Jacobian on 128 windows and solve in sample space
with ridge; choose one implementation before evaluation. Automatic
differentiation computes sensitivities, no optimizer steps on model weights.
This is a supervised second-order local fit, not gradient-free learning
or a general law of training. Use trust radius based on native effective
map norm (joint Frobenius correction <= .5 native norm) and fixed doses
.25,1. Compare exact finite change to predicted linear logits on training
audit (relative error <=.25 at .25 dose, positive predicted NLL reduction).
If audit fails, do not claim local approximation valid. Evaluate native,
candidate and equal-norm output-permuted control on both prior reserved
slices, report NLL, paired intact improvement CI, context gain CI,
recovery relative to step500 (only load after construction). Joint criterion
same as previous law; audit and confirmation both required. No new
architecture, no late donor, no assertion of from-scratch synthesis.
