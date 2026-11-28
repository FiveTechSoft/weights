# Candidate law: reduced-rank predictive residual correction
Prior run 45477023823 shows rank4 VO preserves most measured capability
on trained background, not that corpus frequencies alone explain language.
Test one constructive law, no trained VO targets or late weights in fitting.
Frozen step100 background; compute first-layer attention-weighted normalized
input z_h at final position, using existing Q/K. Remove native VO at that
position only for fitting baseline logits; use the actual downstream
Jacobian of logits with respect to first attention residual, computed by
autograd (no parameter optimizer or gradient updates). Fit an additive
rank4 map per head from corpus next-character targets by a closed-form
linearized least-squares approximation. To avoid expensive Jacobian
construction, first test a simpler explicit output-coordinate alignment:
use the tied unembedding as a linear surrogate of residual-to-logits.
Let X=[z_0,...,z_3], Y=onehot(next)-softmax(native logits).
Center neither X nor Y (no bias allowed). Ridge solve X^T X + lambda I
with lambda=.01 mean Gram diagonal; map predicted logit correction to
residual by pseudoinverse of tied unembedding, with fixed ridge .01
mean diagonal. Truncate each 256x256 correction to rank4 SVD. Add to
native VO, refactor to rank64 (native capacity); report truncation error.
Use training corpus only, 8192 fixed windows length64 seed 1729.
Evaluate fixed doses 0.25,1 (not selected on confirmation), plus matched
random output permutation of correction, early native and late reference.
Primary criterion on both existing held-out slices: NLL improvement
and paired intact improvement CI lower >0, context gain CI lower >.01,
and better NLL than matched permutation. Report recovery vs late,
do not claim full LM or donor-free from initialization: step100 is trained.
No late weights used in construction; late only evaluated after fitting.
This is a candidate law, not established identity; tied readout surrogate
ignores intervening nonlinear layers. If it fails, inspect alignment rather
than search arbitrary doses. Save real heatmaps of early effective map,
constructed correction and late-minus-early for visual inspection only.
