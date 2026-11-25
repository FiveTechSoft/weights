# N3: Fisher-aware local compilation
N2 run 45558581703 reproduced N1; final gap recovery 16.83%/14.26%,
failed 50% gate. Best stage N2_1 NLL 2.238625/2.265044; later stages
reduce logit RMSE but increase NLL. This motivates a metric change,
not proof that Fisher solves the representation bottleneck.
Rebuild N2_1 exactly using existing build and conditioned_fit.
Use same frozen statistical teacher, same corpus split, same architecture,
same random feature recipe as N2 for layers 1..3 (zero-based).
For computational cost use positions 7,15,...,63 from each of the same
2048 windows (16384 positions) for BOTH correction arms. No extra
features. Use matched squared-logit and softmax Fisher arms, each with
three sequential layer corrections. Normalize feature columns by RMS.
Derive analytic Jacobian of final tied readout and LayerNorm restricted
to centered first65 residual coordinates. For Fisher arm solve
(J^T F J + ridge) delta = J^T(q-p) in feature-weight space using
matrix-free conjugate gradient, F=diag(p)-pp^T; p=current, q=teacher.
Euclidean arm uses J^T J and J^T(centered teacher-current logits).
Ridge .001 * n, 24 CG iterations maximum, report residual norm.
No autograd or backpropagation; this is iterative numerical optimization
of an explicit local quadratic, not a claim of zero mathematical fitting.
For each correction use fixed doses 0,.125,.25,.5,1 on a separate
deterministic 128-window audit drawn from permitted fit region, disjoint
from fit windows by using last 8192 characters exclusively for audit.
Choose minimum native teacher KL for Fisher, centered logit MSE for
Euclidean; ties prefer lower dose. Neither eval slice influences choice.
Evaluate every stage on both prior validation and confirmation slices.
Primary gate: final Fisher arm improves NLL by >=.05 vs N2_1 AND
vs matched Euclidean arm on both slices; teacher KL also improves vs
N2_1 on both. Keep N1-gap recovery >=50% as separate legacy gate.
Report convergence, doses, KL, RMSE, NLL, exact checkpoint reload.
Full language gates remain unchanged; final reference tail only if
both pass. No claim of long-context knowledge: teacher uses four chars.
Previously examined slices are development benchmarks, not fresh
independent generalization evidence.
