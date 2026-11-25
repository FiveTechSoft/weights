# N8: residual-driven extension of N7
Keep the N7 candidate selection and count-source split unchanged.
Hypotheses are observed triples (a,b,c) extending the current pair
(b,c), not preselected by frequency. Use count-source teacher.row
((a,b,c)) as candidate distribution and N7 sparse distribution
for (b,c) as parent. Discovery and acceptance data remain disjoint
from counts and development slices. For each triple require >=8
occurrences and positive summed log ratio on BOTH selection splits.
Rank by discovery gain, stable integer context tie-break; accept up
to 512. Compare 64,128,256,512 nested budgets (clamped to accepted).
No tuning against development sets; record all observed candidates.
Control: same number of most frequent count-source triples.
Native implementation extends N7 block 0 before its MLP: retrieve
lag 2 via a second full-width head and decode to centered simplex
coordinates 130:195. Preserve current and lag1 identity. Use a
position-availability scalar pair to distinguish t=0, t=1, t>=2
while maintaining fixed norm (orthogonal zero-mean marker pairs).
Use block 0 to encode the N7 prediction and retain a centered
three-token identity scratch code plus position markers, with fixed
norm. Block 1 then removes scratch and installs selected triple
correction. For selected triples add centered log-ratio and anchor
difference relative to exact N7 parent. Unselected contexts back off.
Implementation may use alternative exact decomposition documented
in code, but no regression, donor or backprop. Compare native
probabilities to explicit sparse hierarchy, audit all selected
triples and deterministic sample of 4096 unselected triples, at
positions 2,7,31, plus first two positions, future and prefix
invariance. Verify native N7 reproduction and retained parent
predictions for unselected contexts. NLL parity <=1e-4 all stages,
max probability error <=1e-4, exact reload. Growth gate nonnegative
increments on both reused development slices (1e-4 tolerance).
Primary predictive gate >=.05 gain over N7 on both slices;
secondary >=.01 gain vs matched frequency control on both.
No claim of independent generalization from reused development
slices. Longer local context still is not distant context or RSI.
No reserved tail unless language gate passes.
