# N6: exact elementary probability law
N5 45562258626 passed representation comparison but worse absolute NLL.
N6 removes all regression and random features. Same native 3230208
parameters, fixed corpus, teacher counts on first 80% of training 90%.
Define centered logits L=log(p)-mean(log(p)); encode L in first 65
coordinates and opposite anchors +/-sqrt((C-||L||^2)/2), where C is
a shared squared norm across all target rows. Final LayerNorm gain
sqrt(C/256+eps) makes the tied centered-simplex readout equal L.
Use paired GELU identity GELU(x)-GELU(-x)=x to map current identity
to that vector in a single MLP. No fitting of weights.
Second model: same bigram law plus explicit correction for 512 most
frequent preceding/current pairs from counts only. Retrieve preceding
char in full 64-dimensional zero-sum orthonormal basis, then use a
large-margin GELU conjunction with a position marker to exclude t=0.
For selected pairs add target trigram vector minus bigram vector,
including anchor corrections. Unselected pairs back off exactly.
This is an explicit finite lookup compiled into weights, not evidence
of unseen relation discovery. Compare full statistical trigram too.
Exhaustively evaluate all 65 bigram rows and all 4225 pair contexts
(native probability max error <= 1e-4), including unselected pairs;
test position variation, prefix and future invariance, first-position
backoff, selected attention mass, native norm error and reload parity.
Measure NLL on both reused development slices against own exact
statistical references (absolute difference <= 1e-4), and compare
sparse trigram vs bigram (gain >= .05 both). Report native vs full
trigram gap; do not tune selection on development data.
No external tables at inference, no regression/optimizer/backprop/donor.
Full language gates unchanged; tail only if both pass. This experiment
isolates the elementary compilation law, not long context or ICL.
