# Joint context
Same 6800 parameter native NanoGPT. Raw streams length64, alphabet4.
Infer joint conditional table from counts for all unordered lag pairs
0..16 (136); compare to all single lags and unigram on identical target
positions 17..63. Half-count smoothing. Select minimum training NLL pair,
require independent validation improvement >=.10 nat over best single
or unigram and >=.03 over second-best pair. No donor or gradient.
Eight fixtures: lag pairs (0,3),(2,7),(5,11),(9,16), each with modular
sum and XOR output plus symmetric noise .16 (correct probability .84).
These interactions have uniform single-input marginals under independent
uniform inputs; recurrence may induce additional correlations, measured.
Train/validation/test 512/256/256 independent streams. Require all
correct lags, native probabilities equal estimated table within 1e-5
on arbitrary histories and test, test improvement >=.10 over best
single, average excess NLL over true law <=.03. Reject IID and
single-lag controls. Native two heads retrieve into separate scratch
spaces; first MLP clears position/current token, retains pair as two
centered one-hot codes. Second MLP solves 16x16 GELU interpolation
to produce centered log probabilities with equalized norms, tied head.
Verify strict reload exact; native sampling continuation and conditional
TV averaged <=.08. Save all models and report, condition number.
No claim of compressed representation: 16 contexts are explicitly
interpolated. Next stage must reduce context table growth, then natural
text on reference split; no more synthetic table scaling.
