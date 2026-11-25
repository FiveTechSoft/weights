# N6b: preserve the residual mean as well as its norm
N6 run 45567498320: bigram fidelity passes (max error 1.3472e-6),
sparse trigram fails (max error .9381) despite correct retrieval.
Root cause candidate: a 64D orthonormal coordinate vector U^T e_i
preserves norm but not coordinate sum. LayerNorm subtracts the mean
of the entire residual, corrupting the constant anchor and the MLP
linear identity and conjunction. Local synthetic exhaustive 4225-pair
residual test reproduced max probability error .0594493; decoding
the 64D coordinates back to the 65D centered simplex gave 3.26275e-7.
First-position synthetic error .0467863 -> 1.38425e-7. This local test
uses synthetic logits and ideal retrieval, not corpus or native attention.
Fix only the attention output projection and conjunction input: O=U
maps the retrieved 64D coordinates back to the centered simplex in
65:130. Current token and previous token now each sum to zero and
have squared norm 64/65. Preserve both mean and norm through retrieval.
Same corpus, selected pairs, architecture, exact target probabilities,
all thresholds and scoring as N6. Additional audits: pre-MLP mean,
variance deviation, gate amplitude near 0 or 16, no positive gate at
position zero and at most one positive gate elsewhere. Do not claim
causal diagnosis confirmed until native test passes. Preserve N6 source.
No training, donor, regression or new features. Next step only after
fidelity passes: autonomous selection on a disjoint selection split,
then context extension; do not call fixed pair lookup RSI.
