# Corpus-only lag/permutation discovery
Build on verified retrieval run 45510817778. Input to discovery is ONLY
a batch of independent raw token streams (length 64, alphabet 0..3).
No labels, supplied lag, permutation, donor, gradients or optimizer.
Test the restricted hypothesis class y_t=P x_(t-k), k=0..16, P any
of 24 permutations. Predict token at t+1, not token at t.
Count C_k[a,b] over identical target positions 17..63 across streams.
Find best permutation by exact enumeration, maximize count sum. Accept
only a unique perfect candidate, full coverage (each input >=32 times),
and independent validation exact match. Otherwise abstain; distinguish
no fit from ambiguity. Deterministic recurrences can have alias lags:
e.g. identity with k=0 fits all lags; never claim uniqueness for those.
This is finite hypothesis discovery, not general language learning.
Pre-register 24 permutations x 8 identifiable lags 9..16 (192 cases);
each has independent train/validation/test streams (64/32/64 x64).
Generators are fixture-only; discovery receives no hidden parameters.
Evaluate all native logits on first and second 32-token chunks, valid
positions only, with held-out true next token (chunks need next token,
use windows starting 0 and 31). Require all 192 recovered correctly,
100% held-out predictions, 100% on independent arbitrary-context next
token probes, native greedy autoregressive continuation from initial
k+1 tokens over 32-token total length, strict safetensors reload exact.
Controls: iid streams, nonpermutation constant output, ambiguous
constant-symbol-per-stream recurrence, shuffled targets/stream tokens,
and train-perfect validation-violated streams; require abstention.
Save count matrices, candidate scores, reports and representative
checkpoints. No hidden truth used to choose weights; oracle only checks
after compile. No claim of novel law outside the restricted class.
