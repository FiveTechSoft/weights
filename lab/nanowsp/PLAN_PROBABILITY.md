# Probabilistic corpus discovery
Previous discovery CI 45548611658 passed 192/192 deterministic cases.
Use same 6800-parameter NanoGPT and analytic positional retrieval.
Input raw streams only; estimate all lags 0..16 on common target positions
17..63 with Dirichlet half-count smoothing. Choose lag by lowest train
NLL; require independent validation NLL advantage >=.03 nat/token over
unigram and >=.02 over every competing lag, else abstain. No donor or
gradient updates. Test 24 hidden positive stochastic transition matrices
(3 per lag 9..16), independent train/validation/test streams 256/128/128
length64. Generator uses a random permutation with dominant mass .7-.85,
remaining positive random masses, never passed to discovery/compiler.
Controls iid uniform and iid skewed, require abstention.
Compile estimated T into logits: centered log probabilities for each
symbol, plus two opposite scratch coordinates to equalize squared norms;
set final LN gain to cancel that common normalization. Reuse first
block retrieval unchanged. Second MLP uses four GELU features on
centered one-hot inputs, invert their 4x4 response matrix to solve for
output minus residual. No optimization. Native tied output reads logits.
Gate: all 24 correct lag, maximum absolute probability error <=1e-5
versus estimated T for arbitrary histories and held-out streams,
native test NLL differs from estimated oracle NLL <=1e-5, oracle excess
NLL vs true distribution <=.02 averaged across cases, each improves on
unigram on test >=.03. Report true-vs-estimated KL and test NLL, not
100% next-token accuracy. Native autoregressive sampling on 128 streams
per case; report transition counts and average conditional total
variation, gate <=.06 average across cases. Reload strict exact.
Save all models and reports. No general language claims: finite family
of single-lag Markov conditionals; next step multi-lag context only
if this gate passes.
