# N10: associative retrieval before weight compilation
N9 fidelity passed, predictive and context gates failed. Keep N8.
Test whether a prior occurrence of the SAME parent relation predicts
the next symbol. N8 parent ID is a deterministic function of last
1-3 characters and selected relation hierarchy, not a semantic concept.
Use exact N8 count/discovery/acceptance splits and reproduce parent.
For current position t, query parent ID at t. Retrieve all earlier
positions s in [t-63,t-8] with identical parent ID, and observe x[s+1].
Require s>=2 so parent IDs have full local history. No future leakage:
s+1<=t-7. Use empirical continuation counts h. Candidate predictor
q=(alpha*p+h)/(alpha+sum h), p=N8 local distribution. If no matches,
q=p. Alpha candidates 1,2,4,8,16,32; choose maximum summed realized
log-ratio on discovery positions only (tie larger alpha).
Acceptance gate: positive mean gain >=.005 nats over N8 on independent
acceptance split, using 64-token histories, and positive gain among
matched positions. If failed, abstain from native compilation.
Report all alpha scores and match coverage, multiplicity, distances.
On both reused dev slices evaluate the chosen alpha without changes:
all sliding 64-character windows, N8 vs q; also 2000 seeded unique
positions matching N9 context test, true prefix vs last8 only (N8),
prefix shuffled preserving last8 and prefix histogram. Recompute
parent IDs on shuffled sequence. Matched position control: retrieve
same number of continuation symbols at uniformly sampled eligible
positions without matching identity, fixed seed; report gain vs this
control. Additional nearest-only and identity-exact triple retrieval
diagnostics at same alpha, not alternative winners. Bootstrap 1000
resamples of contiguous 128-target blocks; report paired mean and
95% intervals, coverage and conditional gains. Primary feasibility
gate: acceptance passes, gain >=.01 on BOTH full sliding dev slices,
and lower 95% CI >.01 vs N8 and shuffled prefix on BOTH sampled slices.
Report random control comparison (lower CI >0 both required).
No reserved tail, no weights constructed in this feasibility stage.
If gate passes, next experiment compiles causal key/query equality
and continuation retrieval into native attention with fidelity audit.
If fails, do not tune alpha or choose diagnostics on dev. This is
a fixed associative rule, not autonomous hypothesis generation,
semantic understanding or RSI. Distinguish from training-free native
model: N10 is an external statistical reference only.
