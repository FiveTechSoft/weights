# N7: grow a fixed family of relations by measured utility
Reuse N6b compile_model and exact count-source bigram/trigram. No new
runtime operators. Count source first 80% of training 90%, as before.
Split remaining 20% training region into equal contiguous discovery
and acceptance portions. Exclude boundary transitions. Neither
development slice nor reserved tail selects candidates or stopping.
For each of 4225 candidate pairs compute sum of realized log probability
ratio of trigram to bigram over discovery triples. Require at least 8
discovery occurrences, positive total gain, then sort descending gain,
ties ascending pair id. Acceptance: at least 8 audit occurrences and
positive realized gain for each candidate. Take first 512 accepted
pairs (or fewer if exhausted); record all examined and rejected pairs,
their two scores and counts. Fixed nested budgets 64,128,256,512;
no budget chosen post hoc. Frequency control uses count-source most
frequent pairs at identical actual budgets, including reduced budgets.
Because pair contexts are disjoint, gains add exactly for a fixed
table; no claim that repeated scoring on acceptance data is unbiased.
Use common C across all bigram and full trigram rows; encode outputs
with zero mean and fixed norm as N6b. Build native checkpoints at all
budgets for both arms plus bigram. Native exhaustive audit at final
budget for both arms; score all stages on validation/confirmation and
report native vs exact table NLL; exact reload each checkpoint.
Primary gate: final utility arm improves NLL >= .01 vs frequency
at same budget on BOTH development slices, and >= .05 vs bigram.
Fidelity gate: final native audit max probability, norm, prefix and
causal errors <=1e-4, retrieval mass >=.99999, no wrong gates, exact
reload, NLL parity <=1e-4 all stages. Growth gate: all utility increments
nonnegative on BOTH development slices (tolerance 1e-4); report any
negative increments, never remove after seeing evaluation. Separate
language gates; tail only if both pass. No optimization/backprop/donor.
This is automated selection in a human-specified finite family, not
self-generated hypotheses, RSI or recursive abstraction.
