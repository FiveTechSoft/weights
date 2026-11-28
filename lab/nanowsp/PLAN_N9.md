# N9: reusable relation identities and genuinely distant information
N8 passes fidelity/prediction/selection/growth; distant gain zero.
Instead of extending contiguous ngrams, give each existing conditional
relation a discrete reusable identity (65 bigram + 501 pair + 512
triple rows = 1078 IDs). A context resolves to its deepest selected
parent; different local contexts can share a parent. No learned
semantic categories are claimed. Encode parent ID by four centered
base-6 onehot factors (24 dims, norm² 10/3, zero mean, match margin 1).
Use N8 count/discovery/acceptance partitions and reproduce its exact
selection. No development or reserved-tail data selects candidates.
Candidate feature: distant character class at lag 8,16,32,48.
Fixed classes: newline, other whitespace, uppercase, lowercase vowel,
other lowercase, digit, punctuation, other. For each lag and each
(parent ID, distant class), count next symbols on count source,
smooth with 8 times parent's distribution, compare realized log
ratio to parent on discovery and acceptance splits. Exclude incomplete
contexts at boundaries. Eligible >=8 discovery observations, positive
sum gain. Rank by discovery gain then key ascending. Choose single
lag by largest sum of top512 eligible discovery gains (ties smaller
lag); no acceptance or development results choose lag. Accept in
that ranking only if >=8 acceptance observations and positive gain,
up to512. If none, abstain. Same budget frequency control ranks
count-source occurrences at chosen lag. Log all candidate statistics.
Compile in native 3230208-param nanoGPT: third head retrieves the
centered class code at chosen lag, extra availability marker pair.
Block0 produces N7 logits/anchors and compact local identities,
parent ID code and class. Pair correction also changes parent ID.
Block1 produces N8 logits/anchors and replaces parent ID when triple
matches, removing local identities. Block2 applies selected
(parent,class) correction plus anchor difference and clears scratch.
A common C for all distributions. Preserve fixed norms and zero
means; no gradient, regression, optimizer or donor. Reproduce N8
with no distant corrections. Save/reload parent, frequency, utility.
Fidelity gate: parent NLL within1e-4 of N8 on both reused dev slices;
all native-reference NLL differences <=1e-4, all audit errors
(probability, early/retention, mean/norm, causal, irrelevant prefix)
<=1e-4; exact reload. Audit every selected (parent,class) in a valid
representative local context, plus up to4096 unselected combinations,
two positions lag and 63 (lag <=48), first two and t<lag backoff.
Parent ID code checks every row and unique factor codes.
Predictive gate: utility NLL gain >=.01 vs N8 on both dev slices;
secondary utility vs matched frequency >=.005 both.
Context test: fixed 2000 target positions per slice with 64-token
history, compare native true history vs shuffled first56 tokens
(last8 intact) and last8-only. Seeded within-row permutation preserves
prefix histogram; distant character is affected, local suffix is not.
Report per-position paired mean NLL gain and 95% block bootstrap CI
(group targets by contiguous 128-character target-position blocks,
1000 resamples). Context gate requires lower CI >.01 for both
contrasts on both slices. N8 baseline should be invariant to prefix
and short-context contrasts; report its negative control.
Language gate as previous experiments; reserved tail only if pass.
No claim of autonomous feature generation, semantic abstraction,
RSI, or generalization to fresh corpora. Candidate classes and lags
are human specified. Failure is evidence; do not tune on dev.
