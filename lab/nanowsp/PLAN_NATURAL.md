# Natural-language construction N1
Factor gate 45554948091 passed 8/8, 19646 held-out context positions,
mean excess NLL .000343. This is the first natural-text construction
after that gate, not the first donor-free natural-text attempt.
Fixed Tiny Shakespeare hash and split from train_reference.py.
Same native architecture 4L4H256d FF1024 ctx256 vocab65, 3230208 params.
No donor weights, gradients, optimizer or external inference memory.
Fit a statistical teacher on first 80% of the permitted 90% training
region only: hierarchical interpolated ngram orders 1..5 (pseudo-count
strength 8, unigram .5). Use remaining training region to fit MLP
readout via a single ridge solve on 2048 fixed random 64-char windows.
No evaluation-region examples in any fit. Teacher is a statistical
predictor, not an LLM donor; report its NLL as a baseline.
Use centered onehot tied embeddings and circular positional codes.
First attention layer: heads retrieve lags 1,2,3 and uniform causal
prefix; project retrieved characters using SVD of centered conditional
log-probability matrix from teacher corpus. Head widths 64,48,32,32
respectively, current token width65. No position-dependent external
state. Use fixed large opposite anchors to approximate constant
normalization; actual normalized residuals are used for fitting.
First MLP contains 256 signed linear feature pairs, 256 frequent
adjacent-pair detectors, 256 seeded random GELU features; all features
fixed before fitting. Fit output residual first65 coordinates by
ridge on centered teacher log probabilities minus current embedding.
All remaining blocks identity (zero residual branches). Report active
weights and acknowledge this is a shallow statistical compilation
baseline in a 4-layer container, not full use of architecture.
Accumulate Gram and cross-products in batches, ridge .001 times mean
diagonal, solve double precision. Save native checkpoint, strict reload
and exact logit parity. Report validation and confirmation NLL, full
1..5-gram ladder, long context and generation, ICL via existing
llmness.evaluate. Compare teacher NLL on same positions. Also report
held-out native vs teacher logit error and zero-prefix-head ablation.
No tuning on confirmation or sealed tail. Only evaluate final 8000
reference tail if validation AND confirmation beat strongest 1..5-gram
by .05 and pass context and generation; final tail was previously
used by reference, so not globally pristine. Report scientific
failure explicitly even if CI succeeds. No claim of new language
capacity if just compressing statistical teacher. Next iteration must
use measured failure, not arbitrary additional synthetic tests.
