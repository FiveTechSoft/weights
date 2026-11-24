# VO geometry: distinguish amplitude from learned orientation
The head trace (run 45469560748) found no joint-pass subset. Full VO
increases paired intact NLL on early background but is necessary on late
background. Attention mass at lag >=8 changes little, while residual
contribution norm grows 3.4--6.7x. These observations do not identify
the represented content. Before attempting synthesis, test whether scaling
early maps is sufficient or learned map geometry matters.
On both frozen backgrounds (100 and 500), replace all four VO maps by:
early, late, early scaled per head to late Frobenius norm, late scaled
to early norm, late with one shared fixed output-coordinate permutation.
The permutation preserves singular values and norms; it disrupts alignment
with residual channels, not a pure orientation-only control for the whole
network. No train or parameter search. Same paired windows and validation/
confirmation slices as trace_vo. Report full NLL, intact and corrupted NLL,
context gain, confidence interval and paired intact improvement vs native
background. Compare equal-norm maps on both backgrounds. Magnitude alone
is supported only if scaled early recovers >=80% gain AND NLL <= early
native on both slices in early background. Report late-background
differences separately (coadaptation). No synthesis claim; if no joint
pass, stop rather than inventing a new synthesis method.
Record per-head effective-map norms, cosine and relative delta, singular
energy ranks. This is diagnostic reuse of donor weights, not a donor-free
compiler. Preserve original reference ICL failure.
