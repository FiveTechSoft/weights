# VO content: test dominant directions before assigning meaning
Prior run 45473334839 rejects simple amplification. Concentrated singular
energy is not evidence of linguistic meaning or sufficient functionality.
Use exact original 100/500 checkpoints. No gradient training. On frozen
500 background, factor each effective VO map B=O V using double SVD.
Keep top k=2,4,12,64 or remove top 4 (keep components 4..63).
Factor into native 64-wide V and O, padding zero. Check rank64 logits
parity against original, and verify factor product. No renormalization:
these interventions test necessity/sufficiency of directions within 500,
not a norm-matched causal claim. Include top4 with shared output permutation
as equal-spectrum alignment control. Original 100 and 500 are references.
Measure NLL and paired losses on validation and confirmation slices as before.
Additionally shuffle first 56 tokens within each example, preserving their
histogram and last 8 positions; compare intact versus shuffled versus random
replacement. Shuffling changes local order and position associations; cannot
identify syntax or semantics by itself. Bootstrap paired differences using
same samples, descriptive intervals (overlapping windows, not independent
documents). Test if retained directions preserve >=80% of original 100->500
NLL improvement and context gain, lower CI >.01, on both slices. Report
smallest passing rank and complement; if none, no low-rank sufficiency claim.
Inspect top4 input directions: project actual normalized layer0 embeddings
for each vocabulary character at positions 0..63, report mean per-character
scores, positional variation, and character class contrasts (letters,
space, punctuation). Also quantify how well a bag-of-characters linear
predictor explains head contributions in top4 output coordinates: fit on
2048 train windows, evaluate fixed 512 windows per reserved slice, ridge
.001 times mean Gram diagonal, report out-of-sample R2 against training
mean and suffix-only baseline, using counts of first56 characters and last8.
Fit is diagnostic only, not a compiler or evidence of causal meaning.
No new model or synthesis until these measurements identify a candidate.
