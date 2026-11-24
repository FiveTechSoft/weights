# Composition of algebraic operators
Cycle run 45501914995 verified +1/-1 compilation and token relabeling.
Next test only composition, not new training. Existing 6800 parameter
NanoGPT, no donor, no gradients. P=(1,2,3,0), Q=(1,0,2,3) in image
notation. P and Q do not commute. Compile P then Q and Q then P in
separate native MLP blocks, and direct composite in a single block as
control. All other weights identical, attention disabled. Use same
one-hot manifold and exact LayerNorm compensation as cycle_law.
Generalize to every one of 24 permutations, every ordered pair (576),
including inverses, identity, noncommuting cases. Test every symbol at
length1 and arbitrary histories, compare intermediate residuals to
expected one-hot and final logits to direct composition with max error
<=1e-4. Verify identity and inverse, order matters on all differing
symbols for noncommuting pairs. Save example checkpoints and report.
Reload saved examples strictly and verify identical logits. No claim
of discovered learning law: this is a constructive algebraic primitive.
Prior trained cycle internal structure still uninspected, not a blocker
to verifying this identity. After success next capability is retrieval,
not larger permutations or additional MLP benchmarks.
