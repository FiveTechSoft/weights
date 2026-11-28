# Native contextual retrieval
Following cycle and composition passes, compile y_t = P x_(t-k) in the
same 6800-parameter KarpathyGPT, context32, vocab4, d16, 2 blocks, 2 heads.
No checkpoint donor, optimizer, learned probes, external runtime routing,
or changes to forward(). Fixed lags 1,2,4,8,16 and permutations identity,
cycle+1, swap(0,1). All parameters analytically set.
Use centered one-hot token codes (first4); zero-mean unit-norm circular
position code (coordinates4..7, period64). Q rotates position by -k,
K reads position; scaled dot product 4096 cos(2pi(t-k-j)/64). Native causal
mask, softmax approximates a selector on valid positions t>=k; prefix
t<k unspecified and excluded. V reads centered token, O writes to scratch
8..11. The post-attention MLP maps the scratch to token coordinates and
clears other channels using paired GELU identity with LayerNorm compensation
on ideal selected-state manifold. Block1 applies P with same identity.
No hard attention, no runtime symbol extraction. Audit actual attention
mass, intermediate residuals, native logits, save/reload parity.
Evaluate 256 random length32 sequences plus all 16 combinations of current
and retrieved token on varied positions (all valid). Counterfactual:
change retrieved token only; compare exact expected output. Change future
tokens only: earlier logits invariant within 1e-5. Controls zero first
attention projection, and k=0 (current-symbol prediction); measure on
random histories, require accuracy <=.30. For each of 15 models require
100% valid-token accuracy, selected attention >=.99999, intermediate
max residual error <=1e-4, reload exact. No claim of arbitrary-context
learning, content-addressed memory, language or exact finite softmax.
If successful, next derive k and P from examples without giving rule.
