# V8 — isolate downstream transport with oracle digit interventions
V7 failed before transport: scalar circuit accuracy 8.5% on shifted test;
categorical joint 77%. Do not interpret as proof native transport impossible.
V8 is diagnostic only, not an integrated skill or preservation claim.
Frozen Qwen2.5-0.5B, no backward, no permanent edits. At MLP output of blocks
index 1 and 2, add a fixed vector for each requested digit at final token.
Oracle labels supplied ONLY for diagnostic intervention, not a deployable model.
Compare three arms: base, oracle direction, cyclically wrong oracle direction.
Estimate directions without ZARP answer labels using finite differences of
digit logits centered across digits (not subtracting base top token, which may
be a digit and zero a row). Use same 32 discovery prompts for every digit.
Basis: ten weighted unembedding directions plus 54 seeded random directions,
orthonormalized. Central differences epsilon=.1 median residual norm.
Ridge .001 mean diagonal; solve inverse for desired centered digit contrast.
Dose grid [0,.125,.25,.5,1,2,4,8], amplitude median top-vs-digit median
margin +2. Select layer and dose on dev 200 by oracle unrestricted accuracy,
ties earlier layer then lower dose. No retention claims because oracle
intervention only at designated positions.
Separately measure categorical decodability of actual normalized MLP inputs
at blocks 1 and 2, with affine ridge from V5. Fit discovery 600, dev 200,
test 200 novel pairs in 100..199 (both operands), seed SEED+8.
Report all dev curves, singular values, selected layer/dose, sealed oracle,
wrong-oracle (same norm distribution not necessarily per-item), base accuracy,
digit-restricted oracle accuracy and categorical joint accuracy on same
shifted test. Gate oracle >=95%, >=20pp over base and wrong oracle;
report categorical >=95% separately. If oracle fails, do not implement another
native extension with same transport; revisit architecture. If oracle passes
but categorical fails, fix operand interface before integration. If both pass,
compile a categorical circuit into a native MLP (requires realizing conjunction
with SwiGLU) and repeat preservation gates on fresh data.
No retroactive change to V7, no skill success declaration from V8.
