# N14: one growing neural model
Reproduce N11/N12 discovery; 99 primitive and 64 composed projected laws
as N13, with task IDs assigned in that order. One causal Transformer
with fixed capacity for 163 laws, shared equality and attention circuits,
and task-conditioned output gates. No per-law model or external
symbolic inference. Reserve all task IDs from initialization. Add only
previously zero MLP rows and output columns, never modify prior weights.
Test stages 0, 1, 16, 64, 99, 163. For every installed law test all
valid tuples over reserved identities 8..15 and 128 seeded invalid
tuples (or all if fewer), plus uninstalled task probes expected reject.
Re-evaluate prior tasks after each insertion stage; compare logits on
fixed probes before/after, tolerance 1e-4, exact predictions required.
Audit unchanged weights except permitted slots and exact reload of
each stage. Require all stages pass, no donor/optimizer/backward.
Task IDs are external prompts, not autonomous task identification.
N12 projection still two outputs, not full four-output programs.
Separate synthetic architecture, no N8 integration or language claims.
Preallocated capacity not open-ended growth; report actual parameter
count and nonzeros. This is an integration milestone, not RSI.
