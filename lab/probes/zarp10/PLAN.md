# ZARP10: algebraic skill edit in a pretrained language model

Separate experiment from E2 ZARP. No claim of direct algorithm-to-pretrained-weight compilation.
Target: Qwen/Qwen2.5-0.5B (base), pinned resolved Hugging Face revision in report.
Rule: (3*a+5*b+7)%10. Input: "ZARP10(a,b) =" (completion, no rule).
Only one answer digit is evaluated; strict top-1 over entire vocabulary, not restricted candidates.
This tests next-token prediction, not full natural-language conversation.

Compiler: final MLP down_proj, feature inputs at final prompt token.
Analytic residual regression onto output-token directions, zero-change constraints from
ordinary arithmetic prompts, ridge in sample space, no backpropagation or donor.
A fixed rank cap 64 via SVD, dev-only choice of dose in [0.25,0.5,1,2,4].
Placebo uses same singular values and randomized left/right singular vectors.
Candidate and placebo saved and independently reloaded as ordinary HF checkpoints.

Fixed seed 20261124. 400 discovery, 100 dev, 200 sealed disjoint pairs sampled
without replacement from [0,49]^2. Test split constructed only after compile and dose selection
by an independent RNG reconstruction of the split. No sealed labels used in compiler.
Separate 200 arithmetic prompts (single digit modulo-free sums/differences),
and natural-text next-token NLL measured on a held-out prose sample.
Report actual rank, realized delta norms, changed tensor, base model revision,
raw token predictions and exact accuracy. Base rule-in-prompt is a diagnostic ceiling.
Gates: sealed candidate >=95%, improvement >=20 pp vs base AND placebo,
arithmetic baseline >=50%, arithmetic degradation <=2 pp, prose NLL increase <=0.1 nats.
Do not adjust these gates after results. This is an exploratory single-seed experiment.
Full generation, GGUF and wider retention tests required before general skill claims.
