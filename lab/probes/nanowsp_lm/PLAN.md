# NanoWSP-LM — causal character language model, not associative memory

All architecture and reference-training steps are declared in WSPLang. Uses the existing
architecture compiler and train op, not a new hidden Python model. No claim of donor-free
language acquisition. No run or measured result at submission.

## Architecture
4 layers, 4 heads, width 256, FFN 1024, context 256, 65-character vocabulary,
learned position embeddings, causal attention, pre-layernorm, GELU, tied output.
Approximately 3.24M unique parameters (bias-free training variant).
Shakespeare char uses the existing repo corpus and vocabulary convention.

## Reproduce from repository root
```sh
python -m engine.wsplang programs/build_nanowsp_lm.wspl --run
python -m engine.wsplang programs/train_nanowsp_lm.wspl --run
python lab/verify_llmness.py --nanogpt-ckpt models/nanowsp_lm/model.safetensors --n-head 4 --corpus lab/data/shakespeare_char/input.txt --holdout-frac 0.1 --eval-chars 8000 --out lab/runs/nanowsp_lm_llmness.json
```
Create lab/runs if absent. CUDA reference configuration intended for a 12 GB GPU.
Record code commit, corpus SHA256, checkpoint SHA256, device and training log.
Do not report success merely from architecture compilation.

## Acceptance
Use existing llmness evaluator, report its individual gates and absolute metrics.
Compare trained checkpoint with random initialization and train-only n-gram baseline on
identical held-out text. Require lower held-out NLL than strongest n-gram baseline,
and report context ablation and free generation; no cherry-picked prompts.
Do not tune on held-out text. A passing result establishes a small domain-limited language
model, not general instruction following or a large language model in scale.

## WSP research follow-up (not implemented)
Use fixed-seed intermediate checkpoints of the reference run to study delta singular
subspaces and attempt closed-form weight synthesis. Compare against W0, shuffled
same-norm delta, n-gram and trained reference. Transplanting a learned delta is donor-based,
not synthesis. A no-gradient variant must independently pass the same language gate.
