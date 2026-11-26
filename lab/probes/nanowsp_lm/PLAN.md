# NanoWSP-LM — causal character language model, not associative memory

All architecture and reference-training steps are declared in WSPLang. Uses the existing
architecture compiler and train op, not a new hidden Python model. No claim of donor-free
language acquisition. No run or measured result at submission.

## Architecture
4 layers, 4 heads, width 256, FFN 1024, context 256, 65-character vocabulary,
learned position embeddings, causal attention, pre-layernorm, GELU, tied output.
3,230,208 unique parameters, bias-free (`bias = false` in the `.wsparch`, so the W0 written by
`build_nanowsp_lm` is exactly the initialisation `train_nanowsp_lm` starts from: same seed, same tensors).
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
Do not tune on held-out text: `wsp_train` evaluates on the last 10 % of the corpus, which contains
the 8 000 characters `verify_llmness` scores, so the run keeps the **last** checkpoint
(`always_save_checkpoint=true`) and no checkpoint is selected on that text.
**Margin (fixed):** held-out NLL at least 0.05 nats/char below the best n-gram of the ladder (orders 1-5,
fit on the training part only). A passing result establishes a small domain-limited language
model, not general instruction following or a large language model in scale.

## Prediction (written 2026-11-24 before the first training step finished)
Reference point: the repo's 6-layer `nanogpt_shakespeare_char` scores held-out NLL 1.826 against a best
5-gram of 2.082 on the same 8 000 characters (`lab/runs/llmness_nanogpt_shakespeare_char.json`).
- `wsp_train` val loss at iteration 5 000: 1.50-1.70 (80 %).
- `verify_llmness` held-out NLL: 1.85-2.00 (70 %); below best n-gram by >= 0.05: 85 %.
- L2 long-context gain passes: 85 %; GEN passes: 80 %; L3 in-context binding fails (as for the 6-layer model): 85 %.

## WSP research follow-up (not implemented)
Use fixed-seed intermediate checkpoints of the reference run to study delta singular
subspaces and attempt closed-form weight synthesis. Compare against W0, shuffled
same-norm delta, n-gram and trained reference. Transplanting a learned delta is donor-based,
not synthesis. A no-gradient variant must independently pass the same language gate.

## Result (2026-11-24, commit a942ee7, RTX 3060 shared with another process, CUDA venv)
Corpus sha256 a29d649d…; checkpoint `models/nanowsp_lm/model.safetensors` sha256 0b78ed84… (iteration 5 000,
last checkpoint). `wsp_train` val loss 1.4368 at 5 000 (best seen 1.4214 at 4 750, not selected).

| | trained | W0 (same seed, untrained) |
|---|---|---|
| L1 held-out NLL (8 000 chars) | **1.8287** (PASSES) | 4.2264 (FAILS) |
| best n-gram (5-gram, train part) | 2.0816 → margin **0.253** ≥ 0.05 | — |
| L2 long-context gain (64 vs 8 chars) | 0.108 nats, CI95 [0.071, 0.146] (PASSES) | FAILS |
| GEN (free sample) | known-word rate 0.963 (PASSES) | FAILS |
| L3 in-context binding | 0.140 vs 0.135, p = 1.0 (FAILS) | FAILS |
| verdict | LANGUAGE_MODEL | not a language model |

Second method for L1: direct sliding-window NLL over the same last 8 000 characters with the full 256-character
context, own loop: **1.8306** (evaluator 1.8287; agree within 0.002). The repo's 6-layer, 384-wide nanoGPT
scores 1.826 on the same text: this 3.23 M-parameter model matches it.

Prediction check: val loss band 1.50-1.70 **missed** (1.437, better); held-out NLL band 1.85-2.00 **missed**
(1.829, better); margin ≥ 0.05 hit; L2 pass hit; GEN pass hit; L3 fail hit.
