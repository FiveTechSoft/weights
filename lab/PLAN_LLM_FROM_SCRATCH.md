# Plan: entrenar un LLM sintetico desde cero

Honest scope: NON-pretrained toy-scale prototype; NOT a production LLM.

## Objectives
1. A small char-level transformer trained from scratch (no donor weights).
2. Held-out perplexity measurement — honest, unblinking.
3. Report: what is verified vs what is NOT.

## Protocol
- Corpus: Shakespearechar (~1.1M chars, real text: public domain Shakespeare).
  NO synthetic data. NO splits of test set. Holdout = last 10% of chars.
- Model: NanoWSP-3 family (char-level; vocab 256; 6 layers, d=256, 8 heads,
  ff=512; tied embeddings). ~1.5M parameters.
- Training: from scratch (seeded RNG), next-token prediction, AdamW
  (lr 8e-4, betas 0.9/0.95, wd 0.01), warmup 100 steps, grad clip 1.0.
- Steps: 1500-2000, seq=128, batch=16 — modest, honest, feasible on CPU.
- Holdout eval: PPL every 50 steps; snapshots every 300 steps.
- NO skills injected. NO native surgery. Pure language-model-from-scratch.
- Honest deliverable: report.json + sample outputs, zero overclaims.

## Go/No-Go gates
- Loss decreasing over time (measured on the training stream).
- Held-out PPL improving by the end vs initialization.
- Held-out PPL plausibly < 10. (A strong, rough bar for char-level prototype.)
- Sample text visibly coherent enough to identify as next-token text.

## Explicit non-goals
- Not a production LLM. Not chatbot-grade. Not a replacement for pretrain.
- Not a multibillion-token model. Not trained on web text at scale.
