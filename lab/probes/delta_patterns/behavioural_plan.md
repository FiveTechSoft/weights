# Behavioural test of a partial delta (plan, written 2026-11-08, before any behavioural number exists)

STATUS: PLAN ONLY. Nothing here has been run. It needs a decision on large streamed reads (see "What is not authorised").

## Question
F27-F31 show a rank-k part of MiMo's delta moves the base layers' outputs toward MiMo's (closer output, versus a same-size random part).
That is proximity. The question here is improvement: does base + part score better on a task metric where MiMo itself beats the base?

## What this machine can and cannot do (measured 2026-11-08, 2 CPUs, about 2 GB RAM, 19 GB free disk)
- Full-model generation: no. A 9B model is 19 GB in bf16; with the weights streamed layer by layer, every generated token would read 19 GB.
- **Teacher-forced scoring (log-likelihood of given text): yes.** One layer at a time is about 0.4 GB. Measured: HTTP range reads about 195 MB/s;
  a layer forward is 1.7 s for 512 tokens (linear attention) and 1.3 s (full attention), so about 0.1 s per token for 32 layers, about 10 tokens/s, plus
  weight handling of about 3-5 s per layer per variant. The hidden states between layers are kept per prompt (fp16 on disk, a few hundred MB).
  The head is 248320 x 4096 (untied); the log-sum-exp is computed over vocabulary chunks.
- Correction of an earlier statement (F29 said inference of a 9B "cannot run" here): that holds for generation; scoring by streaming is feasible and is the proposed route.

## Fair-evaluation requirements (fixed now)
- Tokenisation: one token id sequence per text, produced once with the base tokenizer and fed to every variant (base, MiMo, patches). MiMo's HF tokenizer ships an older
  pre-tokenizer regex (F25); on English, code and Spanish ids are identical, so the choice is not a confound there. Scripts with combining marks are excluded.
- Embeddings: each model uses ITS OWN embedding table and head for its own score; patched models use the base table and head (the patch touches layer matrices only). The gap
  recovery therefore compares base+patch with MiMo's native pipeline, and the embedding/head part of MiMo's change (6.5-6.9 %) is a stated ceiling on what a layers-only patch can recover.
  A second arm, "MiMo layers + base embedding/head", measures how much of MiMo's gap lives outside the layers.
- Template: no chat template is applied; plain text continuation only (no template ships in the HF folder, F25). A template-based arm needs the GGUF template and is out of scope.
- Revisions pinned: Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a, MiMo-V2.6-Distill-Qwen-9B 1f09ddcc1a0ea45751ae0dfe2af20d533168c46f.
- Numerics: float32 compute, layers as in `layer_probe.py` (transformers 5.20.2, torch fallback kernels). First, a parity check of the streamed pipeline against the library's
  full-model forward on a tiny random model (no download), and against the layer probe (bit-level on layers 0-3).

## Data (public, small, licences permit use)
- openai/openai_humaneval (MIT, 164 problems, 91 KB), revision 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544: score the reference solution given the prompt.
- openai/gsm8k (MIT), revision 740312add88f781978c0658806c59bc2815b9866: score the reference answer given the question.
- google-research-datasets/mbpp (CC-BY-4.0), revision 4bb6404fdc6cacfda99d4ac4205087b89d32030c: negative-control style data and a second code set.
- Split fixed before scoring: items sorted by their id; even positions = development (to choose nothing but to check the gate), odd positions = evaluation. About 8k scored tokens per split.

## Metric and gates (pre-registered)
- m = mean per-token negative log-likelihood of the reference text (lower is better), paired by item.
- Gate G1 (is there anything to recover?): on the development split, MiMo's m is lower than the base's with a paired bootstrap 95 % CI that excludes 0 on at least one dataset.
  If not, STOP: with no measurable MiMo gain on teacher-forced text, this test cannot separate improvement from proximity. Report that and do not search for another metric.
- Variants on the evaluation split, only for datasets that passed G1: base, MiMo, MiMo layers + base embedding/head, base + rank-16 part, base + rank-256 part,
  the same two with a random subspace of equal size (2 seeds), base + MLP-only delta.
- Primary outcome: gap recovery R = (m_base - m_variant) / (m_base - m_MiMo), per dataset, with paired bootstrap CI over items.
- Success for "the part improves, not just approaches": R of the rank-256 part > 0 with CI excluding 0, AND exceeds the random-subspace control's R by a margin whose CI excludes 0.
  Anything else is reported as not shown. Rank 16 is secondary.
- Negative control: a dataset where MiMo does NOT beat the base on development must not show a patch gain larger than the random control; if it does, the patch is a generic effect and
  the success reading is void.
- Not claimed even on success: better generation, agentic or benchmark scores, or anything beyond teacher-forced likelihood on these texts.

## Cost, only if authorised
- Delta extraction pass: stream both models once (about 38 GB read, about 3.5 min at 195 MB/s), eigh of 8 matrices per layer (about 5-10 s each), keep rank-256 factors
  (about 100 MB per layer, 3.2 GB disk, not committed).
- Scoring: about 8k tokens x 2 splits per variant at about 10 tokens/s, so about 13 min per split per variant plus streaming; about 10 variants: roughly 3-4 hours of CPU, run in the background, restartable per layer.
- Money: none. Machine: the current one, no GPU, no shared servers.

## What is not authorised (decision pending)
Reading about 40-100 GB of public weights by HTTP range (streamed, not stored whole) and about 3-4 CPU hours. The standing limit was "do not download large models"; this is the same
kind of act, so it is not started. Preparation that needs no large read (harness parity on a tiny random model, the pack format below) can proceed.

## Selective delta package (reproducible, no third-party weights distributed)
- The repository holds a manifest and the code, never the factors or any model tensor. Manifest fields: base repo + revision + sha256 of each source tensor's bytes; model repo + revision + same;
  licences read from the model cards today (base Apache-2.0, MiMo MIT; a factor file is a derivative of both, so its redistribution follows both licences and is left to the owner);
  tensor list; rank; method and library versions; sha256 of every output file; the parameter alpha.
- Apply = add alpha * (left @ right) to each listed base tensor, only after the base tensor's hash matches the manifest. Rollback = subtract the same factors, then verify the hash returns to the manifest's base hash (exact in float32 storage; in bf16 storage the patch is stored as a separate additive file and the base is never overwritten).
- Claim carried by the package: "extraction reproducible"; no benefit is claimed until the gates above pass.
