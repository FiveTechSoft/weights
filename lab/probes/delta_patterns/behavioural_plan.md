# Behavioural test of a partial delta (plan, written 2026-11-08, before any behavioural number exists)

STATUS: PLAN ONLY. Nothing here has been run. It needs a decision on large streamed reads (see "What is not authorised").

## Question
F27-F31 show a rank-k part of MiMo's delta moves the base layers' outputs toward MiMo's (closer output, versus a same-size random part).
That is proximity. The question here is improvement: does base + part score better on a task metric where MiMo itself beats the base?

## What this machine can and cannot do (measured 2026-11-08, 2 CPUs, about 2 GB RAM, 19 GB free disk)
- Full-model generation: no. A 9B model is 19 GB in bf16; with the weights streamed layer by layer, every generated token would read 19 GB.
- **Teacher-forced scoring (log-likelihood of given text): feasible by estimate.** One layer at a time is about 0.4 GB. Measured: HTTP range reads about 195 MB/s (one 101 MB tensor);
  a layer forward is 1.7 s for 512 tokens (linear attention) and 1.3 s (full attention) on layers 2 and 3. Extrapolated, NOT measured on the full model: about 0.1 s per token for 32 layers, about 10 tokens/s, plus
  weight handling of about 3-5 s per layer per variant. The hidden states between layers are kept per prompt (fp16 on disk, a few hundred MB).
  The head is 248320 x 4096 (untied); the log-sum-exp is computed over vocabulary chunks.
- Correction of an earlier statement (F29 said inference of a 9B "cannot run" here): that holds for generation; scoring by streaming is feasible and is the proposed route.

## Fair-evaluation requirements (fixed now)
- Tokenisation: one token id sequence per text, produced once with the base tokenizer and fed to every variant (base, MiMo, patches). MiMo's HF tokenizer ships an older
  pre-tokenizer regex (F25, a sample-based finding that differs only on scripts with combining marks). Measured on the actual scored texts (pinned tokenizer.json of both
  repos, 2026-11-08): HumanEval 164 items / 32,830 tokens, GSM8K test 1,319 items / 249,771 tokens, MBPP full test 500 items / 39,088 tokens: 0 items with different ids. The scoring
  code must repeat this check on every scored item and drop items that differ; equality is established for these three texts only, not in general.
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
- Gate G1 (is there a measurable difference to recover under THIS protocol?): on the development split, MiMo's m is lower than the base's with a paired bootstrap 95 % CI that excludes 0 on at least one dataset.
  If not, STOP this protocol. That limits only this metric and this text: it is not evidence that MiMo has nothing to recover, and no other metric is searched afterwards to rescue it.
  What the gate buys: it costs about a quarter of the full run and decides whether the other variants can say anything about improvement at all.
- Variants on the evaluation split, only for datasets that passed G1: base, MiMo, MiMo layers + base embedding/head, base + rank-16 part, base + rank-256 part,
  the same two with a random subspace of equal size (2 seeds), base + MLP-only delta.
- Primary outcome: gap recovery R = (m_base - m_variant) / (m_base - m_MiMo), per dataset, with paired bootstrap CI over items.
- Success for "the part improves, not just approaches": R of the rank-256 part > 0 with CI excluding 0, AND exceeds the random-subspace control's R by a margin whose CI excludes 0.
  Anything else is reported as not shown. Rank 16 is secondary.
- Negative control: a dataset where MiMo does NOT beat the base on development must not show a patch gain larger than the random control; if it does, the patch is a generic effect and
  the success reading is void.
- NLL of the reference text is a proxy for the probability the model gives that text. It is not HumanEval pass@k or GSM8K accuracy, not agentic skill, not general capability. Not claimed even on success.

## Staged budget (all numbers: reads are bytes of text tensors, 8.954 B parameters per model, about 17.9 GB in bf16 per model; times are estimates from the extrapolation above)
Common to every stage: weights are streamed layer by layer and never stored whole (disk use under 5 GB); no GPU, no money, this machine only; restartable per layer.
- Stage A, no large read: parity of the streaming pipeline on a tiny random model against the library's full forward, and against `layer_probe.py` on layers 0-3 (cached). 0 GB.
- Stage B, the gate: score base and MiMo on the development split only. Reads about 36 GB (2 x 17.9). Tokens: at most 4k scored tokens per dataset (first items in id order within the split), 3 datasets, 12k per model:
  about 20 min per model compute (estimate), about 40 min in total. Hard stops: throughput under 50 MB/s, any layer taking more than 3x the estimate, or disk use over 5 GB. If G1 fails on all datasets: stop, report, nothing else is read.
- Stage C, only if G1 passes and only for the datasets that passed: one pass over both models (about 36 GB read) in which, layer by layer, the delta factors are computed on the fly (eigh of 8 matrices, about 5-10 s each) and every variant's hidden states advance together
  (about 33 MB of fp16 hidden states per variant for 4k tokens). 12k evaluation tokens x about 9 variants is about 100k token-forwards, about 3 h (estimate). Factors are not kept.
- Maximum total if everything runs: about 72 GB read and about 4 h of CPU. Statistical power is limited: 4k tokens per dataset may not detect a small MiMo-vs-base difference; G1 would then fail for that reason and the report would say so.

## What is not authorised (decision pending)
Stage B (about 36 GB streamed read, about 40 min CPU) is the minimum decision; stage C (about 36 GB more, about 3 h) is a separate decision after G1. The standing limit was "do not download large models"; this is the same
kind of act, so it is not started. Preparation that needs no large read (harness parity on a tiny random model, the pack format below) can proceed.

## Selective delta package (reproducible, no third-party weights distributed)
- The repository holds a manifest and the code, never the factors or any model tensor. Manifest fields: base repo + revision + sha256 of each source tensor's bytes; model repo + revision + same;
  licences read from the model cards today (base Apache-2.0, MiMo MIT; datasets MIT, MIT, CC-BY-4.0 from each README at its pinned revision; a factor file is a derivative of both, so its redistribution follows both licences and is left to the owner);
  tensor list; rank; method and library versions; sha256 of every output file; the parameter alpha.
- Apply = add alpha * (left @ right) to each listed base tensor, only after the base tensor's hash matches the manifest. Rollback is by source: apply never modifies its input, so the original stays; to go back, discard the patched copy or re-read the original tensor from its source file and check its hash. Subtracting the factors is only a diagnostic (floating point does not reproduce the original bytes). Corruption is detected by hash comparison, not prevented.
- Claim carried by the package: "extraction reproducible"; no benefit is claimed until the gates above pass.
