# NanoWSP language core
Single objective: construct a genuine small language model without gradient
training or donor weights in the final synthesis. This phase is a gradient-trained
reference only; no claim of synthesis. Keep previous failures intact.
Reuse engine/nanogpt_runtime.py (KarpathyGPT), optimizer and char encoding
from engine/wsp_train.py, and lab/llmness.py. No changes to Qwen or coprocessors.
## Reference
Tiny Shakespeare exact SHA256
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.
90% contiguous train, final 8000 chars sealed, preceding 8000 validation.
4 blocks, 4 heads, 256 d, FF1024, block256, bias false, dropout 0, 65 chars.
Seed 1337, AdamW lr 6e-4, betas .9/.95, weight decay .1 (matrices only),
gradient clip 1, batch 8 x256, 5000 updates, warmup 100, cosine to 6e-5.
Fixed snapshots after 0,10,100,500,1000,2000,5000 updates. Save CPU
safetensors with config, vocabulary, source corpus hash, gradient provenance
and exact update count. Independent training RNG and deterministic eval windows.
CPU CI runs full protocol; CUDA optional on user hardware (FP32, no AMP).
## Evaluation
At each saved checkpoint reload strictly and verify tied weights, exact parity
on fixed inputs. Score 128-token windows on validation only for trajectory.
Final sealed evaluation via existing llmness.evaluate: NLL <= strongest
1..5-gram minus .05, L1 passes, long-context bootstrap lower >.01, generation
heuristics pass. ICL >=10pp delta and p<=.01 separately required for full
target; if reference fails ICL, report explicitly, do not lower criterion.
These are operational tests, not proof of human-like comprehension.
## Causal analysis
Only after reference language gate passes, on validation slice (not sealed):
replace one learned component at a time by its same-seed initial value,
and separately zero its residual output (attention or MLP), restoring after
each intervention. For each block, Q/K jointly, V/O jointly, MLP and attention
output; tied embeddings and position embeddings separately. Evaluate NLL and
paired context gain on identical windows; report effects and rank by lost
context gain. Initial replacement can break coadaptation; ablation establishes
necessity in this model, not sufficiency or a universally transferable rule.
No synthesis until this evidence exists. Persist report even on failed gates.
No extra experiments or post-hoc changes to thresholds.
