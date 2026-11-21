# What does post-training change? Plan fixed before analysis (2026-11-06)

Question: do fine-tunes that improve Qwen3.5-9B change its weights in a common way that WSPLang could apply without
training ("mejora sintética", provenance `synthetic_transfer`)? And is that way specific to improvements, or the mark
of any fine-tune?

Already seen before this plan was written: the Qwen3.5-9B -> MiMo statistics (results.md §38). Everything below about
other models is unseen.

## Models (all from `Qwen/Qwen3.5-9B` unless noted; text model only, 427 tensors, names via `text_model_name`)
| Role | Model | Training (from its card) |
|---|---|---|
| Improvement, reasoning distillation | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B | distillation |
| Improvement, agentic SFT | TokenRhythm/NeoHorse-1-9B | routing-guided SFT + on-policy distillation |
| Improvement, agentic | microsoft/Fara1.5-9B | computer-use agent |
| Improvement, RL only | allenai/tmax-9b | DPPO, 200 steps, terminal agent (+6 pts Terminal Bench) |
| Official post-training (other base) | Qwen/Qwen3.5-9B-Base -> Qwen/Qwen3.5-9B | Qwen's SFT + RL |
| Negative control (not aimed at reasoning) | Indexnusrefather/Nyx-RP-9B-Instruct-2608-v1 | role-play fine-tune |
| Positive control of the method | huihui-ai/Huihui-Qwen3.5-9B-abliterated | refusal direction removed |

## Measures (engine/weight_diff.py)
Per tensor: relative change ||D||/||W||, cosine(W, W'), changed fraction, spectrum of D (top-1 energy, energy in 16
directions, stable rank / smaller dimension). Per model: totals by tensor type and by layer.
Per pair of models, per tensor: cosine of the two deltas; overlap of their top-16 singular subspaces
(||V1^T V2||^2 / 16).

## Controls and nulls
1. **Method check:** the abliterated delta must be near rank one (top-1 energy > 0.9 in the tensors it touches) and
   must touch few tensor types. If not, the spectral measures are not trusted.
2. **Chance overlap:** k/d (16/4096 for the large matrices).
3. **Structured null:** overlap of the same tensor type at *different* layers (layer i of model A with layer j != i
   of model B). Overlap that comes from the general geometry of these matrices shows up here too.
4. **Explained by W:** overlap of each delta's subspace with the top-16 singular subspace of the base weight. If two
   deltas overlap only inside W's top subspace, the shared part is "grow what is already big", not a new direction.
5. **Generic fine-tune mark:** the role-play control. A pattern counts as improvement-specific only if the four
   improvements share it and the control does not.

## Decision rules
- **H1, where (profile):** Spearman correlation of the per-tensor-type relative-change profile between two models.
  "Shared where" if every improvement pair has rho >= 0.7. "Improvement-specific" if, in addition, each improvement's
  rho with the control is below the smallest improvement-improvement rho.
- **H2, shape:** a model's update is "low rank" if the median top-16 energy of its large matrices is >= 0.5,
  "diffuse" if <= 0.1.
- **H3, direction:** a pair shares directions in a tensor type if the median same-layer subspace overlap is >= 3x the
  structured null (control 3) and >= 3x chance. The cosine of deltas is reported with its sign.
- **H4, causal (later, only if H1 or H3 hold):** transplant into the base only the shared part (by tensor type, or the
  shared subspace) and measure GSM8K/MATH-500 with lab/probes/model_compare/eval.py, against the same norm of random
  update in the same tensors.

Numbers are reported with a second, independent computation for at least three tensors per model (numpy, full SVD).
Thresholds are not changed after the data are seen. Whatever the result, it is written up.
