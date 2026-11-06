# Findings log: what post-training changes in Qwen3.5-9B (running notebook)

Plan: `plan.md` (fixed before the other models were seen). Data: `lab/runs/delta_patterns/`. Summary in results.md §38.
Each entry: what was measured, the number, where it comes from, how sure it is, and what it teaches for WSPLang.
Status labels: **verified** (second method agrees), **measured** (one method), **open** (needs a control still running).

## F1. MiMo is a full fine-tune, not an adapter. (verified)
- 97 % of the 8.95 B text weights change; ||ΔW||/||W|| = 9.2 % over the model. `stats_mimo.json`.
- Check: relative change and stable rank of 3 tensors recomputed with numpy float64 and a full SVD: identical
  (`check_mimo.json`).
- WSPLang: an improvement of this kind is not a small edit. Budgets such as `max_modified_weights = 1 %` cannot
  express it; a "transfer" op has to work on whole tensors.

## F2. Where MiMo changes: gates and MLP most, the recurrent dynamics not at all. (verified)
- Median relative change per tensor type: DeltaNet β (`in_proj_b`) 12.7 %, `mlp.gate_proj` 12.5 %, `up_proj` 11.3 %,
  `down_proj` 10.8 %; q/k and DeltaNet inputs 8-10 %; embeddings / `lm_head` 6.5-6.9 %; full-attention `v_proj`
  4.9 % and `o_proj` 5.6 %; conv1d 1.4 %; norms < 1.3 %.
- `A_log` and `dt_bias` (DeltaNet decay dynamics) are not trained: the base stores them in F32, MiMo in BF16, and the
  only difference is that cast (max |Δ| = 0.0078, one bf16 step). `dt_bias` differs in no layer after rounding.
- By depth: bell shape, 8.5 % at layer 0, ~11 % at layers 17-23, 7.5 % at layer 31.
- WSPLang: a map of where post-training acts. Protect-by-default lists could include `A_log`, `dt_bias` and norms
  (post-training leaves them alone); edit budgets could be per tensor type instead of global.

## F3. MiMo's change is diffuse. (verified)
- Large matrices: stable rank 200-500 (6-12 % of the smaller dimension); the 16 largest directions hold 2-4 % of the
  change's energy. Exception: `in_proj_b` (32 × 4096) holds 63 % in 16 directions, but it only has 32 rows.
- WSPLang: rank-one ops (`project_out`, imprints) cannot reproduce such an improvement. Worth knowing before trying.

## F4. MiMo's change mostly goes in new directions, except in the embedding. (measured)
- Share of the delta's top-16 subspace inside the base weight's own top-16 subspace (`base_alignment`, `sig_mimo.pt`):
  embedding 0.56 (chance 0.004); `down_proj` 0.078, `q_proj` 0.072, DeltaNet `in_proj_z/qkv` 0.05, `lm_head` 0.043,
  MLP gate/up 0.036-0.038, `o_proj` 0.003 (chance); DeltaNet β/α at chance (0.5 with 32 rows).
- Reading: in the embedding, the update mainly grows directions the embedding already had; elsewhere it adds new ones.
- WSPLang: "scale what is already big" (cheap spectral ops) can describe at most the embedding part.

## F5. NeoHorse changes 18 times less than MiMo, evenly. (verified)
- 0.5 % relative change; 61 % of weights differ (the rest are below bf16 resolution); flat over layers (0.4-0.6 %);
  norms, `A_log`, `dt_bias` untouched. Check: `check_neohorse.json` agrees.
- Shape: more concentrated than MiMo (16 directions = 6-19 % of the energy in large matrices), still not low rank.
- WSPLang: very different improvements in size (agent skills from a 0.5 % change; reasoning distillation from 9 %).
  Size of change is not a measure of the size of the gain.

## F6. MiMo and NeoHorse change the same kinds of tensors. (measured; specificity open)
- Spearman of the per-tensor-type profile: 0.93 (0.86 for matrices only). Per layer: 0.51 (MiMo bell, NeoHorse flat).
- Meets the pre-registered H1 threshold (>= 0.7) for this pair. Whether it is a mark of improvement or of any
  fine-tune waits for the role-play control (Nyx-RP).

## F7. MiMo and NeoHorse share layer-specific directions of change. (measured; specificity open)
- Same-tensor top-16 subspace overlap (`overlap_mimo_neohorse.json`): embedding 0.71 (×182 chance), `lm_head` 0.33
  (×85), `q_proj` 0.19 (×49), MLP gate/up 0.15-0.16 (×40), DeltaNet `in_proj_z/qkv` 0.15 (×39), `down_proj` 0.09
  (×23), `o_proj` ×15, `v_proj` ×10, DeltaNet `out_proj` ×8, `k_proj` ×6; DeltaNet β/α ×1.1-1.2.
- Cosine of the two deltas: +0.008 to +0.011 in every type, +0.08 in `lm_head`. Small, but with millions of entries
  far above the noise (~0.0002), and the same sign everywhere.
- Structured null (`pairs_mimo_neohorse.json`, `pairs.py`): same-layer overlap / overlap with the same tensor type
  at other layers = `down_proj` ×14.5, `q_proj` ×13.3, `v_proj` ×9.9, DeltaNet `in_proj_qkv` ×9.8 and `in_proj_z`
  ×9.4, `up_proj` ×8.7, `gate_proj` ×8.0, DeltaNet `out_proj` ×7.9, `o_proj` ×6.4, `k_proj` ×4.9: H3 holds for 10
  matrix types. DeltaNet β/α: ×1.1, nothing shared beyond chance.
- Not explained by the base weight: base alignment 0.003-0.08 in those types, against overlaps of 0.03-0.19.
- The embedding overlap (0.71) is: alignment with the embedding's own top subspace is 0.56 (MiMo) / 0.54 (NeoHorse).
  No cross-layer null exists for embedding / `lm_head` (one tensor each).
- The shared subspace is on the smaller side, which for every one of these matrices is the 4096-dim residual stream:
  both fine-tunes re-weight the same residual directions, layer by layer.
- Open: both were trained on chat data with reasoning, so the shared directions may be chat/format directions, not
  improvement. The role-play control (also chat SFT) decides this.
- Method note: the count-sketch cosine has an error of about ±0.016 (sketch size 4096); the exact cosines here are
  ~0.01, so sketch cosines for later pairs only resolve |cos| > ~0.05. Use the exact `overlap` job when needed.

## F8. Same task, measured behaviour: TwIL-8B vs MiMo-9B. (verified by two graders + manual check)
- At an 8192-token cap, greedy: GSM8K 96 vs 93 %, MATH-500 88 vs 68 %; MiMo's 19 cut answers are long reasoning,
  not loops (1 of 19 tails periodic). results.md §37.
- TwIL's balanced variant comes from an RL stage that rewards short reasoning. With a budget, finishing matters
  more than knowing. Candidate study: TwIL balanced vs verbose weight diff (isolates that RL stage).

## Tool lessons (each was a silent error)
- WSPLang `diff` read only the first shard of a sharded model (missed ~3/4 of tensors) and loaded both models in RAM.
  Fixed in `engine/weight_diff.py`, with a test that fails on the old code.
- `engine/comparative_drift.compute_architectural_drift` returns a cached JSON of another model pair whatever its
  arguments; `compute_spectral_delta` ignores a non-GGUF target. The earlier MiMo "anatomy" (gates +14-19 %, norms
  0.14 %) came from GGUF Q6_K and cannot be reproduced; in bf16 the gates are +12.5-12.7 % and norms up to 1.3 %.
  The qualitative claim (gates change most) holds.
- Tensor names differ between multimodal (`model.language_model.*`) and text-only (`model.*`) checkpoints of the
  same base; `Renamed(reader, text_model_name)` lines them up. Without it a diff silently compares nothing.
- Dtype differences (F32 vs BF16) show up as tiny "changes"; check the stored dtypes before reading small deltas.

## F9. Method check passes: the abliteration is rank one, in the residual-writing matrices only. (verified)
- `stats_abliterated.json`: 1.2 % relative change over the model, 18.5 % of weights. Touched: only `mlp.down_proj`
  (31/32 layers), `self_attn.o_proj` (8/8), DeltaNet `out_proj` (23/24), layers 1-31; nothing else.
- Top-1 energy 0.991-0.997 in every touched tensor (pre-registered threshold > 0.9). Second method agrees
  (`check_abliterated.json`).
- So the spectral measures see a low-rank update when there is one: MiMo's and NeoHorse's diffuse updates (F3, F5)
  are real.
- WSPLang: abliteration is exactly the kind of edit WSPLang's `project_out` already expresses (one direction removed
  from the matrices that write to the residual stream). It is the one known "improvement" in this set that is
  compact.

## F10. One refusal direction for all layers; neither MiMo nor NeoHorse moved along it. (measured)
- The abliteration's top-1 output-side direction in `down_proj` is the same in every touched layer: median |cos|
  between layers 1.000 (min 0.293; chance ~0.016). One global direction, projected out of every residual-writing
  matrix: the standard recipe.
- Share of that direction inside MiMo's / NeoHorse's `down_proj` top-16 subspace: median 0.005 / 0.006, max
  0.010 / 0.011, chance 0.004. Neither post-training moved the model along the refusal direction.
- Pairs with the abliteration (`pairs_mimo_abliterated.json`, `pairs_neohorse_abliterated.json`): same-layer overlap
  at the level of the other-layer null (×1.0-1.6), unlike MiMo-NeoHorse (×5-15). Caveat: a rank-one delta has one
  direction with energy; the other 15 vectors of its basis are arbitrary, so this pair test is weak. The direct test
  is the share above.
- WSPLang: a single global direction applied across layers (abliteration) and a layer-specific, many-direction
  change (MiMo, NeoHorse) are different kinds of edit. An op library needs both: `project_out` for the first, and
  for the second something that works per layer on a subspace.

## F11. RL-only (tmax) changes the model by 0.07 %, in the same places. (verified)
- `stats_tmax.json`: relative change 0.073 % (NeoHorse 0.5 %, MiMo 9.2 %); 19 % of weights differ in bf16.
  Second method agrees (`check_tmax.json`). The card reports +6 points on Terminal Bench 2.0 after 200 RL steps.
- Most changed: DeltaNet output norm (`linear_attn.norm`, 0.15 %, 86 % of its weights), MLP down/up/gate
  (0.10-0.11 %), DeltaNet β (0.11 %); then attention (0.06-0.08 %), DeltaNet inputs (0.06 %), conv1d, embeddings,
  `lm_head` (0.02 %). Other norms, `A_log`, `dt_bias` untouched. Slightly more in layers 0-19, falling towards 31.
- Per-tensor-type profile, Spearman: tmax-MiMo 0.70, tmax-NeoHorse 0.83, MiMo-NeoHorse 0.93. Per layer: 0.29,
  0.50, 0.51.
- Caveat: with an update this small, "which weights changed" is limited by bf16 resolution; 19 % is not the true
  sparsity of the RL update.
- Reading: three different trainings (distillation, agentic SFT, RL) share a core: MLP and DeltaNet β change most;
  norms, DeltaNet decay dynamics and embeddings least. Size of the change spans 125× and says nothing about the gain.
- WSPLang: a real improvement can fit in a tiny budget (0.07 % relative change). Budgets should be set in
  relative-norm terms per tensor type, not in counts of changed weights.

## F12. RL-only tmax shares the same layer-specific directions with MiMo and NeoHorse. (measured; specificity open)
- Run as a WSPLang program: `lab/probes/delta_patterns/compare_tmax.wspl` (`delta_compare`), outputs
  `pairs_tmax_mimo.json`, `pairs_tmax_neohorse.json`.
- tmax-MiMo: H3 holds in the same 10 matrix types as MiMo-NeoHorse, ×5.2-10.8 over the other-layer null.
- tmax-NeoHorse: H3 in the same 10 types, strongest in the residual-writing matrices: DeltaNet `out_proj` ×61,
  attention `o_proj` ×31, `k_proj` ×15, `q_proj` ×11, the rest ×7-9. Both are agentic trainings.
- As before: DeltaNet β/α share nothing beyond chance (×0.9-1.1); embedding overlap (0.56-0.58) goes with high
  alignment to the embedding's own top subspace (0.43), i.e. explained by W.
- Reading: three different trainings (distillation, agentic SFT, RL-only) move common residual directions layer by
  layer. Whether those are improvement directions or the mark of any chat fine-tune is decided by the role-play
  control (Nyx-RP), still running.
- Tool lesson: the WSPLang CLI has a second list of implemented ops (`check_feasibility`); unit tests that call
  `execute_plan_v4` bypass it. The new ops failed the CLI until added there. A test now covers the gate.

## F13. Fara (computer-use agent) repeats the core; H1 now holds for all six improvement pairs. (verified)
- `stats_fara.json`: relative change 3.9 %, 86 % of weights. Second method agrees (`check_fara.json`).
- Most changed: DeltaNet β 5.5 %, MLP down/up/gate 4.5-4.9 %, `lm_head` 4.3 % (top-16 energy 0.56: concentrated),
  attention q/k 3.9 %; DeltaNet `A_log`, `dt_bias` untouched. Bell over depth, peak at layers 13-19.
- Per-tensor-type Spearman with Fara: MiMo 0.95, NeoHorse 0.985, tmax 0.76; per layer 0.70, 0.92, 0.60.
- All six pairs of the four improvements have type-profile rho >= 0.70 (pre-registered H1 threshold). Specificity
  (vs the role-play control) still open.

## F14. Six of six improvement pairs share layer-specific directions; agents cluster on residual-writing matrices. (measured; specificity open)
- `compare_fara.wspl` (`delta_compare`): Fara vs MiMo, NeoHorse, tmax: H3 in the same 10 matrix types each time.
  With F7 and F12, all six pairs of the four improvements (MiMo, NeoHorse, tmax, Fara) meet H3 in those 10 types.
- Strongest ratios over the other-layer null:
  - agent pairs (Fara-NeoHorse, Fara-tmax, NeoHorse-tmax): DeltaNet `out_proj` ×28-61, attention `o_proj` ×19-31;
  - pairs with MiMo (reasoning distillation): `down_proj` ×11-17, `v_proj` ×6-14, `q_proj` ×9-13.
- DeltaNet β/α: ×0.9-1.2 in every pair (nothing shared beyond chance), although β is among the most changed tensors
  in every model: they all change it a lot, in unrelated ways.
- Reading: a common core of shared directions, plus a tighter "agent" family in the matrices that write to the
  residual stream. Still to rule out with the role-play control: that this is the mark of chat fine-tuning.

## F15. Role-play control: H1 passes both pre-registered rules, but the control is a merged LoRA (confounded). (verified)
- `stats_nyx_rp.json` (Indexnusrefather/Nyx-RP-9B-Instruct-2608-v1): relative change 0.69 %, 48 % of weights. Second
  method agrees (`check_nyx_rp.json`).
- It touches only attention q/k/v/o and MLP gate/up/down (0.68-1.19 %); every DeltaNet tensor, the embeddings,
  `lm_head` and all norms are exactly unchanged. Its changes are low rank: 16 directions hold 66-88 % of the energy
  (improvements: 2-19 %). That is the fingerprint of a merged LoRA on the usual target modules.
- H1 by the plan: improvement-improvement type-profile rho 0.70-0.985 (min 0.702); improvement-control rho 0.42-0.47,
  all below 0.702 → "shared where" true and "improvement-specific" true.
- Caveat, recorded as such: the control differs from the improvements both in goal (role-play) and in method
  (LoRA vs full fine-tune). The H1 specificity result may come from the method. A full-fine-tune control that is
  not aimed at reasoning or agents is needed before calling the "where" pattern a mark of improvement.
- Still useful for H3: in the 7 matrix types it touched, if the control does not share the improvements' directions
  while the improvements share them with each other, that points to specificity independent of method.
- WSPLang / method lesson: a control model must be checked for its training method from its weights before it is
  used as a control; the weight fingerprint (which modules, which rank) reveals LoRA vs full fine-tune.

## F16. Two more controls, added after F15 to separate method from goal (post-hoc additions, labelled as such)
- Probe: two tensors per candidate read by HTTP range (layer 10 DeltaNet `in_proj_qkv`, layer 10 `mlp.gate_proj`),
  compared with the base; nothing else of their weights seen before adding them.
  - `tencent/WeMM-Embedding-9B` (turned into a multimodal embedding model): full fine-tune — DeltaNet in_proj_qkv
    10.5 %, gate_proj 15.5 %, 16 directions hold 51 % / 39 % of the energy; norm untouched. Goal clearly not
    reasoning or agents: the control F15 asked for (full fine-tune, other goal).
  - `axiomofmind/Angry-Claudius-9B` (novelty fine-tune: a short profane refusal to every request, i.e. a deliberate
    degradation): touches DeltaNet but little (0.20 %), gate_proj 0.31 %, low rank (74-83 % in 16 directions):
    looks like a LoRA on all linear layers. A second, "anti-improvement" control.
- Decision rules of plan.md unchanged; these controls are reported separately from the pre-registered one.
- Method lesson: a two-tensor range probe (seconds, a few MB) tells full fine-tune from LoRA and big from small
  updates before committing to a 19 GB download.

## F17. DeepSeek-V4-Pro distill of Qwen3.5-9B is LoRA-like; its Heretic version documents its layers (probe, 4 tensors)
- `Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B` (card: distillation of DeepSeek-V4-Pro "Max Effect" answers, ~250K math and
  STEM samples, base unsloth/Qwen3.5-9B): layer-10 DeltaNet `in_proj_qkv` unchanged (0), `input_layernorm`
  unchanged; layer-10 `mlp.gate_proj` 9.4 % with 58 % of the energy in 16 directions; layer-20 `down_proj` 6.4 %
  (54 %). Fingerprint of a LoRA on attention/MLP modules, not of a full fine-tune: a reasoning distillation done
  differently from MiMo's (full, diffuse).
- `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` (Heretic v2.0.0.dev0, ARA, card: layers 9-24, refusals
  100/100 -> 3/100, KL 0.0422) on top of it: in the probe, the only extra change is layer-20 `down_proj`
  (6.4 % -> 7.9 %, 16-direction energy 0.54 -> 0.69), inside the documented range.
- A documented ARA on our own local base: ground truth for the decensoring forensics, cheaper than the 27B.
- Same "reasoning distillation" label, two different weight fingerprints: the label on a card does not say how a
  model was changed; the weights do.

## F18. A fast report from blocks of rows works; two lessons from validating it (engine/model_report.py)
- Fingerprint from one block of 128 rows per tensor (contiguous bytes, one HTTP range each). MiMo vs base, local:
  27 s, method `full_fine_tune`, per-type profile Spearman 0.999 against the full statistics (second method).
- Heretic ARA uses a direction per layer, not one global direction: on xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-
  UNCENSORED-heretic the changed tensors are only residual writers, rank 1 per tensor, but the leading directions of
  different layers are unrelated (median |cos| 0.07). The first rule (required one shared direction, from the
  classic abliteration of F10) called it `lora_like`. Rule changed after this validation failure, and recorded as
  such: residual-writers-only + low rank = `directional_ablation`, with `global` or `per_layer` reported. The new
  rule must hold on cases that did not inspire it (huihui abliteration, the 27B Heretic).
- Heretic's `end_layer_index` is exclusive (p-e-w/heretic `modifiers/ara.py`: `range(start, end)`): the card's
  "9-24" is layers 9-23, which is exactly what the report found from weights alone. The 27B plan was corrected the
  same way (26-56 → 26-55) before anything was unsealed.
- Speed: remote report 773 s for ~500 MB = 0.7 MB/s while the link gives ~20 MB/s. The cost was the latency of ~860
  sequential HTTP requests, not Python nor the maths. Fix: blocks fetched concurrently (16 in flight). Rewriting
  WSPLang in Rust would not change this; fewer bytes and parallel requests do.

## F19. model_report validated: 6/6 known cases, 21-42 s per 9B model without downloading it (verified)
- `lab/runs/model_report/validation.json` (blocks of 128 rows, 16 concurrent reads):
  - Heretic ARA (DeepSeek-V4 9B): directional_ablation, layers 9-23 (= card with exclusive end), rank 1,
    per-layer directions (median |cos| 0.069). 42 s, both models remote (773 s before parallel reads).
  - huihui abliteration: directional_ablation, layers 1-31, rank 1, one global direction (|cos| 1.000). 22 s.
  - Nyx-RP: lora_like, block rank 33-41 (a LoRA of rank 32-64). 24 s.
  - tmax (RL): full_fine_tune. 22 s. MiMo: full_fine_tune, profile Spearman 0.999 vs full statistics. 27 s.
  - NeoHorse: full_fine_tune, profile Spearman 0.957 vs full statistics. 21 s.
- The abliteration and Nyx-RP, tmax, NeoHorse did not inspire the corrected rule (F18); the global vs per-layer
  distinction comes out right on the abliteration.
- Caveat: "untouched" means untouched in the sampled block. NeoHorse's `linear_attn.norm` is untouched in the block
  and changed (tiny) in the full statistics. Reports must say "in the sampled rows".
- WSPLang: a 9B model can be fingerprinted in under a minute from Hugging Face without downloading it.
