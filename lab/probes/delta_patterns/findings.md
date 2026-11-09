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
  [Corrected in F26: `A_log` changes 3-7 of 32 entries by 1-8 bf16 steps in some layers; tiny but not a cast.]
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
- [Corrected in F26: the `linear_attn.norm` entry below was the F32->BF16 cast; after rounding it is unchanged.]
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

## F20. "What can be transferred" from blocks of rows: needs 512 rows; a LoRA distillation shares none of the core
- `transfer_candidates` (engine/model_report.py) compares the column-side (residual) directions of a block of the
  model's delta with the atlas signatures of the same tensor, with the H3 rule (>= 3x the other-layer null and chance).
- Second method: MiMo's blocks against the full signatures of NeoHorse, tmax and Fara, verdicts compared with the full
  pair analyses, same rule, only the block size changed:
  128 rows → 12/21 agree (0 false positives, 9 false negatives); 256 → 17/21 (0, 4); 512 → 21/21 (0, 0).
  The report keeps 128-row blocks for the method (validated in F19) and reads 512-row blocks for transfer, only for
  column-side, non-residual-writing types. Residual-writing matrices are reported as needing a full signature.
- New: Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B (LoRA-like reasoning distillation, F17) shares no column-side direction
  with the four full-fine-tune improvements: gate ×1.3-1.9, up ×1.2-1.6, q ×1.3-1.5 (threshold ×3). 78 s, remote.
- Reading, open: the shared core may go with the method (full fine-tune) more than with "improvement". The
  full-fine-tune control with another goal (WeMM-Embedding, queued) decides between the two.

## F21. Qwen's own post-training moves the same layer-specific directions as the four community improvements (measured)
- `stats_official_post.json` (Qwen/Qwen3.5-9B-Base -> Qwen/Qwen3.5-9B, Qwen's SFT + RL): 5.0 % relative change, 96 %
  of weights; most changed DeltaNet β 7.1 %, MLP 5.8-6.0 %; `A_log`, `dt_bias` untouched. Second method agrees.
- Type-profile Spearman with MiMo 0.983, Fara 0.976, NeoHorse 0.958, tmax 0.713; with the role-play LoRA 0.428.
- `compare_official.wspl`: H3 in all 10 matrix types against each of MiMo, NeoHorse, tmax, Fara; strongest
  `down_proj` ×28.7 (MiMo), ×12-14 (others), `v_proj` ×8-16, `q_proj` ×9-13, `out_proj` ×9-11.
- Every community improvement starts from Qwen3.5-9B, i.e. after this post-training. Their updates keep moving the
  model along the directions Qwen's post-training already moved: one training trajectory continued, not new
  directions.
- Candidate for "mejora sintética": move Qwen3.5-9B further along the official post-training delta projected on its
  top subspace (`subspace_transfer` with source = official instruct, base = Base, α > 0), against a random subspace
  of the same size (plan H4). Alternative reading still open: generic "instruction-tuning" directions; the
  full-fine-tune control with another goal (WeMM-Embedding) tests it.

## F22. The "where" pattern is the mark of any full fine-tune, not of improvement (H1 specificity rejected) (verified)
- `stats_wemm_embed.json` (tencent/WeMM-Embedding-9B, turned into an embedding model; 425 text tensors, the output
  head is gone): 13.7 % relative change, 99 % of weights. Most changed: DeltaNet β 21 %, MLP gate 18 %, DeltaNet
  z 16 %, α 16 %, up 15 %; untouched: all norms, conv1d, `A_log`, `dt_bias`. Second method agrees.
- Type-profile Spearman with the improvements: MiMo 0.897, NeoHorse 0.861, Fara 0.858, tmax 0.705; with the official
  post-training 0.862. The smallest improvement-improvement rho is 0.702: by the pre-registered rule, H1's
  "improvement-specific" is false. F15's pass came from the method (the role-play control is a LoRA).
- Conclusion: "MLP and DeltaNet β change most, norms and DeltaNet decay dynamics stay" is how full fine-tuning of this
  architecture moves the weights, whatever the goal. For WSPLang: a map of where training acts (protect lists,
  per-type budgets), not a recipe for improvement.
- Still open and now the key question: H3 (shared directions). If WeMM shares the improvements' layer-specific
  directions too, that core is generic as well; if not, the directions carry what the "where" does not.

## F23. H3 is specific: a full fine-tune with another goal shares none of the improvements' directions (measured)
- `compare_wemm.wspl` (`delta_compare`, WeMM-Embedding against MiMo, NeoHorse, tmax, Fara and the official
  post-training): same-layer overlap / other-layer null ×1.2-2.5 in every matrix type; H3 in 0 of 12 types for each
  of the four improvements and 0 types for the official post-training. Among the improvements the same types give
  ×4-61 and H3 in 10 of 10 types for all six pairs.
- With F22: WHERE a model changes (which tensor types, how much) is the mark of any full fine-tune of this
  architecture; the DIRECTIONS it changes, layer by layer, separate the improvements (and Qwen's own post-training)
  from a full fine-tune with another goal.
- Limit of this conclusion: all improvements are generative chat fine-tunes; WeMM is no longer generative (trained
  as an embedding model). The shared directions may mark "continued assistant training" rather than "improvement".
  Next control: Angry-Claudius (generative chat fine-tune that degrades the model on purpose).
- WSPLang: the transferable object is the shared subspace (direction), not a where-profile. `model_report`'s transfer
  section and `subspace_transfer` work on exactly that.

## F24. A deliberate degradation (Angry-Claudius) shares no core direction with the improvements, but neither does a neutral LoRA: the test cannot separate "degradation" from "LoRA" (measured, 2026-11-06)
- `stats_angry_claudius.json` (axiomofmind/Angry-Claudius-9B): 0.23 % relative change, 36 % of weights moved,
  LoRA-like on every linear layer; norms, embeddings, conv1d, `A_log`, `dt_bias` untouched. Second method agrees
  (`check_angry_claudius.json`, 3/3). Pipeline resumed after a low-memory kill (threads 4, 2 download workers).
- `compare_angry.wspl` against MiMo, NeoHorse, tmax, Fara, official post-training: H3 in 0/12, 2/12, 1/12, 2/12,
  1/12 types. The only passes are `k_proj`/`v_proj` (×3.1-4.1, just over the ×3 bar; these have 1024 rows, chance
  1/64). The core types (MLP, DeltaNet, `o_proj`, `q_proj`) are ×1.0-2.6, i.e. at the null.
- Method control, run now for the first time: `compare_nyx.wspl` (Nyx-RP, a role-play LoRA, not meant to degrade):
  H3 in 2, 2, 1, 2, 2 of 12 types, the same marginal `k_proj`/`v_proj` (plus `down_proj` ×3.0-3.7 for MiMo and
  the official post-training).
- Reading: both LoRAs behave alike, so this control does not tell "degradation" from "improvement". The difference
  from the improvements (×4-61, 10/10 types) may be the method (LoRA vs full fine-tune), as F15 already warned.
  F23 (WeMM, a full fine-tune, 0/12) still stands as the specificity evidence. Its caveat ("continued assistant
  training" vs "improvement") is still open. The control that would settle it is a full-fine-tune generative chat
  model that is worse, or not better, than Qwen3.5-9B. Search for one by the method shown in its weights
  (`model_report` classify = full_fine_tune), not by what its card says.
- Side note: the marginal `k_proj`/`v_proj` agreement appears with every LoRA and with WeMM's ×2-2.5. The small
  attention K/V spaces may make the H3 bar easier to cross there. For K/V, use a stricter bar or a permutation null
  before reading anything into a pass.

## F25. MiMo ships an older pre-tokenizer regex than its base; harmless in GGUF, wrong in transformers/vLLM (measured, 2026-11-06)
- Non-weight diff, MiMo (`hf_cache/mimo-distill-hf`) against Qwen/Qwen3.5-9B (config, tokenizer and template
  downloaded to `hf_cache/qwen35-9b-hf`):
  - **config:** same architecture. The differences are serialization only (transformers 5.12 vs 4.57;
    `partial_rotary_factor` 0.25 moved out of `rope_parameters`).
  - **vocabulary:** vocab and merges identical (merges equal after normalizing list vs string format).
  - **added tokens:** 7 audio/TTS special tokens added (`<|audio_start|>`, `<tts_pad>`, ...; ids 248070-248076).
  - **chat template:** none in the HF folder. It ships in the GGUF.
- **The pre-tokenizer split regex differs.** Base Qwen3.5: letters with combining marks `[\p{L}\p{M}]+`. MiMo:
  the Qwen2-era `\p{L}+`. Same ids on English, Spanish, code, CJK, Korean, Vietnamese and 290/290 500-character
  chunks of `corpus_n4.txt`. Different on scripts with combining marks: Hindi 11 -> 19 tokens, Thai 7 -> 15,
  Bengali 8 -> 15, Arabic with diacritics 7 -> 9. MiMo's segmentation there is one the base was not trained on.
- **Second check: the GGUF.** The MiMo GGUF (ollama blob `sha256-ef96...`) has `tokenizer.ggml.pre = qwen35`, so
  llama.cpp uses the base regex. Our benchmarks (§37) are not affected. Users of the HF tokenizer
  (transformers, vLLM) get the old split.
- Unknown from the weights: which tokenizer MiMo was trained with. If it was this one, the model has seen the
  old split on those scripts, and GGUF users get the mismatch instead. Test: perplexity on Hindi/Thai text under
  both tokenizations.
- Point for model_report: a full report needs a "non-weight" section (config, vocab, merges, added tokens,
  pre-tokenizer, template) and an encode test on several scripts. A hash says "different". The encode test says
  where it matters.

## F26. A storage cast was counted as change; corrected. Conclusions hold, profile correlations rise (verified, 2026-11-06)
- Bug: `tensor_delta_stats` compared F32 base tensors (`linear_attn.norm`, `linear_attn.A_log`, 48 per model) with
  BF16 copies without rounding both to BF16 first. Fixed in `engine/weight_diff.py` (`common_dtype`, test
  `test_dtype_cast_is_not_a_change`); `fix_dtype_cast.py` recomputed only those tensors and rewrote the stats.
  Audit: `lab/runs/delta_patterns/dtype_cast_fix.json`.
- Affected: mimo, neohorse, tmax, fara, wemm_embed, angry_claudius, dsv4_heretic (48 cast tensors each). Not affected:
  abliterated, nyx_rp, official_post, dsv4_distill (same dtypes on both sides).
- After rounding, the 24 "changed" norms of wemm_embed, angry_claudius and dsv4_heretic are exactly unchanged.
  Real changes remain in mimo (19 tensors > 1 ulp, 19 at <= 1 ulp), fara (6 > 1 ulp), neohorse (2), tmax (1).
- Correction to F2: `A_log` is not just a cast. The base's F32 `A_log` values are exactly representable in BF16 (cast
  error 0), so a pure cast would show no difference; MiMo moves 3-7 of 32 entries by 1-8 BF16 steps at their value
  (|delta| <= 0.004; e.g. layer 14: -0.10742 -> -0.10938). Tiny, but trained or rewritten. Layers 6 and 14 recur in
  mimo, fara, neohorse and tmax (open: shared starting revision, or training).
- Correction to F11: tmax's "most changed: `linear_attn.norm` 0.15 %, 86 % of weights" was the cast; after rounding
  all of tmax's norms are unchanged; only `A_log` of layers 6 (2 ulps) and 14 (1 entry, 1 ulp) differ. The "most
  changed" list now starts at MLP down/up 0.11 %, DeltaNet β 0.106 %, gate 0.099 %.
- Per-type profile Spearman (average ranks for ties, `dtype_cast_fix.json`; these "before" values differ by up to
  0.13 from earlier findings, which ranked ties in order):
  - improvement pairs after the fix: mimo-neohorse 0.947, mimo-tmax 0.893, mimo-fara 0.958, neohorse-tmax 0.932,
    neohorse-fara 0.988, tmax-fara 0.905; minimum 0.893 (was 0.687 by this method). H1 "shared where" holds, stronger.
  - WeMM control: mimo 0.920, neohorse 0.877, tmax 0.821, fara 0.893. Not every value is below 0.893, so the H1
    "improvement-specific" verdict of F22 (rejected) stands.
  - Nyx-RP LoRA control: 0.54-0.68, all below: F15's pass stands, and so does its LoRA confound.
- H3 (directions) is unaffected: signatures use only matrices, and none of the cast tensors is a matrix.
- Lesson (same class as the viral WSP posts we audited): a storage format difference read as a weight change. Every
  diff now rounds to the common dtype before comparing.

## F27. Reusing MiMo's change on Qwen3.5-9B: the full delta is trivial, rank-k parts carry little (measured, 2026-11-08)
Goal stated by the user: compare the Qwen base with MiMo-V2.6 Qwen, see what changed, and see how to reuse it on the base.
- **Pair (verified from primary sources today):** the model card of `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` says it is
  supervised fine-tuning of Qwen3.5-9B on MiMo-generated data (77.4B tokens). Text configs match field for field with
  `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-9B-Base`. Read live by header: 427 text tensors with identical names and shapes in
  MiMo and Qwen3.5-9B. "Base" is ambiguous: the SFT parent is Qwen3.5-9B (post-trained), not `-Base`; the existing
  analysis uses Qwen3.5-9B. Using `-Base` instead would mix Qwen's own post-training into the delta.
- **Three different claims, only the first two are made here:**
  1. structural comparison (what changed, where): F1-F3, F26. Done.
  2. delta extraction and how much a part carries (this entry). Done for a sample.
  3. behavioural proof that applying a part to the base helps: NOT done. Needs inference on a 9B model, which this
     environment cannot run (2 CPUs, about 2 GB RAM).
- **Full delta:** base + (MiMo - base) is MiMo exactly. That is a copy, not reuse, so it proves nothing.
- **Energy by family** (from `stats_mimo.jsonl`, squared delta norm): mlp 60.3 % (54.0 % of parameters), ssm/DeltaNet
  20.6 % (18.1 %), head 8.1 % (11.4 %), attn 5.5 % (5.2 %), embed 5.5 % (11.4 %), norms about 0. Delta energy follows
  parameter share closely; the MLP is not unusually concentrated. Top-16 directions per tensor keep 4.4 % in total.
- **Rank needed** (`reuse_curve.py`, full tensors by HTTP range, 4 tensors, `lab/runs/delta_patterns/reuse_curve_mimo.json`):
  | tensor | min dim | rank for 50 % / 80 % / 90 % | energy kept at rank 16 / 64 / 256 / 1024 |
  |---|---|---|---|
  | L3 v_proj | 1024 | 191 / 476 / 651 | 9 % / 25 % / 59 % / 100 % |
  | L17 gate_proj | 4096 | 657 / 1688 / 2369 | 4 % / 10 % / 27 % / 64 % |
  | L17 down_proj | 4096 | 708 / 1748 / 2425 | 3 % / 8 % / 25 % / 62 % |
  | L17 in_proj_qkv | 4096 | 516 / 1363 / 1979 | 4 % / 12 % / 32 % / 71 % |
  A rank-k transplant is cheap only for tiny k, and there it carries a few percent of the change. Reaching 80 % of
  one MLP matrix's delta takes about 40 % of its rank, i.e. a large part of the full cost. Energy is not function:
  a small-energy part could still matter, and a large-energy part could be mostly noise. Only the behavioural test
  can say, and it has not been run.
- **Reading for reuse:** the update is diffuse in magnitude. A compact, transferable "patch" is not visible in the
  spectrum. Candidates that remain open are tensor-type subsets (for example MLP only, or layers 14-23 where the
  change peaks), compared with a random subset of equal delta norm, as pre-registered in plan.md H4. They need a
  behavioural evaluation and therefore an inference machine.
- Limits: 4 tensors of 427, one layer mostly; the families' rank curves were not all measured.
- **F27 addendum, depth check (8 more tensors, about 1.1 GB read in total, 8 tensors in `reuse_curve_mimo_depth.json`;
  12 of 427 tensors measured in all).** Layers 0 and 30 for gate_proj, down_proj and in_proj_qkv, v_proj at layers 7
  and 31. The rank for 80 % of the energy stays at 1630-1890 (MLP) and 1250-1390 (in_proj_qkv); rank-16 energy stays
  at 3-8 % for large matrices. v_proj keeps rank 400-485 for 80 % (rank-16 energy 7-15 %). No depth trend that would
  make a shallow or deep layer a compact patch. Reference used: Qwen3.5-9B (the declared parent), an exploratory choice, not
  the user's confirmed "base". Sources: HF repos `Qwen/Qwen3.5-9B` and `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` at `main`
  as read on 2026-11-08 (revisions not pinned; pin them before any behavioural run).

## F28. MiMo's delta is not sparse either: no neuron, channel or entry patch (measured, 2026-11-08)
Second axis besides rank (F27): can a pruned delta (TIES/DARE-style entries, or a set of neurons or hidden channels)
carry most of the change? Same 4 tensors as F27, revisions pinned: base `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
model `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B@1f09ddcc1a0ea45751ae0dfe2af20d533168c46f` (the third repo, `Qwen/Qwen3.5-9B-Base@68c46c4b3498877f3ef123c856ecfde50c39f404`, is not used).
Data: `lab/runs/delta_patterns/reuse_sparsity_mimo.json` (reuse_curve.py, entry thresholds from a 1M-entry sample).
- Largest 1 % / 10 % / 30 % of entries keep 9.5-10.8 % / 46-50 % / 80-82 % of the energy. A Gaussian keeps about 8 % / 46 % / 79 %,
  so the tails are only slightly heavier than noise.
- Largest 1 % / 10 % of rows keep 1.2-1.8 % / 12-16 %; of columns 1.2-2.2 % / 12-18 %. Uniform would be 1 % / 10 %.
  No neuron-level or channel-level (outlier dimension) concentration in these tensors.
- Reading: not low rank (F27), not sparse (F28). Whatever MiMo's SFT adds is spread over most weights. A small reusable
  patch, if it exists, is defined by function and not by size, so only a behavioural test can find it. Not tested here.
- Same limits: 4 of 427 tensors. A first run of the sparsity code was killed by the memory limit (sorting 50M entries),
  and was replaced by the chunked version; no number from the killed run is used.

## F29. One layer, real embeddings: a rank-16 part of MiMo's delta reproduces 26 % of the layer's output change (measured, 2026-11-08)
Script `layer_probe.py`, data `lab/runs/delta_patterns/layer_probe_L0_mimo.json`, pinned revisions as in F28. Definitions
(`captured`, `resid`, random controls) are in the script docstring and were written before the first run.
**Scope: layer 0 of 32 (a Gated DeltaNet layer), float32 CPU, 4 prompts / 95 tokens. Input = real token embeddings from
the base table (what layer 0 receives; no earlier layer exists). It is NOT behaviour and NOT the full model. Deeper layers need real
activations from the layers before them, not done.** The MiMo shift of the layer output is 4.7 % of the output norm. Sanity rows:
base+full delta = exactly the MiMo layer (captured 1.0, resid 0.0), base alone = 0.0, 1.0.
- Parts of the delta added to the base layer (captured; same-norm Gaussian control on the same tensors in brackets):
  MLP all 0.44 [0.00], DeltaNet all 0.58 [0.03], gate+up 0.26 [0.00], down 0.17 [0.00], in_proj_qkv 0.22 [-0.01], out_proj 0.16 [0.03],
  norms 0.003 [0.00] (they barely move this layer).
- Truncated to rank k in all 8 matrices: rank 16 0.26, rank 256 0.61, rank 1024 0.83 (per prompt 0.16-0.32, 0.40-0.73, 0.70-0.91).
  Energy-fair control, a random subspace of the same size on the same side, 2 seeds: rank 16 gives 0.00 and 0.00, rank 256 gives 0.05 and 0.05.
- Largest 10 % / 30 % of entries: 0.29 / 0.61.
- Reading: rank-16 parts hold 4-7 % of the weight energy (F27) but reproduce 26 % of this layer's output change; the same size in a random
  subspace reproduces none. So in this layer the delta's function is more concentrated than its energy, consistent with it acting along the
  directions where the input carries most of its variance. This undoes the earlier weights-only
  impression that nothing small is reusable: weights alone cannot decide that, and this probe is evidence in the other direction.
- NOT shown: that it holds beyond layer 0, on other inputs (4 short prompts, one of them a chat template), through the stack of layers, or in generation;
  that transplanting the part into the base improves any task. The first-layer input is also the least processed one.
- Next, bounded by memory (2 GB RAM): layers 1-3 need exact activations from the layers before, which means running those layers of the base
  (one layer is about 0.4 GB per model in bf16, run one at a time and keep only the hidden states): a depth-wise chain is possible without ever loading
  the full model.

## F30. Layers 0-2, 12 prompts (4 dev + 8 held out), cosine added: the rank-k effect holds on held-out prompts and weakens after layer 0 (measured, 2026-11-08)
Same probe as F29 (data `layer_probe_L{0,1,2}_mimo.json`, pinned revisions). Changes since F29: (1) 8 held-out prompts, written after the dev numbers
of F29 were seen and before any number on them existed, reported apart from the 4 dev prompts; the delta bases come from weights only; (2) cosine and norm
ratio added; (3) layer 0 re-run (the dev numbers reproduce exactly); (4) layers 1 and 2 use the hidden states the BASE model produces on the same
tokens (saved layer by layer), NOT MiMo's own trajectory: both models receive the base model's activations at that depth. Layer 0 receives the base embedding table's rows, also not MiMo's own embeddings.
Metric (pooled over all tokens and dims of the group): captured = <d, shift>/<shift, shift> (projection coefficient, can be < 0 or > 1; not 1 - error), cos = direction agreement, d = variant output - base output, shift = MiMo output - base output.
captured (cos in brackets), dev | held out; "rand sub." = random subspace of the same size on the same side of the same 8 matrices (seed 1):
| layer | rank 16 | rand sub. 16 | rank 256 | rand sub. 256 | rank 1024 | MLP only | attention only |
|---|---|---|---|---|---|---|---|
| 0 | .26 (.59) \| .24 (.55) | .00 \| .00 | .61 \| .57 | .05 \| .06 | .83 \| .82 | .44 \| .47 | .58 \| .55 |
| 1 | .11 (.33) \| .14 (.39) | .00 \| .00 | .45 \| .48 | .06 \| .06 | .76 \| .78 | .64 \| .59 | .35 \| .41 |
| 2 | .11 (.34) \| .20 (.49) | .00 \| .00 | .38 \| .50 | .07 \| .06 | .64 \| .74 | .52 \| .51 | .46 \| .48 |
- Held out holds: rank 16 gives .24 / .14 / .20 against .00 for the random subspace, and rank 256 gives .57 / .48 / .50 against .05-.06. The effect is not an artefact of the 4 prompts used to design the probe.
- It is weaker than in layer 0 (rank 16: .26 -> .11-.20) and depends on the prompt: per prompt in layer 2, rank 16 goes .06 to .37 and rank 256 .28 to .69. Three layers say nothing about the other 29.
- A random subspace of rank 256 still gets cos .2-.3 against .0 for rank 16; direction agreement is not zero for noise projected on the weights' own subspace, so cos is read next to the control, not alone.
- The MLP delta carries more of the change than the attention delta from layer 1 on (.64 vs .35 in layer 1, dev).
- What this is: a partial reproduction of a LOCAL output difference, on inputs that are the base model's own activations. What it is not: a benefit. MiMo's change on these tokens might be neutral or harmful
  for a given task, and a part that reproduces it would then reproduce that. No generation, no benchmark, no behavioural claim. The earlier statement that the full-delta layer equals the MiMo layer
  only shows the code rebuilds the weights.
- Not done: layer 3 and beyond (layer 3 is full attention and needs the causal-mask path, implemented but not run), MiMo's native trajectory, other prompt domains at scale (12 short prompts), any task.

## F31. Layer 3 (full attention) and a composed stack of layers 0-3: the rank-k effect survives composition (measured, 2026-11-08)
Data: `layer_probe_L3_mimo.json`, `stack_L0-3_mimo.json`; same probe, pinned revisions, 4 dev + 8 held-out prompts. Cost: 0.4 GB per model for layer 3, no new reads for the stack (cache),
about 6 min of CPU for the stack. Still: base embeddings in, a short stack, no generation, no task.
- Implementation checks. Layers are the transformers 5.20.2 `Qwen3_5DecoderLayer` in float32 (eager attention, explicit causal mask for the full-attention layer 3; the DeltaNet
  layers use the library's torch fallback because `flash-linear-attention` is not installed). Checked: changing the last token's input leaves earlier outputs bit-identical (max diff 0.0 in layer 2 and in layer 3) and moves the last one. Not checked: agreement with the vendor runtime (sglang/llama.cpp) numerically.
- Layer 3 alone (captured | resid | cos | norm ratio, held out in brackets): rank 16 .15 | .92 | .39 | .39 (.14 | .93 | .36 | .37); rank 256 .50 | .68 | .74 | .67 (.48 | .68 | .74 | .66);
  rank 1024 .77 | .43 | .90 (.77 | .43 | .90). Random subspace of the same size: rank 16 .00-.01 | 1.00 | .07-.13 (norm ratio .05); rank 256 .08-.09 | .94 | .35-.39 | .22.
- Composition (the same rank in all 8 matrices of layers 0, 1, 2, 3 at once, each layer fed by the patched stack's own previous layer). The MiMo shift of the stack output is
  the reference; full delta in the 4 layers rebuilds it (sanity 1.00 / 0.00). dev | held out:
  | variant | captured | resid | cos | norm ratio |
  |---|---|---|---|---|
  | rank 16 | .11 \| .15 | .94 \| .91 | .35 \| .44 | .30 \| .33 |
  | rank 16, random subspace (2 seeds) | .01 \| .01 | 1.00 \| 1.00 | .11 \| .10-.12 | .06 |
  | rank 256 | .41 \| .47 | .70 \| .66 | .74 \| .77 | .55 \| .61 |
  | rank 256, random subspace (2 seeds) | .08 \| .07 | .95 \| .95 | .35 \| .34 | .21 |
  | top 10 % entries | .33 \| .37 | .72 \| .68 | .79 \| .82 | .42 \| .46 |
  | MLP delta only, full | .60 \| .63 | .55 \| .52 | .85 \| .86 | .70 \| .74 |
  | all 8 matrices, full delta (norms and DeltaNet constants left at base) | .98 \| .99 | .13 \| .12 | .99 \| .99 | .99 |
- Reading: composed in four layers, rank 256 reduces the relative error against MiMo's stack output from 1.00 (base) to .66-.70, and the random control only to .95. Rank 16 reduces it to .91-.94 (control 1.00). The per-layer effects
  do not cancel or collapse when stacked; the composed numbers sit near the isolated ones (layer 0-3 rank 256 .38-.61). resid answers "is the output closer to MiMo's": yes, partially and clearly better than a same-size random part.
  The difference between full matrices (.13) and everything (0) lies in the norms, A_log/dt_bias and conv1d, whose delta is small but not zero.
- The gap to the control is the evidence; the absolute level (rank 16 closes 6-9 % of the distance) is small, and rank 256 closes 30-34 %, about a quarter of the full delta's size per matrix (256 of 4096).
- Still NOT shown: any benefit on a task, MiMo's native trajectory (embeddings differ), layers 4-31, long contexts, that the closer output is a better output.

## F32 - delta_pack reproduces the probe's rank mode (layer 0, rank 16, 6 matrices)

`demo_pack_layer0.py` builds a pack (manifest + factors, factors kept under /tmp, not committed) for the six large layer-0 matrices and compares `apply_pack` with
the probe's own "rank" formula (basis files from `layer_probe.py`). Max abs difference 2e-8 to 6e-8, relative to |delta| 7e-7 to 2.4e-6 (float32 rounding and
eigenvector sign choice do not matter for a projection). `verify_tensors` on the base returned no mismatches. Energy kept at rank 16: 3.1% to 6.9%.
Manifest (hashes only, no weights): `lab/runs/delta_patterns/pack_manifest_layer0_rank16.json`; it states `behavioural_benefit: "not shown"`.
This checks the packaging against the probe, one layer. It says nothing about benefit. Redistribution of factors is not decided (both licences apply).

## F33 - select-and-apply the delta (engine/delta_apply.py): mechanics only

Layer-0 cache, both models: 98.0% of the 218.4 M elements of the 10 comparable tensors differ between Qwen3.5-9B and MiMo, and F27 measured diffuse,
not low-rank structure (rank 16 keeps 3-7% of the delta energy). That rules out a compact EXACT reconstruction of the delta from what was measured
(12/427 tensors). It does not rule out a compact functional component: F29-F31 show rank-16/256 parts reproduce part of the layer output shift
(a local measure, not benefit). Which subsets are useful is open.
`apply_selected(base, model, select, alpha, rank)` is an in-memory function over tensor readers, covered by 4 unit tests on tiny tensors. It is NOT a
ready 9B patch on disk and has no CLI; it has not been run on the full checkpoints. Per tensor it leaves unselected tensors as the base, skips and reports
shape/dtype mismatches instead of casting, never modifies its inputs, and records sha256 of base, model and output. alpha=1 with the full delta reproduces
the model up to float32 rounding (layer 0: 2-27 elements per matrix, abs error <= 1.2e-10): a tautology, not a result.
No behavioural test of any selection was run. The streaming scoring protocol (behavioural_plan.md) exists but was not run, because the user asked
for short checks; that is a scope decision, not an inability.

## F34 - closing reading of F31 for a concrete patch choice (no new compute except a cost count and a tool-vs-probe check)

Question: does a reduced patch bring the output of layers 0-3 closer to MiMo than a random patch of the same size? Data: stack_L0-3_mimo.json, base embeddings in,
each layer fed by the patched stack's own output, 4 dev prompts and 8 separate held-out prompts, resid = distance to MiMo's output relative to the unpatched base (1.0 = no change).

| patch on the matrices of layers 0-3 | params (of 864.0 M matrix params) | resid dev / held-out | random subspace, same size |
|---|---|---|---|
| rank 16 | 5.0 M (0.6%, ~10 MB bf16) | 0.94 / 0.91 | 0.995 / 0.995 |
| rank 256 | 80.2 M (9.3%, ~160 MB bf16) | 0.70 / 0.66 | 0.945-0.947 / 0.950-0.951 |
| top 10% entries by size | 86.4 M values + indices (~518 MB with int32 indices) | 0.72 / 0.68 | not run |
| all matrices, full delta | 864.0 M (1.7 GB bf16) | 0.13 / 0.12 | - |

Reading: yes, for these four layers, a rank-256 patch gets residual 0.66-0.70 where a random patch of the same size gets 0.95; the effect repeats on held-out prompts.
Entry-thresholding reaches a similar residual at ~3x the stored size. Rank 16 is the most size-efficient but moves the output little.
Recommended patch (local candidate, not a generated file; the ~160 MB of factors is a size estimate from tensor shapes): rank 256 on the matrices, alpha 1.
Criterion: held-out resid <= 0.80 AND at least 3x the (1 - resid) gap of the random control. These thresholds were chosen AFTER seeing the layers 0-3 numbers,
so for layers 0-3 the result is post hoc, not a pre-registered success (0.66 and 0.34 vs 0.05 only show it clears a bar set with the data in view). The criterion is
prospective only for data not yet seen. A failure on other layers would show only that this protocol fails there, not that the patch serves only early layers in general.
Limits: layers 0-3 only (not 4-31), 4 dev + 8 held-out short prompts, MiMo's own trajectory not used (base trajectory in), distance to MiMo's output is not task quality.
Tool link: `apply_selected(rank=k)` agrees with the probe's rank formula on two layer-0 matrices (rank 16 and 256): |difference| <= 3.9e-4 of |delta|, which is the
bf16 rounding of the output tensor (the probe works in float32).

## F35 - choosing the rank-r subspace with activations instead of weights alone (idea taken from areu01or00/qwen0.8B-wsp; layer 0, one matrix, cache only)

Source reviewed (read only, not run): the X post by @TensorSlay (2026-11-08) and the repo it links, github.com/areu01or00/qwen0.8B-wsp (no licence file declared in the
GitHub API; created 2026-11-08). It writes FACTS into Qwen3.5-0.8B `mlp.down_proj` of layers 17-21 by a closed-form MEMIT-style solve from the activations entering
down_proj ("keys"); there is no donor model and no delta transfer, so it is not a method for moving MiMo's changes onto Qwen. Its results are the author's own (2 of 5
unseen wordings word-perfect, others drift or loop; reported in its README); I did not reproduce them.
What transfers to our question: its subspace comes from real activations. Test (`data_aware_rank.py`, data `data_aware_rank_L0_down_proj.json`): for layer 0 `mlp.down_proj`,
patch = U U^T delta with U from (a) the left singular vectors of delta (weights only, our current choice), (b) those of delta @ K_dev^T with K_dev = keys of the 4 dev
prompts (95 tokens, so rank <= 95), (c) random. Scored on the 8 held-out prompts (217 tokens), base trajectory:

| rank | captured weights-only | captured data-aware | random | rel. error weights-only / data-aware |
|---|---|---|---|---|
| 8 | 0.032 | 0.080 | 0.002 | 0.984 / 0.959 |
| 16 | 0.057 | 0.096 | 0.004 | 0.971 / 0.951 |
| 32 | 0.087 | 0.126 | 0.008 | 0.956 / 0.935 |
| 64 | 0.131 | 0.156 | 0.015 | 0.932 / 0.919 |

(captured = <patch K, delta K>/<delta K, delta K> on the held-out keys; local reproduction of the matrix's output difference, not benefit.)
Reading: with the same size, picking U from activations beats picking it from weights on held-out tokens at every rank tried (2.5x at rank 8, shrinking to 1.2x at rank 64);
both are far above random. Limits: one matrix, one layer, 95 fit tokens, our own prompts, base trajectory, no task metric; the gap may change with more fit data,
other matrices or deeper layers, and the dev set cannot support ranks above 95 (rank 256 untested). Not applied to the F34 recommendation.

## F36 - pre-registered test of F35 on all large matrices of layers 0-3: criterion NOT met (25 matrices, ranks 16 and 64, cache only)

Criterion fixed and committed before the first run (d4e911d; `data_aware_stack.py` docstring): data-aware beats weights-only if, at BOTH ranks, it has lower resid in >= 75% of matrices,
the mean resid is lower by >= 0.01, and it beats the random control in >= 90%. Fit on the 4 dev prompts (95 tokens), score on the 8 held-out prompts' keys, base trajectory, per matrix.
After that commit the script only changed in how it computes (resumable per matrix, no patch matrix materialised for memory); the criterion and the metrics were not touched.
Data: `lab/runs/delta_patterns/data_aware_stack_L0-3.json`.

| rank | win rate vs weights-only | mean resid weights / data-aware / random | mean cos weights / data-aware / random | mean norm_ratio weights / data-aware / random | beats random | passes |
|---|---|---|---|---|---|---|
| 16 | 72% (18/25) | 0.905 / 0.853 / 0.998 | 0.395 / 0.469 / 0.055 | 0.395 / 0.469 / 0.055 | 100% | no (win rate 72% < 75%) |
| 64 | 48% (12/25) | 0.809 / 0.802 / 0.992 | 0.564 / 0.557 / 0.113 | 0.564 / 0.557 / 0.113 | 100% | no (win rate 48%, mean gain 0.007 < 0.01) |

Result: not shown. F35's gain on layer 0 down_proj does not generalise as a rule to these matrices at rank 64, and at rank 16 it misses the win-rate bar by 3 points (the mean gain, 0.052, is above 0.01).
Post hoc (looked at after the verdict, not a result of the criterion): by matrix type, data-aware wins in every attention / linear-attention matrix at rank 16 (in_proj_qkv, in_proj_z, out_proj, q/k/v/o: 13/13) and in
10/13 of them at rank 64, while it loses on MLP gate/up at rank 64 (0/8) and is neutral on down_proj. Possible reasons (not tested): 95 fit tokens against 12288 input dimensions, and for gate/up
the weights-only basis is on the input side while the data-aware one is on the output side, so the comparison mixes two parametrisations. A follow-up would have to be a new pre-registered test on fresh prompts.
Both subspace choices beat the random control in every matrix. This is per-matrix reproduction of an output difference, not the composed stack and not task benefit.
Also: the X post's replies beyond the 2 visible without a login (of 16) were not recoverable: the vault has no X login, so that limit stays.

## F37. The "older pre-tokenizer" of F25 comes from transformers 5.6.0-5.17.0, fixed in 5.18.0 (verified, 2026-11-07)
- **Mechanism, read in the code** (`models/qwen2/tokenization_qwen2.py`):
  - Qwen ships `tokenizer_config.json` with `tokenizer_class: Qwen2Tokenizer`, both for Qwen3.5-9B and for
    Qwen3.8-27B.
  - In the affected versions `AutoTokenizer` builds that class. Its `__init__` builds the pre-tokenizer and decoder
    in code, using the Qwen2 regex `\p{L}+`; it takes only vocab and merges from `tokenizer.json`.
  - Qwen3.5's regex `[\p{L}\p{M}]+` is lost, and `save_pretrained` writes the Qwen2 one to disk.
- **Two methods agree.** On a Hindi sentence, `AutoTokenizer` (5.12.1) gives 20 tokens, against 13 with the
  `tokenizers` library. Applying the Qwen2 regex by hand to `tokenizer.json` gives the same 20 ids.
- **Which versions** (minimal repro on Qwen/Qwen3.5-9B from the Hub, run in clean venvs):

  | transformers | tokenizer class used | tokens |
  |---|---|---|
  | 4.57.6, 5.0.0 – 5.5.0 | `Qwen2TokenizerFast` / `TokenizersBackend` (reads tokenizer.json) | 13 (right) |
  | 5.6.0, 5.6.2, 5.12.1, 5.15.1, 5.16.0, 5.16.1, 5.17.0 | `Qwen2Tokenizer` (pipeline from code) | 20 (wrong) |
  | 5.18.0, 5.20.2, 5.21.1, 5.22.0, 5.22.2 | `Qwen3_5Tokenizer` | 13 (right) |

  It is already fixed upstream: no issue to file.
- **Effect.** It changes the split only where combining marks occur: Hindi, Bengali, Thai, and Arabic with
  diacritics. English, code, CJK and Russian are unchanged. The ByteLevel decoder options also change, but the
  decode round trip was identical on 6 samples (English, spaces, Hindi, Thai, code, emoji), so no visible effect.
- **Published models.** MiMo (saved with 5.12.1), Arete-27B (5.12.1) and Angry-Claudius (5.15.0) were saved
  inside the affected range and carry the Qwen2 regex.
  - Fara was saved with 5.6.2, also in the range, but kept the right regex. So `transformers_version` alone does
    not tell whether the tokenizer went through `save_pretrained`; it may have been copied.
  - Whether those models were also *trained* with the wrong split is not known.
- **Fix for users:** transformers >= 5.18, or load `Qwen3_5Tokenizer` explicitly. For models already published
  with the Qwen2 regex, the regex is in their own `tokenizer.json`, so every version reproduces it. Only
  replacing the file restores the base split.
- **Correction (same day):** a first version of this finding said "transformers 5.12+", inferred from the saved
  models, and treated 5.6.2 as safe. Testing the versions directly showed 5.6.0 is the first bad version and
  5.18.0 the first good one.
- **This repo:** the local environment has 5.12.1, which is affected. Our benchmarks use llama.cpp/GGUF (own
  `qwen35` pre-tokenizer) and the math sets are in English, so no result here depends on it. Code that tokenizes
  Qwen3.5 text with `AutoTokenizer` in Hindi/Thai/Bengali/Arabic would be affected.

## F38. Every trained derivative drops the base's MTP head; file-level edits keep it (measured, 2026-11-07)
- Qwen3.5-9B and Qwen3.8-27B ship a multi-token-prediction head (`mtp.*`, 15 tensors, `mtp_num_hidden_layers` 1),
  usable for speculative decoding. Count of `mtp.*` tensors per derivative, read from the safetensors headers:
  - 0 in MiMo (transformers 5.12.1), tmax (4.57), Fara (5.6.2), WeMM (5.2), Angry-Claudius (5.15) and Arete-27B
    (5.12.1): every model that went through a training framework;
  - 15 in Huihui abliterated (a weight edit) and Nyx-RP (a merged LoRA): edits made on the files.
- Reading: the transformers model class does not load `mtp.*`, so `save_pretrained` after training writes a model
  without it. This happens with every version seen (4.57 to 5.15), unlike F37, which depends on the version.
  Consequence: these derivatives lose MTP speculative decoding, and the base's head cannot simply be put back,
  because it was trained against the base trunk.
- Tooling: `model_report` hid `mtp.*` on both sides and did not report the loss. It now lists hidden parts present
  on one side only, and tells a model on another base (Arete-27B on Qwen3.8-27B) that the Qwen3.5-9B atlas does not
  apply by direction. Tests: `TestHiddenParts`, `TestAtlasFromAnotherBase`.

## F39. A small training moves along a narrow lane set by task and data, not by the seed (measured, 2026-11-07, H6)
- Qwen3.5-0.8B, 3×3 products, 500 AdamW steps in 10 chunks, 3 seeds (data orders). Per chunk, ~89-93 % of the
  increment's energy is in 5 % of its directions; consecutive increments overlap ~0.68 (top-16).
- Cumulative change, cross-seed overlap (top-16, median over matrices): 0.72-0.74 at chunk 1, 0.79-0.80 at chunk 10;
  chance 0.016. Read-side matrices (in_proj_qkv, gate, in_proj_z, q, up) overlap more than write-side (out_proj, down,
  o, k, v).
- Weights move most in chunks 1-3, when accuracy does not move; accuracy rises in chunks 4-10 with ever smaller
  increments. Results §40.1.

## F40. Extrapolating the first frames does not continue the training (measured, 2026-11-07, H6 E_k)
- W0 + α·C_k, k ∈ {1, 2}: dev picks α = 1 in all 6 runs; α ≥ 2 collapses accuracy. Recovery −0.05..0.07, no better
  than a norm-matched random change. The early change is in the lane but does not carry the skill. Results §40.2.

## F41. The lane of OTHER runs carries the skill: projecting one seed's change on it keeps all of it (measured, 2026-11-07, H6 stage 1)
- Rank 20 per matrix (2 %) recovers 0.91 of the skill (own SVD) and 0.92 (lane of the other two seeds); rank 51: ~1.0.
- From r = 20 up, the other seeds' lane does as well as or better than the seed's own SVD, with less energy, less
  interference and less forgetting; at r = 102 it beats the full training (R 1.06). What the seeds share is the
  skill; what is seed-specific is mostly noise.
- The norm-matched random control is hard (it projects ΔW itself): M2 beats it by 12-14 points, p < 1e-13, short of
  the pre-registered 20-point margin (H6b fails). Caveat: G2 missed (+23 points, not 30). Results §40.3.

## F42. One gradient cannot replace the training, even inside the lane (measured, 2026-11-07, H6e)
- Summed gradient over 512 examples at M0, rescaled globally to a fraction of a real training's norm: at the smallest
  scale tried every arm is below M0 (0-1.6 %); sign-of-gradient projected on the lane is least bad (12-14 %).
- The lane gives where; the per-direction amounts come from the optimizer over the whole training. Likely cause: a
  raw gradient puts its norm in few matrices/directions, Adam equalises per coordinate. Tested next in H6f.
  Results §40.4.

## F43. Training only 20 directions per matrix learns most of the skill, and a random 20-dim basis does almost as well as the lane (measured, 2026-11-07, H6f)
- ΔW = U·C with U fixed (rank 20 per matrix, 2.1 % of the parameters), AdamW on C, 200 steps: lane of the other seeds
  R 0.76, random orthonormal basis R 0.63; lane vs random McNemar p 0.03 / 0.009 / 0.10 per seed, so the pre-registered
  "beats random in each seed" fails. At 100 steps: 0.57 vs 0.51.
- Neither interferes: 3×2 accuracy rises to 78-86 % (base 74 %, full training 55-60 %); GSM8K NLL unchanged.
- A one-shot shaped update (gradient or sign in the lane, per-matrix norms from other seeds) no longer breaks the
  model, unlike H6e, but teaches little (R ≤ 0.15).
- Reading with F41: the lane matters to carry an existing change, not to learn one; low rank alone removes the
  interference. Results §40.6.

## F44. Coefficient trajectories are nearly straight, yet stretching them does not finish the training (measured, 2026-11-08, H7)
- 9 digit tasks in one shared random rank-20 basis: the first temporal mode holds 94.8-98.8 % of each trajectory's
  energy and cos(C_25, C_200) is 0.66-0.92.
- Learned jumps from step k to 200 (scale, per-kind linear on C_k, C_k/2, Adam m and m/√v, MLP; leave one task out)
  never beat stopping at k: mean R at k = 25 is 0.38 / 0.23 / 0.13 against 0.50 for C_25 itself, and 0/7 wins.
  The dev-chosen extra scale is 0.5-0.75: predictions overshoot.
- Reading: direction is right early, but the accuracy lives in the small part of C_200 off that line; the
  optimizer's late steps are not predictable from the early ones with these features. Results §41.3.

## F45. Different tasks use almost orthogonal coefficients in the same basis (measured, 2026-11-08, H7)
- Cosine between final C of different tasks 0.03-0.19 (mul3x3/mul4x3 0.53); the share per matrix kind is the same in
  all tasks (linear-attention qkv ~24 %, MLP ~47 %).
- Cross-application changes other tasks by −14 to +44 points; the large gains (into rev4, add3x3) look like output
  format, not shared arithmetic. Results §41.2.

## F46. The learned change works only as a whole: neither its early-direction part nor the rest works alone (measured, 2026-11-08, H8-A)
- C_200 split into P (along C_k, oracle scale) and O (off it). At k = 25: R(P) 0.38, R(O) −0.43 (alone it hurts),
  together R = 1. The oracle scale of C_25 equals H7's P_scale (0.38): no rescaling of early coefficients can do
  better. Results §42.1.

## F47. Two learned changes add up; seven do not (measured, 2026-11-08, H8-B)
- Summing the C of two tasks keeps median 0.89 of each one's gain (42 cells); summing seven (λ 0.5 on dev) keeps 0.60,
  with long arithmetic losing most (div6_2 0.12, mul3x3 0.26) and format-like tasks keeping ~1.0. Cosine between
  the C does not predict the loss. No general damage (GSM8K NLL 1.319 vs 1.321). Results §42.2.
- TIES merge of the seven (H8 addendum a): best on dev is no trimming (d 1.0, λ 2), test retention 0.59, same as the
  plain sum (0.60); trimming hurts. Near-orthogonal C leave few sign conflicts to resolve. Results §42.4.

## F48. The 9B base already knows MATH answers; the improvement is closing on time (measured, 2026-11-08, results §43)
- Own Q6_K, MATH-500 n 50: base 20/50, all 30 failures are truncations at 8 192 tokens (16 already had the answer and
  looped on "I will write the response" or format doubts). Forcing `</think>` + "Final answer: \boxed{" gives 27 / 35 /
  41 / 43 of 50 at 512 / 1 024 / 2 048 / 4 096 tokens; MiMo 41/50 at 8 192 (McNemar p 0.69), Jackrong 31/50.
- Second grader 35/50, every difference a fraction written another way (checked by hand).

## F49. A fine-tune's useful change can be cut out and reused as a small LoRA at half scale (measured, 2026-11-08, results §43.4)
- Jackrong's delta as a rank-64 LoRA keeps 99.97 % of it; on the base at scale 0.5, 2 048 tokens, it scores 14/16
  "has the answer, cannot close" MATH items + 19/20 items the base solves at 8 192 (base at 2 048: 1 + 5).
- Rank 16 or MLP-only at ×0.5 does as well (14 + 18); either half of the layers or attention-only fails and damages.
- MiMo's delta is not low-rank (r64 keeps 29 %) and its r64 LoRA does not transfer closing (3/16).
- Prompting ("be concise") or a +4 logit bias on `</think>` does not replace it (3/16, 1/16).
- Rank-16 ×0.5 on GSM8K (2 048 tokens): 18/28 base failures + 20/20 base solves, base 3 + 6. On MiMo it makes MiMo
  close every time without losing answers. The heretic edit (heretic − Jackrong) is a rank-4 LoRA on 30 residual
  writers; stacked on the closing LoRA it costs little (results §43.5). A mean-activation control vector from MiMo
  helps far less (6 → 15 of 36).
## F50. Whether a change is reusable as a LoRA is read off its exact spectrum (measured, 2026-11-08, results §43.6)
- Exact SVD matters: randomized SVD with q = r + 8 missed the rank-16 truncation by 11-16 % on flat-spectrum deltas.
- MiMo (full fine-tune), ||ΔW||²-weighted share kept: r64 11 %, r512 45 %. No LoRA of it (r64 at ×1/1.5/2, MLP r256,
  MLP r512) beats the base on the 16 stuck items (≤ 4/16, base 1, MiMo 12; vs MiMo p ≤ 0.008). Jackrong (r64 keeps
  99.97 %): r16 ×0.5 14/16, vs base p = 0.0002. Below r8 it is lost (r4: 11/36 vs r16 32/36).
- Correction to F49: "r64 keeps 29 %" for MiMo was an unweighted mean; weighted it is 11 %.
- WSPLang op `delta_to_lora` writes the adapter and the energy curve (ranks 1..1024) in one pass.
## F51. A full fine-tune's behaviour can be distilled into a LoRA when its weights cannot (measured, 2026-11-08, results §43.7)
- 93 of MiMo's own closed, correct MATH answers, 1 epoch, r16 MLP LoRA on the frozen 9B base (22 min on a 12 GB GPU
  by layer streaming with a hand-written backward): MATH probe 6 → 16/36 (p = 0.013), GSM8K 9 → 34/48 (p < 0.0001,
  not trained on GSM8K). MiMo's weight-delta LoRAs got ≤ 4/16 on the stuck items; this one 5/16 there, 11/20 on the
  rest. Still below MiMo (31/36) and Jackrong's weight LoRA (32/36) on MATH; on GSM8K not distinguishable from it.
## F52. More teacher answers keep improving the distilled LoRA; the base's own answers do not (measured, 2026-11-09, results §43.8)
- MiMo behaviour distillation with 171 instead of 93 answers: MATH probe 16 → 24/36 (p = 0.039), GSM8K 34 → 41/48
  (p = 0.016); IWIBench reward 0.400 / 0.720, above Jackrong's weight LoRA on GSM8K (0.560-0.600, n.s.), below on MATH.
- Donor-free self-distillation (one greedy round, 39 of 300 base answers closed and correct): not distinguishable from
  the base (MATH 4/2, GSM8K 9/3). The examples are the items it already solved briefly; initial loss 0.06.
## F53. A closed-form (no-gradient) fit of down_proj to a teacher's residual stream transfers some behaviour (measured, 2026-11-09, results §43.8)
- Layer by layer, ridge chosen on held-out sequences, exported as LoRA r256: MATH 6 → 17/36 (p = 0.003), GSM8K
  9 → 25/48 (p = 0.0004) in 19 min, without a gradient; about distillation round 1 on MATH, below it on GSM8K.
- The fit explains ≤ 20 % of the held-out mismatch past layer 9 and its ΔW is not low-rank (r256 keeps 31 %): a
  full fine-tune is not followed by one matrix kind; o_proj / other matrices and higher export rank are untested.
