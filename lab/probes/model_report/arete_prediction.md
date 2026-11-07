# Arete-27B vs Qwen3.8-27B: prediction written before the report (2026-11-07)

Model: `NousResearch/Arete-27B`. Base: `Qwen/Qwen3.8-27B`. Recipe from the card:
- SFT only (no RL, no architecture change), teacher Inkling;
- 52.8M tokens, 2 epochs, 1,202 steps, batch 48, 64k context;
- AdamW, lr 1e-6 → 1e-5 warmup then cosine to 1e-6, weight decay 0.1;
- text-only data.

This is the first model in our set whose training recipe is fully published. It works as known truth for the
classifier and as a second "full fine-tune with a non-capability goal" (after WeMM, F22) on another base and scale.

Predictions, fixed before running `report_arete.wspl`:
1. **Method:** `full_fine_tune`, not `lora_like` (the card says full SFT; nothing suggests adapters).
2. **Size:** global relative change between 0.1 % and 2 %. Reference points: tmax RL 0.07 %, NeoHorse SFT 0.5 %,
   MiMo distill 9.2 %. Low lr, few steps.
3. **Where:** MLP (gate/up/down) and DeltaNet β (`in_proj_b`) among the most changed types; embeddings and
   `lm_head` among the least; norms, `A_log`, `dt_bias` unchanged or nearly so (F22 pattern).
4. **Vision tower:** unchanged (text-only training). Hidden from the text report; if visible, identical.
5. **Outside the weights:** vocab and merges identical; config differences cosmetic only. Chat template: no
   prediction (they ship `chat_template.jinja`; 50/50).

A wrong prediction is recorded as wrong, not re-explained.

## Outcome (report `lab/runs/model_report/arete.md`, run right after committing the above)

1. Method `full_fine_tune`: **right**.
2. Size: **right** for what block mode measures. Per-type median change of the matrices is 0.32-0.84 %. The global
   figure needs a full `delta_stats`, which was not run.
3. Where: **partly wrong**.
   - Right: DeltaNet β is the most changed type (0.84 %), MLP down/up/gate are high (0.59-0.73 %), and embeddings
     and `lm_head` are the least changed matrices (0.05-0.08 %).
   - Wrong: the layer norms were trained. `input_layernorm` and `post_attention_layernorm` are touched in 63-64 of
     64 layers (0.06 %), and `A_log` / `dt_bias` in 21 and 10 of 48 (tiny). Only `linear_attn.norm` is unchanged.
     This differs from the WeMM full fine-tune, where all norms were untouched (F22).
   - Not predicted: `conv1d` is third (0.70 %).
4. Vision tower: not checked; the report hides it. Not predicted: the **MTP head was removed**
   (`mtp_num_hidden_layers` 1 → 0, 15 tensors gone). The report missed this, because `text_model_name` hides
   `mtp.*` on both sides.
5. Outside the weights:
   - Right: vocab and merges are identical, and the chat template is byte-identical.
   - **Wrong** on the config: it has 2 real changes, MTP removed and `vision_config.model_type` renamed.
   - Not predicted: the pre-tokenizer and decoder differ, the same as MiMo (F25). Cause found in findings F27.

Report defects to fix:
- (a) It must list hidden tensors that exist only in the base (the MTP head).
- (b) The atlas sections ("closest studied models", "shared directions") compare against the Qwen3.5-9B atlas.
  For another base they should be skipped or labelled as cross-base. Here the "shared directions" table is empty
  and malformed.
