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
