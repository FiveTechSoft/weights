# Model report: NousResearch/Arete-27B

Base: Qwen/Qwen3.8-27B  
Blocks of 128 rows per tensor.

## Card (what the model page says; not verified)

- repo: NousResearch/Arete-27B
- base_model: Qwen/Qwen3.8-27B
- license: apache-2.0
- words the page mentions: distillation, rl, sft

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 0 (compared after rounding to the lower precision)
- not in the text model, only in base: mtp (15 tensors)

## How it was changed

- method: **full_fine_tune**
- layers 0-63 (64 touched)
- rank of the change in a block: median 91.0, max 111 (directions for 90 % of the energy)
- touched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.out_proj.weight, lm_head.weight, mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_norm.weight, self_attn.q_proj.weight, self_attn.v_proj.weight
- untouched: linear_attn.norm.weight

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| linear_attn.in_proj_b.weight | 48/48 | 0.8369% | 36.0 | 0-62 |
| mlp.down_proj.weight | 64/64 | 0.7304% | 110.0 | 0-63 |
| linear_attn.conv1d.weight | 48/48 | 0.7036% | - | 0-62 |
| self_attn.k_proj.weight | 16/16 | 0.6808% | 88.0 | 3-63 |
| mlp.up_proj.weight | 64/64 | 0.6683% | 101.5 | 0-63 |
| mlp.gate_proj.weight | 64/64 | 0.5949% | 92.5 | 0-63 |
| self_attn.v_proj.weight | 16/16 | 0.5610% | 97.5 | 3-63 |
| self_attn.o_proj.weight | 16/16 | 0.4486% | 102.5 | 3-63 |
| linear_attn.out_proj.weight | 48/48 | 0.4098% | 107.0 | 0-62 |
| linear_attn.in_proj_a.weight | 48/48 | 0.4015% | 29.0 | 0-62 |
| linear_attn.in_proj_qkv.weight | 48/48 | 0.3582% | 82.0 | 0-62 |
| self_attn.q_proj.weight | 16/16 | 0.3298% | 72.5 | 3-63 |
| linear_attn.in_proj_z.weight | 48/48 | 0.3228% | 72.5 | 0-62 |
| lm_head.weight | 1/1 | 0.0755% | 7 | - |
| post_attention_layernorm.weight | 63/64 | 0.0625% | - | 0-63 |
| input_layernorm.weight | 64/64 | 0.0604% | - | 0-63 |
| model.embed_tokens.weight | 1/1 | 0.0528% | 3 | - |
| self_attn.k_norm.weight | 15/16 | 0.0050% | - | 3-63 |
| self_attn.q_norm.weight | 9/16 | 0.0025% | - | 3-59 |
| model.norm.weight | 1/1 | 0.0004% | - | - |
| linear_attn.A_log | 21/48 | 0.0000% | - | 5-50 |
| linear_attn.dt_bias | 10/48 | 0.0000% | - | 5-61 |
| linear_attn.norm.weight | 0/48 | 0.0000% | - | - |

"Untouched" means unchanged in the sampled rows; a full `delta_stats` settles it.

## Closest studied models (Spearman of where they change)

The studied models are on another base (other tensor shapes): only the per-type profile is compared.

- tmax (agent RL, full): 0.91
- neohorse (agent, full): 0.83
- fara (agent, full): 0.82
- official_post (official post-training (Base -> instruct)): 0.81
- mimo (reasoning distillation, full): 0.79

## Shared directions with studied improvements

- not applicable: the studied models' signatures have other tensor shapes (another base).

## Outside the weights (config, tokenizer, chat template)

- config changes: text_config.mtp_num_hidden_layers, vision_config.model_type
- pre-tokenizer differs from the base
- text encodes differently in: hindi (11 -> 19 tokens), bengali (9 -> 16 tokens), thai (7 -> 15 tokens), arabic_diacritics (12 -> 15 tokens)
- files only in base: -; only in model: -; byte-identical: vocab.json, merges.txt, chat_template.jinja
- config: 2 real changes, 2 cosmetic (library version, moved keys, absent vs None)
- tokenizer: vocab same, merges same, normalizer same, pre-tokenizer **differs**, decoder **differs**

| script | same ids | tokens base -> model |
|---|---|---|
| english | yes | 13 -> 13 |
| spanish | yes | 20 -> 20 |
| code | yes | 19 -> 19 |
| math | yes | 34 -> 34 |
| chinese | yes | 6 -> 6 |
| japanese | yes | 11 -> 11 |
| korean | yes | 10 -> 10 |
| russian | yes | 13 -> 13 |
| vietnamese | yes | 8 -> 8 |
| hindi | **no** | 11 -> 19 |
| bengali | **no** | 9 -> 16 |
| thai | **no** | 7 -> 15 |
| arabic_diacritics | **no** | 12 -> 15 |
| emoji | yes | 10 -> 10 |

- chat template: same

## What is usable, and how

- Diffuse full fine-tune: rank-one ops cannot reproduce it. Candidate for `subspace_transfer` along the residual directions that improved models share (findings F7, F12, F14); a transfer must beat a random subspace of the same size before it counts (plan H4, not yet run).
- Shared directions not tested: the studied models are on another base. Direction comparisons need models fine-tuned from this model's base.
- Closest studied model by where it changes: tmax (agent RL, full), Spearman 0.91 (studied models are on another base: only the per-type profile is compared).
- The model drops the base's MTP (multi-token prediction) head: speculative decoding with the model's own MTP layer is not available, and it cannot be restored from the base (it was trained with the base's trunk).
- Text encodes differently from the base in hindi, bengali, thai, arabic_diacritics. Use one tokenizer for both models when comparing or transferring, and check which one the model was trained with (perplexity under each) before trusting results in those scripts.

