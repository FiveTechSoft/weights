# Model report: DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU

Base: Qwen/Qwen3.8-27B  
Blocks of 128 rows per tensor.

## Card (what the model page says; not verified)

- repo: DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU
- base_model: Qwen/Qwen3.8-27B
- license: apache-2.0
- words the page mentions: heretic, uncensored

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 0 (compared after rounding to the lower precision)

## How it was changed

- method: **full_fine_tune**
- layers 0-63 (64 touched)
- rank of the change in a block: median 109.0, max 113 (directions for 90 % of the energy)
- touched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, linear_attn.out_proj.weight, lm_head.weight, mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_norm.weight, self_attn.q_proj.weight, self_attn.v_proj.weight
- untouched: -

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| mlp.gate_proj.weight | 64/64 | 40.0731% | 110.0 | 0-63 |
| mlp.up_proj.weight | 64/64 | 36.5875% | 110.0 | 0-63 |
| mlp.down_proj.weight | 64/64 | 35.5519% | 110.0 | 0-63 |
| self_attn.q_proj.weight | 16/16 | 34.2434% | 102.0 | 3-63 |
| linear_attn.in_proj_z.weight | 48/48 | 33.8363% | 108.0 | 0-62 |
| linear_attn.in_proj_b.weight | 48/48 | 31.7323% | 39.0 | 0-62 |
| linear_attn.in_proj_qkv.weight | 48/48 | 29.9769% | 108.5 | 0-62 |
| model.embed_tokens.weight | 1/1 | 29.5846% | 109 | - |
| linear_attn.in_proj_a.weight | 48/48 | 28.4257% | 39.5 | 0-62 |
| linear_attn.out_proj.weight | 48/48 | 27.9190% | 110.0 | 0-62 |
| self_attn.k_proj.weight | 16/16 | 26.1985% | 104.5 | 3-63 |
| lm_head.weight | 1/1 | 25.6665% | 106 | - |
| self_attn.o_proj.weight | 16/16 | 21.7607% | 108.0 | 3-63 |
| self_attn.v_proj.weight | 16/16 | 18.1604% | 108.0 | 3-63 |
| linear_attn.conv1d.weight | 48/48 | 5.0505% | - | 0-62 |
| input_layernorm.weight | 64/64 | 4.8804% | - | 0-63 |
| self_attn.k_norm.weight | 16/16 | 4.3560% | - | 3-63 |
| self_attn.q_norm.weight | 16/16 | 4.3114% | - | 3-63 |
| post_attention_layernorm.weight | 64/64 | 3.8452% | - | 0-63 |
| model.norm.weight | 1/1 | 2.9493% | - | - |
| linear_attn.norm.weight | 48/48 | 0.6479% | - | 0-62 |
| linear_attn.A_log | 48/48 | 0.4259% | - | 0-62 |
| linear_attn.dt_bias | 48/48 | 0.2680% | - | 0-62 |

"Untouched" means unchanged in the sampled rows; a full `delta_stats` settles it.

## Closest studied models (Spearman of where they change)

- mimo (reasoning distillation, full): 0.96
- official_post (official post-training (Base -> instruct)): 0.94
- fara (agent, full): 0.91
- neohorse (agent, full): 0.91
- tmax (agent RL, full): 0.87

## Outside the weights (config, tokenizer, chat template)

- config changes: eos_token_id, model_name, pad_token_id, text_config.dtype, text_config.torch_dtype, text_config.use_cache, torch_dtype, unsloth_version
- chat template differs (10 lines)
- files only in base: merges.txt; only in model: -; byte-identical: vocab.json
- config: 10 real changes, 2 cosmetic (library version, moved keys, absent vs None)
- tokenizer: vocab same, merges same, normalizer same, pre-tokenizer same, decoder same

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
| hindi | yes | 11 -> 11 |
| bengali | yes | 9 -> 9 |
| thai | yes | 7 -> 7 |
| arabic_diacritics | yes | 12 -> 12 |
| emoji | yes | 10 -> 10 |

- chat template: different

## What is usable, and how

- Diffuse full fine-tune: rank-one ops cannot reproduce it. Candidate for `subspace_transfer` along the residual directions that improved models share (findings F7, F12, F14); a transfer must beat a random subspace of the same size before it counts (plan H4, not yet run).
- Closest studied model by where it changes: mimo (reasoning distillation, full), Spearman 0.96.
- The card mentions decensoring, but no separate ablation component was found in the sampled rows.
- The chat template differs: benchmark each model with its own template, and a transferred delta may need the donor's template to show its behaviour.

