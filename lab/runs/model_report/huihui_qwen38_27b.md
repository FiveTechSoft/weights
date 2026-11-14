# Model report: huihui-ai/Huihui-Qwen3.8-27B-abliterated

Base: Qwen/Qwen3.8-27B  
Blocks of 128 rows per tensor.

## Card (what the model page says; not verified)

- repo: huihui-ai/Huihui-Qwen3.8-27B-abliterated
- base_model: Qwen/Qwen3.8-27B
- license: apache-2.0
- words the page mentions: abliteration, lora, uncensored

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 0 (compared after rounding to the lower precision)

## How it was changed

- method: **directional_ablation**
- layers 17-51 (35 touched)
- rank of the change in a block: median 1.0, max 1 (directions for 90 % of the energy)
- ablation directions: global (median |cos| between tensors 1.000)
- touched: linear_attn.out_proj.weight, mlp.down_proj.weight, self_attn.o_proj.weight
- untouched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, lm_head.weight, mlp.gate_proj.weight, mlp.up_proj.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.q_norm.weight, self_attn.q_proj.weight, self_attn.v_proj.weight

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| mlp.down_proj.weight | 35/64 | 2.0903% | 1 | 17-51 |
| linear_attn.out_proj.weight | 26/48 | 2.0618% | 1.0 | 17-50 |
| self_attn.o_proj.weight | 9/16 | 2.0610% | 1 | 19-51 |
| input_layernorm.weight | 0/64 | 0.0000% | - | - |
| linear_attn.A_log | 0/48 | 0.0000% | - | - |
| linear_attn.conv1d.weight | 0/48 | 0.0000% | - | - |
| linear_attn.dt_bias | 0/48 | 0.0000% | - | - |
| linear_attn.in_proj_a.weight | 0/48 | 0.0000% | - | - |
| linear_attn.in_proj_b.weight | 0/48 | 0.0000% | - | - |
| linear_attn.in_proj_qkv.weight | 0/48 | 0.0000% | - | - |
| linear_attn.in_proj_z.weight | 0/48 | 0.0000% | - | - |
| linear_attn.norm.weight | 0/48 | 0.0000% | - | - |
| lm_head.weight | 0/1 | 0.0000% | - | - |
| mlp.gate_proj.weight | 0/64 | 0.0000% | - | - |
| mlp.up_proj.weight | 0/64 | 0.0000% | - | - |
| model.embed_tokens.weight | 0/1 | 0.0000% | - | - |
| model.norm.weight | 0/1 | 0.0000% | - | - |
| post_attention_layernorm.weight | 0/64 | 0.0000% | - | - |
| self_attn.k_norm.weight | 0/16 | 0.0000% | - | - |
| self_attn.k_proj.weight | 0/16 | 0.0000% | - | - |
| self_attn.q_norm.weight | 0/16 | 0.0000% | - | - |
| self_attn.q_proj.weight | 0/16 | 0.0000% | - | - |
| self_attn.v_proj.weight | 0/16 | 0.0000% | - | - |

"Untouched" means unchanged in the sampled rows; a full `delta_stats` settles it.

## Closest studied models (Spearman of where they change)

The studied models are on another base (other tensor shapes): only the per-type profile is compared.

- abliterated (abliteration): 1.00
- dsv4_distill (reasoning distillation, LoRA-like): 0.84
- nyx_rp (role-play LoRA (control)): 0.78
- angry_claudius (degradation (control)): 0.55
- tmax (agent RL, full): 0.42

## Shared directions with studied improvements

- not applicable: the studied models' signatures have other tensor shapes (another base).

## Outside the weights (config, tokenizer, chat template)

- chat template differs (10 lines)
- files only in base: -; only in model: -; byte-identical: config.json, generation_config.json, tokenizer.json, vocab.json, merges.txt
- config: 0 real changes, 0 cosmetic (library version, moved keys, absent vs None)
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

- Decensoring component: rank 1 per tensor, global directions, layers 17-51, residual-writing matrices only. Reproducible with WSPLang `project_out` per layer (directions taken from the model's own delta). It removes safety behaviour: use only where that is acceptable.
- Shared directions not tested: the studied models are on another base. Direction comparisons need models fine-tuned from this model's base.
- Closest studied model by where it changes: abliterated (abliteration), Spearman 1.00 (studied models are on another base: only the per-type profile is compared).
- The chat template differs: benchmark each model with its own template, and a transferred delta may need the donor's template to show its behaviour.

