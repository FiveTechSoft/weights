# Model report: ConceptCountry/qwen-concept-27b

Base: Qwen/Qwen3.8-27B  
Blocks of 128 rows per tensor.

## Card (what the model page says; not verified)

- repo: ConceptCountry/qwen-concept-27b
- base_model: Qwen/Qwen3.8-27B
- license: apache-2.0
- words the page mentions: -

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 0 (compared after rounding to the lower precision)

## How it was changed

- method: **directional_ablation**
- layers 32-48 (9 touched)
- rank of the change in a block: median 1, max 1 (directions for 90 % of the energy)
- ablation directions: global (median |cos| between tensors 0.915)
- touched: mlp.down_proj.weight
- untouched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, linear_attn.out_proj.weight, lm_head.weight, mlp.gate_proj.weight, mlp.up_proj.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_norm.weight, self_attn.q_proj.weight, self_attn.v_proj.weight

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| input_layernorm.weight | 0/64 | 0.0000% | - | - |
| linear_attn.A_log | 0/48 | 0.0000% | - | - |
| linear_attn.conv1d.weight | 0/48 | 0.0000% | - | - |
| linear_attn.dt_bias | 0/48 | 0.0000% | - | - |
| linear_attn.in_proj_a.weight | 0/48 | 0.0000% | - | - |
| linear_attn.in_proj_b.weight | 0/48 | 0.0000% | - | - |
| linear_attn.in_proj_qkv.weight | 0/48 | 0.0000% | - | - |
| linear_attn.in_proj_z.weight | 0/48 | 0.0000% | - | - |
| linear_attn.norm.weight | 0/48 | 0.0000% | - | - |
| linear_attn.out_proj.weight | 0/48 | 0.0000% | - | - |
| lm_head.weight | 0/1 | 0.0000% | - | - |
| mlp.down_proj.weight | 9/64 | 0.0000% | 1 | 32-48 |
| mlp.gate_proj.weight | 0/64 | 0.0000% | - | - |
| mlp.up_proj.weight | 0/64 | 0.0000% | - | - |
| model.embed_tokens.weight | 0/1 | 0.0000% | - | - |
| model.norm.weight | 0/1 | 0.0000% | - | - |
| post_attention_layernorm.weight | 0/64 | 0.0000% | - | - |
| self_attn.k_norm.weight | 0/16 | 0.0000% | - | - |
| self_attn.k_proj.weight | 0/16 | 0.0000% | - | - |
| self_attn.o_proj.weight | 0/16 | 0.0000% | - | - |
| self_attn.q_norm.weight | 0/16 | 0.0000% | - | - |
| self_attn.q_proj.weight | 0/16 | 0.0000% | - | - |
| self_attn.v_proj.weight | 0/16 | 0.0000% | - | - |

"Untouched" means unchanged in the sampled rows; a full `delta_stats` settles it.

## Closest studied models (Spearman of where they change)

- nyx_rp (role-play LoRA (control)): 0.86
- abliterated (abliteration): 0.85
- dsv4_distill (reasoning distillation, LoRA-like): 0.77
- angry_claudius (degradation (control)): 0.43
- wemm_embed (embedding model (control)): 0.26

## Outside the weights (config, tokenizer, chat template)

- different vocabulary: token ids do not line up with the base
- chat template missing in model (may ship inside a GGUF instead)
- files only in base: config.json, generation_config.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, chat_template.jinja; only in model: -; byte-identical: -
- tokenizer: vocab **differs**, merges **differs**, 

- chat template: missing in model

## What is usable, and how

- Decensoring component: rank 1 per tensor, global directions, layers 32-48, residual-writing matrices only. Reproducible with WSPLang `project_out` per layer (directions taken from the model's own delta). It removes safety behaviour: use only where that is acceptable.
- Closest studied model by where it changes: nyx_rp (role-play LoRA (control)), Spearman 0.86.
- The vocabulary differs from the base: weight deltas of embed_tokens/lm_head do not line up row by row, and nothing can be transferred there without a token map.

