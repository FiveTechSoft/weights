# Model report: Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B

Base: unsloth/Qwen3.5-9B  
Blocks of 128 rows per tensor.

## Card

- repo: Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B
- base_model: unsloth/Qwen3.5-9B
- license: apache-2.0
- flags: ['distillation', 'merge', 'rl', 'sft']

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 0 (compared after rounding to the lower precision)

## How it was changed

- method: **lora_like**
- layers 0-31 (32 touched)
- rank of the change in a block: median 35.0, max 39 (directions for 90 % of the energy)
- touched: linear_attn.out_proj.weight, mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_proj.weight, self_attn.v_proj.weight
- untouched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, lm_head.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.q_norm.weight

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| mlp.gate_proj.weight | 32/32 | 9.9439% | 35.0 | 0-31 |
| mlp.up_proj.weight | 32/32 | 9.0610% | 36.5 | 0-31 |
| self_attn.q_proj.weight | 8/8 | 7.1672% | 28.5 | 3-31 |
| self_attn.k_proj.weight | 8/8 | 7.0872% | 31.0 | 3-31 |
| linear_attn.out_proj.weight | 24/24 | 6.8684% | 34.0 | 0-30 |
| mlp.down_proj.weight | 32/32 | 6.7067% | 35.5 | 0-31 |
| self_attn.o_proj.weight | 8/8 | 4.6961% | 36.5 | 3-31 |
| self_attn.v_proj.weight | 8/8 | 4.6345% | 22.5 | 3-31 |
| input_layernorm.weight | 0/32 | 0.0000% | - | - |
| linear_attn.A_log | 0/24 | 0.0000% | - | - |
| linear_attn.conv1d.weight | 0/24 | 0.0000% | - | - |
| linear_attn.dt_bias | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_a.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_b.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_qkv.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_z.weight | 0/24 | 0.0000% | - | - |
| linear_attn.norm.weight | 0/24 | 0.0000% | - | - |
| lm_head.weight | 0/1 | 0.0000% | - | - |
| model.embed_tokens.weight | 0/1 | 0.0000% | - | - |
| model.norm.weight | 0/1 | 0.0000% | - | - |
| post_attention_layernorm.weight | 0/32 | 0.0000% | - | - |
| self_attn.k_norm.weight | 0/8 | 0.0000% | - | - |
| self_attn.q_norm.weight | 0/8 | 0.0000% | - | - |

"Untouched" means unchanged in the sampled rows; a full `delta_stats` settles it.

## Closest studied models (Spearman of where they change)

- nyx_rp (role-play LoRA (control)): 0.95
- abliterated (abliteration): 0.83
- neohorse (agent, full): 0.53
- tmax (agent RL, full): 0.52
- mimo (reasoning distillation, full): 0.50

## What is usable, and how

- Adapter-like update (rank about 39) on: linear_attn.out_proj.weight, mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_proj.weight, self_attn.v_proj.weight. Small and self-contained: it can be applied to the base as a low-rank delta; its effect is whatever the adapter was trained for, nothing more.
- Closest studied model by where it changes: nyx_rp (role-play LoRA (control)), Spearman 0.95.

