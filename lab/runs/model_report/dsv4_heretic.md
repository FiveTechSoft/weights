# Model report: xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic

Base: Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B  
Blocks of 128 rows per tensor.

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 48 (compared after rounding to the lower precision)

## How it was changed

- method: **directional_ablation**
- layers 9-23 (15 touched)
- rank of the change in a block: median 1.0, max 2 (directions for 90 % of the energy)
- ablation directions: per_layer (median |cos| between tensors 0.069)
- touched: linear_attn.out_proj.weight, mlp.down_proj.weight, self_attn.o_proj.weight
- untouched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, lm_head.weight, mlp.gate_proj.weight, mlp.up_proj.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.q_norm.weight, self_attn.q_proj.weight, self_attn.v_proj.weight

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| self_attn.o_proj.weight | 4/8 | 4.4268% | 1.0 | 11-23 |
| input_layernorm.weight | 0/32 | 0.0000% | - | - |
| linear_attn.A_log | 0/24 | 0.0000% | - | - |
| linear_attn.conv1d.weight | 0/24 | 0.0000% | - | - |
| linear_attn.dt_bias | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_a.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_b.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_qkv.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_z.weight | 0/24 | 0.0000% | - | - |
| linear_attn.norm.weight | 0/24 | 0.0000% | - | - |
| linear_attn.out_proj.weight | 11/24 | 0.0000% | 1 | 9-22 |
| lm_head.weight | 0/1 | 0.0000% | - | - |
| mlp.down_proj.weight | 15/32 | 0.0000% | 1 | 9-23 |
| mlp.gate_proj.weight | 0/32 | 0.0000% | - | - |
| mlp.up_proj.weight | 0/32 | 0.0000% | - | - |
| model.embed_tokens.weight | 0/1 | 0.0000% | - | - |
| model.norm.weight | 0/1 | 0.0000% | - | - |
| post_attention_layernorm.weight | 0/32 | 0.0000% | - | - |
| self_attn.k_norm.weight | 0/8 | 0.0000% | - | - |
| self_attn.k_proj.weight | 0/8 | 0.0000% | - | - |
| self_attn.q_norm.weight | 0/8 | 0.0000% | - | - |
| self_attn.q_proj.weight | 0/8 | 0.0000% | - | - |
| self_attn.v_proj.weight | 0/8 | 0.0000% | - | - |
