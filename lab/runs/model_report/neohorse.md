# Model report: hf_cache/neohorse-1-9b-hf

Base: C:\weights\hf_cache\qwen35-9b-hf  
Blocks of 128 rows per tensor.

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 48 (compared after rounding to the lower precision)

## How it was changed

- method: **full_fine_tune**
- layers 0-31 (32 touched)
- rank of the change in a block: median 106.0, max 111 (directions for 90 % of the energy)
- touched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.out_proj.weight, lm_head.weight, mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_norm.weight, self_attn.q_proj.weight, self_attn.v_proj.weight
- untouched: linear_attn.norm.weight

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| lm_head.weight | 1/1 | 0.6800% | 55 | - |
| linear_attn.in_proj_b.weight | 24/24 | 0.6745% | 28.0 | 0-30 |
| mlp.down_proj.weight | 32/32 | 0.6489% | 110.0 | 0-31 |
| mlp.up_proj.weight | 32/32 | 0.6401% | 107.0 | 0-31 |
| mlp.gate_proj.weight | 32/32 | 0.6061% | 107.0 | 0-31 |
| self_attn.k_proj.weight | 8/8 | 0.5645% | 92.0 | 3-31 |
| self_attn.q_proj.weight | 8/8 | 0.5162% | 95.0 | 3-31 |
| linear_attn.in_proj_z.weight | 24/24 | 0.4779% | 106.0 | 0-30 |
| linear_attn.out_proj.weight | 24/24 | 0.4578% | 107.0 | 0-30 |
| linear_attn.in_proj_qkv.weight | 24/24 | 0.4308% | 99.0 | 0-30 |
| self_attn.o_proj.weight | 8/8 | 0.4290% | 102.0 | 3-31 |
| linear_attn.in_proj_a.weight | 24/24 | 0.4152% | 26.0 | 0-30 |
| self_attn.v_proj.weight | 8/8 | 0.3647% | 101.0 | 3-31 |
| model.embed_tokens.weight | 1/1 | 0.1640% | 37 | - |
| linear_attn.conv1d.weight | 24/24 | 0.1110% | - | 0-30 |
| post_attention_layernorm.weight | 31/32 | 0.0275% | - | 0-31 |
| input_layernorm.weight | 32/32 | 0.0208% | - | 0-31 |
| self_attn.k_norm.weight | 5/8 | 0.0003% | - | 3-31 |
| model.norm.weight | 1/1 | 0.0001% | - | - |
| linear_attn.A_log | 5/24 | 0.0000% | - | 6-14 |
| linear_attn.dt_bias | 1/24 | 0.0000% | - | 22-22 |
| linear_attn.norm.weight | 0/24 | 0.0000% | - | - |
| self_attn.q_norm.weight | 1/8 | 0.0000% | - | 3-3 |
