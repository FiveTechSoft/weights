# Model report: hf_cache/mimo-distill-hf

Base: C:\weights\hf_cache\qwen35-9b-hf  
Blocks of 128 rows per tensor.

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 48 (compared after rounding to the lower precision)

## How it was changed

- method: **full_fine_tune**
- layers 0-31 (32 touched)
- rank of the change in a block: median 104.0, max 111 (directions for 90 % of the energy)
- touched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, linear_attn.out_proj.weight, lm_head.weight, mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_norm.weight, self_attn.q_proj.weight, self_attn.v_proj.weight
- untouched: -

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| linear_attn.in_proj_b.weight | 24/24 | 12.7159% | 27.0 | 0-30 |
| mlp.gate_proj.weight | 32/32 | 12.5166% | 106.0 | 0-31 |
| mlp.up_proj.weight | 32/32 | 11.1651% | 105.0 | 0-31 |
| mlp.down_proj.weight | 32/32 | 10.7785% | 109.0 | 0-31 |
| self_attn.q_proj.weight | 8/8 | 9.7134% | 95.5 | 3-31 |
| linear_attn.in_proj_z.weight | 24/24 | 9.5847% | 104.0 | 0-30 |
| self_attn.k_proj.weight | 8/8 | 9.4063% | 93.5 | 3-31 |
| linear_attn.in_proj_a.weight | 24/24 | 8.4747% | 26.0 | 0-30 |
| linear_attn.in_proj_qkv.weight | 24/24 | 7.7150% | 97.0 | 0-30 |
| linear_attn.out_proj.weight | 24/24 | 7.4619% | 107.0 | 0-30 |
| lm_head.weight | 1/1 | 6.0735% | 72 | - |
| self_attn.o_proj.weight | 8/8 | 5.5828% | 105.0 | 3-31 |
| model.embed_tokens.weight | 1/1 | 5.3236% | 75 | - |
| self_attn.v_proj.weight | 8/8 | 4.6422% | 102.5 | 3-31 |
| linear_attn.conv1d.weight | 24/24 | 1.7015% | - | 0-30 |
| self_attn.q_norm.weight | 8/8 | 1.2270% | - | 3-31 |
| self_attn.k_norm.weight | 8/8 | 1.1779% | - | 3-31 |
| post_attention_layernorm.weight | 32/32 | 0.8512% | - | 0-31 |
| input_layernorm.weight | 32/32 | 0.8377% | - | 0-31 |
| model.norm.weight | 1/1 | 0.2053% | - | - |
| linear_attn.norm.weight | 24/24 | 0.1516% | - | 0-30 |
| linear_attn.A_log | 14/24 | 0.0120% | - | 1-30 |
| linear_attn.dt_bias | 11/24 | 0.0000% | - | 5-30 |
