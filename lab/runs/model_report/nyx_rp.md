# Model report: Indexnusrefather/Nyx-RP-9B-Instruct-2608-v1

Base: C:\weights\hf_cache\qwen35-9b-hf  
Blocks of 128 rows per tensor.

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 0 (compared after rounding to the lower precision)

## How it was changed

- method: **lora_like**
- layers 0-31 (32 touched)
- rank of the change in a block: median 33.0, max 41 (directions for 90 % of the energy)
- touched: mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_proj.weight, self_attn.v_proj.weight
- untouched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, linear_attn.out_proj.weight, lm_head.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.q_norm.weight

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| mlp.gate_proj.weight | 32/32 | 1.1714% | 33.0 | 0-31 |
| mlp.up_proj.weight | 32/32 | 1.1684% | 32.0 | 0-31 |
| self_attn.k_proj.weight | 8/8 | 0.9577% | 29.0 | 3-31 |
| self_attn.q_proj.weight | 8/8 | 0.8741% | 26.0 | 3-31 |
| self_attn.o_proj.weight | 8/8 | 0.8146% | 25.5 | 3-31 |
| self_attn.v_proj.weight | 8/8 | 0.7119% | 18.0 | 3-31 |
| mlp.down_proj.weight | 32/32 | 0.6903% | 37.0 | 0-31 |
| input_layernorm.weight | 0/32 | 0.0000% | - | - |
| linear_attn.A_log | 0/24 | 0.0000% | - | - |
| linear_attn.conv1d.weight | 0/24 | 0.0000% | - | - |
| linear_attn.dt_bias | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_a.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_b.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_qkv.weight | 0/24 | 0.0000% | - | - |
| linear_attn.in_proj_z.weight | 0/24 | 0.0000% | - | - |
| linear_attn.norm.weight | 0/24 | 0.0000% | - | - |
| linear_attn.out_proj.weight | 0/24 | 0.0000% | - | - |
| lm_head.weight | 0/1 | 0.0000% | - | - |
| model.embed_tokens.weight | 0/1 | 0.0000% | - | - |
| model.norm.weight | 0/1 | 0.0000% | - | - |
| post_attention_layernorm.weight | 0/32 | 0.0000% | - | - |
| self_attn.k_norm.weight | 0/8 | 0.0000% | - | - |
| self_attn.q_norm.weight | 0/8 | 0.0000% | - | - |
