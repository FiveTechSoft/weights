# Model report: hf_cache/sensenova-loop-gguf/SenseNova-Loop-9B-BF16.gguf

Base: hf_cache/qwen35-9b-hf  
Blocks of 128 rows per tensor.

## Compatibility

- only in base: 0; only in model: 0; shape mismatch: 0; stored dtype differs: 129 (compared after rounding to the lower precision)
- not in the text model, only in base: visual (333 tensors), mtp (15 tensors)
- not in the text model, only in model: mhc (96 tensors), other (11 tensors), mtp (4 tensors)

## How it was changed

- method: **full_fine_tune**
- layers 0-31 (32 touched)
- rank of the change in a block: median 108.0, max 112 (directions for 90 % of the energy)
- touched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, linear_attn.out_proj.weight, lm_head.weight, mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_norm.weight, self_attn.q_proj.weight, self_attn.v_proj.weight
- untouched: -

## Where (median relative change per tensor type, in the sampled blocks)

| type | touched / tensors | rel. change | rank90 | layers |
|---|---|---|---|---|
| mlp.gate_proj.weight | 32/32 | 31.8164% | 109.0 | 0-31 |
| mlp.down_proj.weight | 32/32 | 30.5762% | 111.0 | 0-31 |
| mlp.up_proj.weight | 32/32 | 30.2111% | 108.0 | 0-31 |
| linear_attn.in_proj_b.weight | 24/24 | 28.1558% | 27.0 | 0-30 |
| lm_head.weight | 1/1 | 26.6292% | 99 | - |
| linear_attn.in_proj_z.weight | 24/24 | 25.9469% | 108.0 | 0-30 |
| self_attn.q_proj.weight | 8/8 | 25.6500% | 100.5 | 3-31 |
| self_attn.k_proj.weight | 8/8 | 23.6532% | 101.0 | 3-31 |
| linear_attn.in_proj_qkv.weight | 24/24 | 21.8850% | 104.0 | 0-30 |
| linear_attn.in_proj_a.weight | 24/24 | 21.0182% | 27.0 | 0-30 |
| linear_attn.out_proj.weight | 24/24 | 20.9301% | 109.0 | 0-30 |
| model.embed_tokens.weight | 1/1 | 20.2263% | 101 | - |
| self_attn.o_proj.weight | 8/8 | 15.5903% | 108.0 | 3-31 |
| self_attn.v_proj.weight | 8/8 | 14.0184% | 108.0 | 3-31 |
| post_attention_layernorm.weight | 32/32 | 5.5973% | - | 0-31 |
| linear_attn.conv1d.weight | 24/24 | 4.4586% | - | 0-30 |
| input_layernorm.weight | 32/32 | 2.9679% | - | 0-31 |
| self_attn.k_norm.weight | 8/8 | 2.1572% | - | 3-31 |
| self_attn.q_norm.weight | 8/8 | 2.1073% | - | 3-31 |
| model.norm.weight | 1/1 | 0.7791% | - | - |
| linear_attn.norm.weight | 24/24 | 0.3712% | - | 0-30 |
| linear_attn.A_log | 24/24 | 0.1565% | - | 0-30 |
| linear_attn.dt_bias | 19/24 | 0.0174% | - | 1-30 |

"Untouched" means unchanged in the sampled rows; a full `delta_stats` settles it.

## Closest studied models (Spearman of where they change)

- fara (agent, full): 0.98
- official_post (official post-training (Base -> instruct)): 0.97
- neohorse (agent, full): 0.97
- mimo (reasoning distillation, full): 0.96
- tmax (agent RL, full): 0.88

## What is usable, and how

- Diffuse full fine-tune: rank-one ops cannot reproduce it. Candidate for `subspace_transfer` along the residual directions that improved models share (findings F7, F12, F14); a transfer must beat a random subspace of the same size before it counts (plan H4, not yet run).
- Closest studied model by where it changes: fara (agent, full), Spearman 0.98.
- The model drops the base's MTP (multi-token prediction) head: speculative decoding with the model's own MTP layer is not available, and it cannot be restored from the base (it was trained with the base's trunk).
- Parts of the base missing in the model: visual (333). Check the model still does what those parts did (e.g. image input).
- Parts only in the model: mhc (96), other (11), mtp (4).

