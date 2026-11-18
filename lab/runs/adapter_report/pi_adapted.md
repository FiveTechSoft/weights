# Adapter report: hf:harness-hub/Qwen3.8-27B-pi-adapted

PEFT LoRA, r = 32, alpha = 64.0, scale 2.0000; targets ['k_proj', 'o_proj', 'gate_proj', 'v_proj', 'up_proj', 'down_proj', 'q_proj'].
Base in the card: `Qwen/Qwen3.8-27B`; compared against: `Qwen/Qwen3.8-27B`.
Adapter: 256 modules, layers [0, 63], 159,383,552 parameters (header count 159,383,552), 637.6 MB; total ||dW||_F 2.968.

## Families

- touched: mlp.down_proj.weight, mlp.gate_proj.weight, mlp.up_proj.weight, self_attn.k_proj.weight, self_attn.o_proj.weight, self_attn.q_proj.weight, self_attn.v_proj.weight
- untouched: input_layernorm.weight, linear_attn.A_log, linear_attn.conv1d.weight, linear_attn.dt_bias, linear_attn.in_proj_a.weight, linear_attn.in_proj_b.weight, linear_attn.in_proj_qkv.weight, linear_attn.in_proj_z.weight, linear_attn.norm.weight, linear_attn.out_proj.weight, lm_head.weight, model.embed_tokens.weight, model.norm.weight, post_attention_layernorm.weight, self_attn.k_norm.weight, self_attn.q_norm.weight
- embeddings touched: False; head touched: False

## Per kind

| kind | modules | layers | rank90 median | median ||dW|| | median rel_delta (block) | direction consistency |
|---|---|---|---|---|---|---|
| mlp.down_proj.weight | 64 | 0-63 | 27.0 | 0.123 | 0.0011 | 0.023 |
| mlp.gate_proj.weight | 64 | 0-63 | 25.5 | 0.223 | 0.0021 | 0.005 |
| mlp.up_proj.weight | 64 | 0-63 | 25.0 | 0.226 | 0.0021 | 0.005 |
| self_attn.k_proj.weight | 16 | 3-63 | 22.5 | 0.055 | 0.0016 | 0.023 |
| self_attn.o_proj.weight | 16 | 3-63 | 24.0 | 0.125 | 0.0013 | 0.026 |
| self_attn.q_proj.weight | 16 | 3-63 | 16.0 | 0.188 | 0.0016 | 0.007 |
| self_attn.v_proj.weight | 16 | 3-63 | 9.0 | 0.056 | 0.0013 | 0.020 |

## Modules (largest ||dW|| first, up to 40)

| tensor | shape | r | ||dW|| (trace) | ||dW|| (svd) | rank90 | rel_delta block |
|---|---|---|---|---|---|---|
| model.layers.36.mlp.up_proj.weight | 17408x5120 | 32 | 0.271 | 0.271 | 19 | 0.0025 |
| model.layers.38.mlp.gate_proj.weight | 17408x5120 | 32 | 0.267 | 0.267 | 20 | 0.0023 |
| model.layers.36.mlp.gate_proj.weight | 17408x5120 | 32 | 0.266 | 0.266 | 22 | 0.0024 |
| model.layers.37.mlp.gate_proj.weight | 17408x5120 | 32 | 0.265 | 0.265 | 23 | 0.0025 |
| model.layers.40.mlp.gate_proj.weight | 17408x5120 | 32 | 0.264 | 0.264 | 22 | 0.0024 |
| model.layers.39.mlp.gate_proj.weight | 17408x5120 | 32 | 0.264 | 0.264 | 24 | 0.0024 |
| model.layers.35.mlp.gate_proj.weight | 17408x5120 | 32 | 0.260 | 0.260 | 21 | 0.0025 |
| model.layers.39.mlp.up_proj.weight | 17408x5120 | 32 | 0.258 | 0.258 | 22 | 0.0024 |
| model.layers.43.mlp.gate_proj.weight | 17408x5120 | 32 | 0.257 | 0.257 | 22 | 0.0024 |
| model.layers.38.mlp.up_proj.weight | 17408x5120 | 32 | 0.256 | 0.256 | 20 | 0.0024 |
| model.layers.48.mlp.gate_proj.weight | 17408x5120 | 32 | 0.255 | 0.255 | 23 | 0.0025 |
| model.layers.45.mlp.up_proj.weight | 17408x5120 | 32 | 0.255 | 0.255 | 22 | 0.0023 |
| model.layers.44.mlp.gate_proj.weight | 17408x5120 | 32 | 0.254 | 0.254 | 22 | 0.0025 |
| model.layers.63.mlp.up_proj.weight | 17408x5120 | 32 | 0.253 | 0.253 | 22 | 0.0021 |
| model.layers.45.mlp.gate_proj.weight | 17408x5120 | 32 | 0.252 | 0.252 | 23 | 0.0024 |
| model.layers.37.mlp.up_proj.weight | 17408x5120 | 32 | 0.252 | 0.252 | 23 | 0.0024 |
| model.layers.47.mlp.up_proj.weight | 17408x5120 | 32 | 0.251 | 0.251 | 21 | 0.0023 |
| model.layers.47.mlp.gate_proj.weight | 17408x5120 | 32 | 0.251 | 0.251 | 23 | 0.0025 |
| model.layers.40.mlp.up_proj.weight | 17408x5120 | 32 | 0.251 | 0.251 | 21 | 0.0022 |
| model.layers.42.mlp.gate_proj.weight | 17408x5120 | 32 | 0.251 | 0.251 | 22 | 0.0024 |
| model.layers.44.mlp.up_proj.weight | 17408x5120 | 32 | 0.250 | 0.250 | 20 | 0.0021 |
| model.layers.49.mlp.gate_proj.weight | 17408x5120 | 32 | 0.250 | 0.250 | 23 | 0.0024 |
| model.layers.46.mlp.gate_proj.weight | 17408x5120 | 32 | 0.249 | 0.249 | 22 | 0.0024 |
| model.layers.50.mlp.gate_proj.weight | 17408x5120 | 32 | 0.249 | 0.249 | 23 | 0.0023 |
| model.layers.41.mlp.up_proj.weight | 17408x5120 | 32 | 0.249 | 0.249 | 23 | 0.0023 |
| model.layers.41.mlp.gate_proj.weight | 17408x5120 | 32 | 0.249 | 0.249 | 23 | 0.0024 |
| model.layers.43.mlp.up_proj.weight | 17408x5120 | 32 | 0.248 | 0.248 | 21 | 0.0022 |
| model.layers.35.mlp.up_proj.weight | 17408x5120 | 32 | 0.248 | 0.248 | 22 | 0.0023 |
| model.layers.49.mlp.up_proj.weight | 17408x5120 | 32 | 0.247 | 0.247 | 23 | 0.0023 |
| model.layers.62.mlp.up_proj.weight | 17408x5120 | 32 | 0.247 | 0.247 | 24 | 0.0020 |
| model.layers.46.mlp.up_proj.weight | 17408x5120 | 32 | 0.246 | 0.246 | 23 | 0.0022 |
| model.layers.48.mlp.up_proj.weight | 17408x5120 | 32 | 0.245 | 0.245 | 23 | 0.0022 |
| model.layers.42.mlp.up_proj.weight | 17408x5120 | 32 | 0.245 | 0.245 | 21 | 0.0023 |
| model.layers.50.mlp.up_proj.weight | 17408x5120 | 32 | 0.244 | 0.244 | 23 | 0.0021 |
| model.layers.51.mlp.gate_proj.weight | 17408x5120 | 32 | 0.243 | 0.243 | 25 | 0.0023 |
| model.layers.33.mlp.gate_proj.weight | 17408x5120 | 32 | 0.242 | 0.242 | 22 | 0.0024 |
| model.layers.63.mlp.gate_proj.weight | 17408x5120 | 32 | 0.241 | 0.241 | 24 | 0.0019 |
| model.layers.34.mlp.up_proj.weight | 17408x5120 | 32 | 0.240 | 0.240 | 20 | 0.0023 |
| model.layers.34.mlp.gate_proj.weight | 17408x5120 | 32 | 0.240 | 0.240 | 21 | 0.0024 |
| model.layers.52.mlp.gate_proj.weight | 17408x5120 | 32 | 0.239 | 0.239 | 26 | 0.0022 |

No claim about behaviour: where and how large the change is, nothing about what it does.
