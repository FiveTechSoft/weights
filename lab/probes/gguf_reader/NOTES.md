# `engine/gguf_reader.py` — a GGUF as a `model_report` reader (roadmap gap 29b) — built 2026-11-18, tests first

**What it is.** `LocalGGUF(path)` gives the `keys / shape / get / get_rows / close` interface of
`engine/weight_diff.py` over a local GGUF (torch-order shapes; BF16 / F16 / F32 in the file's dtype; quantised types
dequantised by gguf-py's `gguf.quants.dequantize`, row blocks only). `open_gguf_as_hf(path)` is the same file seen
with HF names and layout, so it can be the `model` side of `model_report` against a safetensors base. Tests:
`tests/test_gguf_reader.py`, 14 tests, all green (`PYTHONUTF8=1 python -m unittest tests.test_gguf_reader`, ≈ 20 s;
the three `RealA0Case` tests skip when `hf_cache/h4/A0-Q6_K.gguf` or `hf_cache/qwen35-9b-hf` is absent).

## What llama.cpp's conversion does to Qwen3.5 that the HF view undoes (all found on A0 vs the HF 9B, not from memory)

| stored in the GGUF | HF | how found |
|---|---|---|
| `blk.N.attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`, `output_norm` = **w + 1** | `input_layernorm` … = w | GGUF − HF = 1.000 exactly on every one; `ssm_norm` is stored as is. Qwen3.5's RMSNorm scales by (1 + w). Applied when `general.architecture` starts with `qwen35`, `qwen3next`, `gemma`, `sensenova_loop`; `open_gguf_as_hf(path, norm_offset=True/False)` overrides |
| `ssm_a` = **−exp(A_log)** | `linear_attn.A_log` | exact after the head permutation below |
| `ssm_conv1d.weight` (C, K) | `conv1d.weight` (C, 1, K) | shape |
| **value heads interleaved**: GGUF head n = HF head `(n % n_k)·(n_v/n_k) + n // n_k` (n_v 32, n_k 16: 0, 2, 4, … 30, 1, 3, …) on `ssm_a`, `ssm_dt.bias`, `ssm_alpha` / `ssm_beta` rows, `attn_gate` rows (blocks of head_dim), the v rows of `attn_qkv` and `ssm_conv1d`, the **columns** of `ssm_out` | HF order | derived from `ssm_a` vs `−exp(A_log)` (plain cos of z / out / v was 0.06-0.07; after the permutation every tensor is exact or cos 0.9998). q and k rows are not permuted |
| `blk.32.nextn.*` + a full attention block 32 | `mtp.*` | hidden on both sides (text_model_name hides `mtp.`) |
| geometry | — | n_v = numel(`ssm_a`), head_dim = rows(`attn_gate`) / n_v (or numel(`ssm_norm`)), n_k from the `attn_qkv` rows; per block, no config read |

Name map (`gguf_to_hf_name`): `token_embd` → `model.embed_tokens`, `output` → `lm_head`, `output_norm` → `model.norm`,
`attn_q/k/v/output` → `self_attn.q/k/v/o_proj`, `attn_q_norm/k_norm`, `ffn_gate/up/down` → `mlp.*_proj`,
`attn_norm` → `input_layernorm`, `post_attention_norm` / `ffn_norm` → `post_attention_layernorm`,
`attn_qkv` → `linear_attn.in_proj_qkv`, `attn_gate` → `in_proj_z`, `ssm_alpha` → `in_proj_a`, `ssm_beta` → `in_proj_b`,
`ssm_out` → `out_proj`, `ssm_conv1d` → `conv1d`, `ssm_dt.bias` → `dt_bias`, `ssm_a` → `A_log`, `ssm_norm` → `norm`.
Unknown names → hidden (`hidden()` lists them with their stored names).

**Real-file numbers (A0 Q6_K vs HF bf16):** all 427 text-model tensors map to an HF tensor of the same shape, 0
unmatched, 15 hidden (the MTP block); F32 tensors equal after bf16 rounding (max abs error 5.96e-08 over 11
tensors, including A_log, dt_bias, conv1d, the five norm families); `blk.3.attn_q` dequantised vs HF `q_proj`
cos 0.999838, rel error 0.0180 (the Q6_K floor); `in_proj_a / b / z / qkv / out_proj` cos 0.99983-0.99984 after the
head permutation. Through `model_report.fingerprint` (layers 0, 3): every F32 tensor `changed 0 / rel 0`, every
quantised matrix `rel 0.018`, so **a quantised GGUF against its own base reads as a diffuse 1.8 % change**: compare
BF16 GGUFs, or read `rel_delta` with that floor in mind.

## SenseNova-Loop-9B's GGUF (their fork's converter; `hf_cache/sensenova-loop-gguf/SenseNova-Loop-9B-BF16.gguf`, 18.4 GB, finished 22:53)

`general.architecture = sensenova_loop`, 490 tensors (280 F32, 210 BF16), blocks **0-27 and 52-55** (execution index:
layers 4-27 run twice, so the last four HF layers are 52-55) plus `blk.56` = nextn. The view compacts the non-MTP
blocks in order → `model.layers.0-31` (`layer_map`). Two pairs are **fused**: `attn_qkv_gate.weight` (12288, 4096) =
[`in_proj_qkv` 8192 rows ; `in_proj_z` 4096 rows] and `ssm_beta_alpha.weight` (64, 4096) = [`in_proj_b` ; `in_proj_a`]
(order checked by cosine: rows 0-31 vs `in_proj_b` 0.914 / vs `in_proj_a` 0.063; rows 32-63 vs `in_proj_a` 0.991);
the view splits them and applies the same head permutation (plain `z` cos 0.0, permuted 0.98). Norms are stored
w + 1 like upstream (attn_norm − (HF + 1) max 0.016). New tensors with no HF counterpart: `hc_attn_fn.{weight (8,
8192), bias (8)}` and `hc_ffn_fn.{weight, bias}` on blocks 4-27 only (96 tensors: the mHC routing of the looped block,
FP32 as the card says) → `hidden()`, **not** `only_in_model` (model_report never sees them; list `hidden()` in the
report). Coverage: 427 view keys = 427 HF text keys, 0 only-in-either, 0 shape mismatches.

First numbers through the view (block reads, rows 0-4095; `model_report.fingerprint` on layers 0, 3, 31, 128 rows):
`dt_bias` **identical** to Qwen3.5-9B (rel 0.0000-0.0002, every layer checked), `A_log` rel 0.0009, `linear_attn.norm`
rel 0.0065, `model.norm` 0.0078, norms 0.02-0.07; matrices rel 0.11-0.44 (q_proj 0.19-0.28, MLP 0.25-0.33, `in_proj_b`
0.44, `embed_tokens` 0.23, cos 0.95-0.99); `classify` → `full_fine_tune`. Reading for the parent: **it starts from
Qwen3.5-9B** (the frozen decay parameters and A_log are the base's to 3-4 digits; a fresh pretrain would not share
them) **and was then trained heavily everywhere** (every matrix moved 10-40 % in norm, all layers, all families), with
the mHC routing added on the looped layers.

## Patch for `engine/model_report.py` (I did not touch it)

```python
def reader_for(spec: str, root=None):
    """A local directory/file of safetensors, a local .gguf, or a Hugging Face repo id read by HTTP ranges."""
    from pathlib import Path
    p = Path(spec)
    if not p.exists() and root is not None and (Path(root) / spec).exists():
        p = Path(root) / spec
    if p.exists() and p.is_file() and p.suffix.lower() == ".gguf":
        from engine.gguf_reader import open_gguf_as_hf
        return open_gguf_as_hf(p)                     # already HF-named and hides mtp.*; has .hidden() like Renamed
    if p.exists():
        return wd.Renamed(wd.LocalSafetensors(p), wd.text_model_name)
    return wd.Renamed(wd.RangeSafetensors(wd.hf_files(spec), wd.hf_fetch(spec)), wd.text_model_name)
```

Caveats for the integrator: (1) `nonweight_diff` (config / tokenizer / template) expects safetensors side files —
guard it for a `.gguf` model (`nonweights=false`, or skip when the model is a GGUF; the GGUF header has the
tokenizer, not read here). (2) The view's `hidden()` holds the hc_* tensors: print them in the report's "only in
model" section (the fingerprint's `only_in_model` is empty because unmapped names never reach it). (3) dtype:
`fingerprint` rounds both sides to the lower dtype, so BF16 GGUF vs BF16 safetensors is exact and F32 GGUF norms vs
BF16 HF norms compare at bf16. (4) Call `reader.close()` (or let it go out of scope + `gc.collect()`) before any
rename/delete of the GGUF on Windows. (5) `layer_map` compaction is silent: if a GGUF ever has a genuinely missing
block, its later layers would shift; `GGUFAsHF._layers` holds the map for the report.

Command once integrated:

```
step model_report model=hf_cache/sensenova-loop-gguf/SenseNova-Loop-9B-BF16.gguf base=hf_cache/qwen35-9b-hf nonweights=false out=lab/runs/model_report/sensenova_loop_9b.md
```

Expect: 427 tensors compared, no shape mismatch (same vocab 248,320 and head), `only_in_model` empty but
`hidden()` with 96 `hc_*` + 15 `blk.56.*`; `dt_bias` untouched, everything else touched with rel 0.1-0.4 and ranks
near full (diffuse) → `full_fine_tune`; the base link is the F32 decay parameters, not the matrices.
