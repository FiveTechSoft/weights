# `adapter_report`: a PEFT LoRA adapter read from the adapter alone — built 2026-11-18 (roadmap gap 29a)

**What this is.** `engine/adapter_report.py` (tests first: `tests/test_adapter_report.py`, 11, synthetic base + adapter
with known A, B, r = 2, scale 2; a second fixture with rslora; an IA3 config refused). For an adapter repo (local dir or
`hf:owner/repo`), it parses `adapter_config.json`, pairs `lora_A` / `lora_B` by module stem (both PEFT name styles,
with or without `base_model.model.`), maps each stem to the canonical base tensor name, and computes from the factors
only (never a base-sized dW): ‖dW‖_F two ways (trace identity on BᵀB ⊙ AAᵀ; singular values through thin QR of B and
Aᵀ), effective rank at 90 % energy, the leading output-side direction, and, when `base=` is given, ‖dW[rows]‖ / ‖W[rows]‖
on model_report's middle block of rows read from the base by one HTTP range per tensor. Families touched / untouched
(against the base's tensor kinds), embeddings / head touched, modules outside the stated targets, per-kind direction
consistency (median |cos| of leading directions across layers), parameter count from the safetensors header vs from
the shapes. CLI: `python -m engine.adapter_report hf:owner/repo --base Qwen/… --out report.md` (JSON beside it).

**Proposed registration (wsplang.py, not applied):**
```
step adapter_report adapter=<dir|hf:repo> [base=<dir|repo id>] [rows=128] [workers=16] out=<report.md>
```
Recommendation: a separate op, not a branch inside `model_report`. `model_report` answers "model vs base" from blocks
of both; an adapter has no full tensors, only factors, so its statistics are exact and cheap, and the two should not
share a classifier that expects sampled blocks. Read-only; goes in the same "reports, never weights" list.

## Real run: `hf:harness-hub/Qwen3.8-27B-pi-adapted` against `Qwen/Qwen3.8-27B` (18 Nov 2026)

Adapter-only pass 28 s; with the base's row blocks over HTTP 53 s total (16 threads). Files
`lab/runs/adapter_report/pi_adapted.{md,json}` (and `pi_adapted_adapter_only.md`).

- LoRA r 32, alpha 64 (scale 2), no rslora; 256 modules, 159,383,552 parameters (header count identical), 637.6 MB.
  Every module's shape matches the base; nothing outside the stated targets; embeddings, head, norms and all
  `linear_attn.*` tensors untouched.
- Where: MLP `gate / up / down` on all 64 layers (192 modules) and `q / k / v / o` only on the 16 full-attention
  layers (3, 7, …, 63), because the other 48 layers are Gated-DeltaNet (`in_proj_*`, not targeted). 89 % of the delta
  energy is MLP; by depth 22 / 22 / 30 / 26 % over layers 0-15 / 16-31 / 32-47 / 48-63 (slightly heavier in the back
  half; layer 63 largest, 2.6 %).
- How large: ‖dW‖ per module 0.05-0.27; block ratio ‖dW‖/‖W‖ **0.0008-0.0025** (median 0.0019): a ~0.2 % relative
  change everywhere, nothing concentrated.
- Rank: median effective rank 25 of 32 for MLP (diffuse within the adapter's rank); attention lower: `q_proj` median 16,
  **`v_proj` 1-4 in layers 3-15 and 6-11 in 19-47** (rank-one-like value updates early, e.g. layer 3: s = 0.051,
  0.009, 0.008), rising to ~20 in layers 51-63.
- Direction consistency across layers ≤ 0.026 for every kind: per-layer directions, no shared global direction
  (unlike abliteration).
- Two methods agree: trace vs SVD norms differ by ≤ 2e-15 relative; header vs shape parameter counts identical.

Nothing here says what the adapter does; it says the change is small, everywhere in the MLP, nearly full-rank-32 there,
and nearly rank-one in the early value projections.

## Open
- GGUF adapters (`convert_lora_to_gguf.py` output) are not read; only PEFT safetensors.
- Fused modules (`qkv_proj`, `gate_up_proj`) would map by name but the base block comparison assumes the same shape.
- `modules_to_save` (fully trained modules shipped beside the LoRA) are not reported; the real repo has none.
