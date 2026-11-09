# short_reasoning — one decisive experiment instead of more proxies (fixed 2026-11-09, before any number)

## Why this one
- The only measured, behavioural deficit of MiMo-9B in this repo is MATH-500 at an 8192-token cap: 34/50, with 15 of 50 cut at the cap
  (results.md §37). Re-reading `lab/runs/model_compare/*.json`:
  - On the 35 MATH items MiMo finishes, it gets **34/35**; TwIL gets 33/35 on the same items.
  - Of the 15 MiMo cuts, TwIL solves 11. The whole 68 % vs 88 % gap is length, not skill.
  - GSM8K: 4 cuts, 92/96 on finished items.
- So the target is **reasoning length**, a property one existing sibling model changes on purpose:
  `iAmBoosted/Qwen3.5-9B-OSS-Distilled`, LoRA r16 SFT of **Qwen/Qwen3.5-9B** (MiMo's own SFT parent) on gpt-oss-20b traces;
  its card reports no-answer ("spiral-out") 36.2 % → 0.5 % on 400 held-out prompts. Apache-2.0.
- F27–F36 measured proxies (rank, sparsity, layer outputs) on a 2-CPU machine and cannot reach a behavioural verdict.
  This runs on the 3060 and ends in keep / discard.

## Step 0 (no weights, ~1 h): is it only the cap?
`rerun_cut.py`: the 19 cut items again at 32768 tokens, same greedy protocol. If MATH goes to ≥ 44/50, MiMo's "MATH problem" is a
budget setting, and the honest fix is to report it with a 32k cap. Steps 1–5 then target **cost** (tokens), not accuracy.

## Arms (steps 1–4)
All four go through the same `convert_hf_to_gguf.py` → `llama-quantize Q6_K` (no imatrix), so only the weights differ.
| arm | weights |
|---|---|
| mimo_ref | MiMo |
| oss_a100 | MiMo + 1.0·(OSS − Qwen3.5-9B) |
| oss_a050 | MiMo + 0.5·(OSS − Qwen3.5-9B) |
| placebo_a100 | MiMo + random delta, same Frobenius norm and same 90 %-energy rank per tensor as OSS − Qwen3.5-9B |
Protocol: `engine/bench_runner.py` as in §37 (GSM8K 100 + MATH-500 50, greedy, 8192 tokens).

## Criterion (written now; `compare.py` applies it mechanically)
An oss arm PASSES only if all three hold:
- C1: MATH-500 items cut at the cap fall by ≥ 5 against mimo_ref.
- C2: total correct (GSM8K + MATH) is not lower than mimo_ref by more than 2.
- C3: the placebo does not reach C1.
PASS → confirmation on unseen items (MATH-500 `select_indices(2027, 50)` excluding the 2026 set) before any claim. FAIL → stop; the result
is written as negative. No re-tuning of alpha, layers or subsets on these 150 items.

## Known limits
- n = 150, one seed, greedy: a 2-item accuracy change is noise. C1 targets a large effect on purpose.
- The donor delta was trained on Qwen3.5-9B, not on MiMo (9.2 % away): transfer may fail; that is what is being tested.
- The arm runs MiMo's tokenizer and template (F25 regex difference does not affect GGUF).
- `merge_delta.py` compares F32 vs BF16 storage at BF16 so a cast is not counted as change (F26). Tested on synthetic tensors only
  (exact merge, untouched non-selected tensors, name-prefix mapping, placebo norm/rank, cast case).

## Implementation used on the 3060 (2026-11-09, written before any arm was run; criterion C1-C3 unchanged)
- **Adapter instead of merged checkpoints.** The donor is a merged LoRA r16, so (OSS − Qwen3.5-9B) is extracted with the
  WSP-Lang op `delta_to_lora` (exact SVD, rank 16, read from the hub by byte ranges: `model=hf:...@7a7c96f`) and applied
  on MiMo's Q6_K at run time (`llama-server --lora ... --lora-init-without-apply`, per-request scale = alpha). Same arms,
  no 3 × 18 GB merged copies. `energy.json` records how much of the donor change rank 16 keeps (≈ 1.0 expected for a
  merged r16 LoRA; if it is much lower, the adapter is not the merge and this is reported).
- **Placebo:** `lab/probes/close_probe/placebo_lora.py` on that adapter: same rank (16) and same Frobenius norm per
  matrix, random directions (the zip's placebo matched the 90 %-energy rank, capped at 64).
- **mimo_ref:** MiMo own Q6_K without imatrix under this protocol already exists (`lab/runs/model_compare/own_mimo.json`,
  results §43.1: GSM8K 92/100 grader 1, MATH-500 41/50, **13 cut**); reused, copied to `lab/runs/short_reasoning/mimo_ref.json`.
  The 34/50 and 15 cuts quoted above are the bartowski Q6_K (§37); quantization changes these numbers (§39.6).
- **Step 0** (cut items at 32k) is not run: it measures cost, not the weights.
- `merge_delta.py` and `run_windows.sh` are kept as received, not used. Note: `Snapshot.get` opens `safe_open` per tensor,
  which crashed natively on Windows in this repo (cf2, tmax); use `engine.weight_diff.LocalSafetensors` if it is ever run.
