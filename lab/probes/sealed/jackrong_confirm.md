# Sealed confirmation 1: the main result (Jackrong adapter on the base) — written 2026-11-10 before any run

Claim under test (README "Main result", results §43.6/§43.8): `delta_to_lora` of Jackrong's distill, rank 16, applied
×0.5 on Qwen3.5-9B, makes the base solve far more math at a 2,048-token budget (dev: MATH 6 → 30/36, GSM8K 9 → 37/48).

**Set:** `data/sealed_confirm.json` (sha256 3acfbc8d…), first and only use for this claim.
**Arms (one server, per-request adapter):** base (adapter ×0), adapter `hf_cache/close_lora/demo_jack_r16.gguf` ×0.5,
placebo ×0.5 (`placebo_lora.py`, same rank and Frobenius norm per matrix, random directions, seed 2026).
Base GGUF `hf_cache/h4/A0-Q6_K.gguf`; `engine.bench_runner.run(..., gsm_idx, math_idx, max_tokens=2048)`, greedy, 2 slots.

**Criterion (all three to confirm):**
- C1 MATH-500 (50): adapter > base, exact McNemar p < 0.05 with grader 1 and with grader 2.
- C2 GSM8K (100): adapter > base, exact McNemar p < 0.05 with grader 1 and with the last-number grader
  (the `\boxed` grader is not used for GSM8K: Jackrong-style answers put the box in the reasoning, results §43.14).
- C3 placebo not better than base: p >= 0.05 or placebo below base, both benchmarks.

Outcome is written to results.md whatever it is; nothing is re-run or re-tuned after seeing it.
