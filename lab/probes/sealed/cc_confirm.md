# Sealed confirmation 2: donor-free "conclude" vector (cc) on the base — written 2026-11-10 before any run

Claim under test (lab/probes/cc/PLAN.md, dev PASS 2026-11-10, results §43.17): the control vector read from Qwen3.5-9B's
own traces (last 48 reasoning tokens before `</think>` minus the rest), scale 1.0 (chosen on the screen items), makes the
base solve more GSM8K / MATH-500 at a 2,048-token budget (dev: GSM8K 50 -> 84/100, MATH 6 -> 32/50; same-norm placebo
37 / 3).

**Set:** `data/sealed_confirm.json` (sha256 3acfbc8d…), second use; first use for this claim. Its items were never
used by any decision of the cc study (discovery and screen items exclude it by construction, `cc.py split`).
**Arms (fresh, same server settings as dev, one at a time):** base; `hf_cache/cc/cc.gguf` at 1.0; placebo
`hf_cache/cc/cc.placebo2026.gguf` at 1.0 (same norm per layer, random direction). Base GGUF `hf_cache/h4/A0-Q6_K.gguf`,
`--control-vector-scaled`, `bench_runner.run(..., gsm_idx, math_idx, max_tokens=2048)`, greedy, 2 slots.

**Criterion (all three to confirm):**
- C1 GSM8K (100): vector > base, exact McNemar p < 0.05 with grader 1 and with the last-number grader;
  **or** MATH-500 (50): vector > base, p < 0.05 with grader 1 and with grader 2.
- C2 placebo not better than base: p >= 0.05 or placebo below base, both benchmarks.
- C3 vector not below base by more than 2 on GSM8K (grader 1).

Also reported, not part of the criterion: answers that are correct but cut at the cap (the answer repeated in a
`\boxed` loop), and the fraction-format disagreements between MATH graders (seen on dev: 8/8 were `a/b` vs `\frac`).
Outcome is written to results.md whatever it is; nothing is re-run or re-tuned after seeing it.
