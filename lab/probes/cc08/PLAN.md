# "Conclude" without a donor on Qwen3.5-0.8B: does the 9B result carry across size? — plan fixed 2026-11-11, before any run

**Why.** On Qwen3.5-9B the base's own "conclude" direction, read from its traces and applied as a control vector,
passed dev and the sealed set (results §43.17-18: sealed GSM8K 43 → 87, MATH 4 → 25, same-norm placebo below base) and
survives compilation into `down_proj` (§43.19 dev). The question raised by "porting" a subspace to another model has a
cheaper first form: **re-derive the same thing in the other model**. Qwen3.5-0.8B has the pathology worse (Hu et al.,
arXiv 2611.09382: base cut in 111/128 on MATH; their +29.45 MATH edit is one `down_proj` column, a "Wait/However vs
Therefore" switch). If re-derivation works there, it is (a) a replication across size and (b) the ceiling and control
for any later cross-model map (step 2 of the porting proposal, not in this plan). If it fails, mapping a 9B direction
into the 0.8B has nothing to aim at.

Same protocol as `lab/probes/cc/PLAN.md` (§43.17), changed only where the model forces it. Nothing tuned on dev.

## Model and serving
`hf_cache/qwen35-08b-hf` (HF, bf16; d = 1 024, 24 layers, 3 584 neurons) for the vector; `hf_cache/qwen35-08b-gguf/
Qwen_Qwen3.5-0.8B-Q6_K.gguf` for traces and benchmarks, llama-server same flags as §43.17 (thinking template, greedy,
2 slots, `-c 16384`). The control vector has 23 increments (layers 1-23). Code: `lab/probes/cc/cc.py` gains a model /
runs-dir switch (env `CC_HF`, `CC_GGUF`, `CC_RUNS`; defaults unchanged so the 9B run is reproducible); outputs in
`lab/runs/cc08/`, vector `hf_cache/cc08/cc08.gguf`, placebo seed 2026.

## Data (no download)
The **same items** as the 9B study (`lab/runs/cc/split.json`): 64 MATH-500 discovery + 64 GSM8K train discovery, 64
MATH-500 screen. None is in dev, sealed set 1 or sealed set 2. They decided nothing about the 0.8B.

## Steps
1. **Traces**: base 0.8B Q6_K, thinking on, greedy, 4 096-token cap, the 128 discovery problems. Keep traces that
   close `</think>`. Need ≥ 30. **Fixed fallback** (the 0.8B closes rarely): if fewer than 30, add GSM8K train 64-191
   (128 more, once) and recount; still < 30 → stop, report "no closing traces to read a direction from" (that is a
   result: the direction cannot be re-derived in a model that almost never concludes).
2. **Rows**: as §43.17 — pos = last 48 tokens before `</think>`, neg = earlier reasoning (≤ 1 536). Also reported:
   how many kept traces are GSM8K vs MATH (the 9B's were 39 / 14).
3. **Vector**: WSP-Lang `contrast_vector`, placebo seed 2026 (same norm per layer, random direction).
4. **Screen** (scale; screen items only, MATH, 2 048 tokens): base and vector at 0.25 / 0.5 / 1 / 2. Rule: most correct
   (grader 1), ties → smaller scale. **Also fixed now:** if base and every scale score ≤ 2/64 (floor), the rule uses
   answers *finished* (closed `</think>` within budget) instead of correct, ties → smaller scale; written as such. No
   scale beats base on that measure → FAIL, nothing goes to dev.
5. **Dev** (`bench_runner` defaults: GSM8K 0-99 + MATH-500 select_indices(2026, 50), 2 048 tokens): base, vector at the
   chosen scale, placebo at the same scale. One run per arm.

## Criterion on dev (all of C1-C3, identical to §43.17)
- **C1 gain**: vector beats base, exact McNemar p < 0.05, on MATH with grader 1 and grader 2, **or** on GSM8K with
  grader 1 and the last-number grader; **or** mean tokens −15 % or more on both benchmarks with accuracy not below base
  by more than 2 items on either.
- **C2 placebo**: not better than base (p ≥ 0.05 or below) on both benchmarks and not −15 % tokens.
- **C3 no damage**: vector not below base by more than 2 items on GSM8K (grader 1).

Reported, not criteria: cut answers, loop tails, mean tokens; comparison with Hu et al.'s 0.8B numbers (different
protocol and budget, so only as context); the cosine, after a least-squares map fitted on the same texts, between the
0.8B and 9B vectors is **not** computed here (that is the porting step, with its own plan if this passes).

PASS → sealed check on a fresh set 3 (made before that run, excluding everything in `lab/runs/`), arms base / vector /
placebo. FAIL → results.md says so, with which step stopped it.

## Cost and queue
0.8B: traces + 5 screen arms + 3 dev arms ≈ 1-2 h on the 3060. Runs after the GPU queue now in place (Turbo forensic →
sealed set 2 of cc_weights → cc_neurons phase A); one GPU job at a time.

## Limits known now
One small model, one budget, greedy. The 0.8B may close mostly on GSM8K, so the direction may be "GSM8K-shaped" and
transfer less to MATH; the screen is on MATH, so that would show up before dev.
