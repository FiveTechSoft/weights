# "Conclude" direction on Qwen3.5-9B without a donor — plan fixed 2026-11-10 before any run

Replication, adapted, of concept.country's weight-space programming on Qwen3.8-27B
(https://concept.country/research/weight-space-programming, paper v1 2026-11-08; post https://x.com/varien/status/2119880941711827025):
a direction read from the model's own reasoning traces — residual stream over the last tokens before `</think>`
minus the rest of the thinking ("cc") — pushes the model to conclude, cutting thinking tokens 27-54 % and raising
GPQA / LiveCodeBench, with a norm-matched random edit doing nothing. They compile it into 9 `down_proj` matrices;
here it is first applied as a run-time control vector (same push, nothing written to weights; nathanrs.rs found a
constant push gives almost all of such an edit). Compiling into weights only if this passes.

Why here: at 2,048 tokens Qwen3.5-9B leaves 57/100 GSM8K and 46/50 MATH answers unfinished (results §43.15); the
Jackrong adapter fixes that with a donor. Question: does the base's own "conclude" direction do it with **no donor**?

## Data (no download; nothing from the dev set or the sealed set)
`cc.py split`: MATH-500 items not in `select_indices(2026, 50)` (dev) and not in `data/sealed_confirm.json`, shuffled
with seed 20261110: first 64 = discovery, next 64 = screen. Discovery also uses GSM8K **train** items 0-63.

## Steps
1. **Traces** (`cc.py traces`): base Q6_K, thinking on (template default), greedy, 4,096-token cap, 128 discovery
   problems. Keep the traces that close `</think>`. Need >= 30, else stop and report.
2. **Rows** (`cc.py rows`): per kept trace, pos = the last 48 tokens before `</think>`; neg = the reasoning tokens
   before those (at most the first 1,536). Same trace, positions only.
3. **Vector**: WSP-Lang `contrast_vector` (rows with `think` and `focus`, engine/contrast_vector.py), placebo seed 2026
   (same norm per layer, random direction).
4. **Screen** (choose the scale; screen items only, 2,048 tokens, MATH only): base and vector at scales
   0.25 / 0.5 / 1 / 2. Rule: the scale with the most correct (grader 1); ties -> smaller scale. If no scale beats the base
   on the screen, stop: FAIL, nothing goes to dev.
5. **Dev** (`bench_runner` defaults: GSM8K 0-99 + MATH-500 select_indices(2026, 50), 2,048 tokens): base, vector at the
   chosen scale, placebo at the same scale. One run per arm.

## Criterion on dev (all of C1-C3)
- **C1 gain**: vector beats base, exact McNemar p < 0.05, on MATH with grader 1 and grader 2, **or** on GSM8K with
  grader 1 and the last-number grader; **or** mean tokens -15 % or more on both benchmarks with accuracy not below base
  by more than 2 items on either.
- **C2 placebo**: placebo not better than base (p >= 0.05 or below base) on both benchmarks and not -15 % tokens.
- **C3 no damage**: vector not below base by more than 2 items on GSM8K (grader 1).

PASS -> confirm once on the sealed set (rule in lab/probes/sealed/README.md), then try compiling into `down_proj`.
FAIL -> results.md says so; "needs a donor" stands for this kind of gain at this size.
Outcome reported whatever it is.
