# Sealed confirmation set (2026-11-09)

`data/sealed_confirm.json`: 100 GSM8K test problems and 50 MATH-500 problems, made once by `python -m engine.sealed_set make`
(seed 20261109) and committed with a sha256 of their text and answers
(`3acfbc8d1e731a474a59b0df9189f2f6057b4df080591f7d667f65c72ab2c0d6`). `python -m engine.sealed_set check` verifies it.

**Why.** The transplant dev set (`engine/bench_runner` defaults: GSM8K test 0-99, MATH-500 `select_indices(2026, 50)`) has
been used many times to pick doses, scales and donors, so a pass there is optimistic. The sealed set excludes those items.
Idea taken from the hashed sealed test sets in matthewgribben.com/blog/written-neural-nets-functional.

**Rule (fixed now, before any use).**
1. Run only on an arm that already **passed** its pre-registered criterion on the dev set, plus that study's reference
   and placebo arms. Same server settings and code (`bench_runner.run(..., gsm_idx=s["gsm8k"], math_idx=s["math500"])`).
2. One run per arm. No dose, scale or prompt is changed after seeing it.
3. The same criterion as the study's plan is applied, with exact McNemar against the reference, and the result is
   reported whatever it says. A dev pass that fails here is reported as "does not confirm".
4. Nobody reads the sealed problems or answers to tune anything. Opening them for debugging voids the seal; make a new one.

**Limits.** The sealed items are unseen *by this work's decisions*, not by the world. Earlier repo experiments
(`rig/run_full_mimo_ab_benchmark.py`, `blog_prompts/eval_math.py`, `exp8_control`) scored whole GSM8K / MATH-500 on other
models before 2026-11-09, and both datasets are public, so pretraining contamination is possible for every arm alike.
