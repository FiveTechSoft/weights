# Happiness steering with a WSP-Lang control vector (fixed 2026-11-09, before any run)

**Goal (driver's request):** reproduce the "happy Qwen" effect (an overly cheerful, enthusiastic tone) on Qwen3.5-9B
with WSP-Lang, and measure it with controls.

**Method.** 120 everyday prompts (`happy.py prompts`, seeded split 70 train / 10 dev / 40 test). The base model answers
each train prompt twice, with a "happy" and a "neutral" system prompt (`happy.py pairs`); the system prompt is dropped
from the texts. `step contrast_vector` (engine/contrast_vector.py) averages the residual stream over the response tokens
per layer and writes happy − neutral as per-layer increments, plus a placebo with the same norm per layer in random
directions (`programs/happy_vector.wspl`). It is applied at run time (`llama-server --control-vector-scaled`), never
merged: an additive push cannot be stored in Qwen3.5's bias-free weights.

**Scale (rule fixed now):** on the 10 dev prompts, scales 0.5 / 1 / 2 / 4; the largest whose answers the judge calls
coherent in ≥ 9/10. If none: FAIL, no retuning.

**Measures on the 40 test prompts** (no system prompt, thinking off, greedy, 200 tokens): base, vector at the chosen
scale, placebo at the same scale.
- Method 1, judge: MiMo-9B (another model) rates tone 1-5 and coherence yes/no.
- Method 2, lexicon: positive-affect words + "!" per 100 words (fixed list in `happy.py`).
- Retention: GSM8K fast set (48 items, 2 048 tokens) with the vector, against the base's 9/48.

**Criterion (all must hold; `happy.py verdict`):**
- C1 judge tone +1.0 or more over the base, paired sign test p < 0.01;
- C2 lexicon agrees: +2 per 100 words or more, p < 0.01;
- C3 placebo tone gain < +0.5;
- C4 vector answers coherent in ≥ 36/40;
- C5 GSM8K with the vector not below the base by more than 2.
FAIL → written as negative, no retuning on the test prompts.

**Limits:** one model, 40 test prompts, a model judge (MiMo) and a crude lexicon; tone is not a capability.

## Amendment 2026-11-09 19:20 (after the dev run, before any test arm with the vector)
1. **Judge bug fixed.** MiMo ignores `enable_thinking: false` and opens with `<think>`; with 8 tokens every coherence
   answer was cut inside it and read "no" (first run: 0/10 coherent at every scale). `judge_answer` now reads the verdict
   after `</think>` with 1024 tokens; replies cut inside the think block are counted (`judge_unparsed`). First-run files
   kept in `lab/runs/happy/judge_v1/`.
2. **Re-judged dev:** scale 0.5 tone 5.0, 4/10 coherent; 1.0 0/10; 2.0 and 4.0 degenerate ("OHOOOO..."). By the rule as
   written no scale qualifies: **the pre-registered grid FAILS** (reported as such).
3. **Grid extended downward on dev only:** 0.125 and 0.25, same rule (largest scale with >= 9/10 coherent). Test set
   untouched until the scale is chosen; criteria C1-C5 unchanged. A pass is reported as "pass after grid extension".

## Amendment 2, 2026-11-09 19:30 (still before any test arm with the vector)
The judge called answers cut by the 200-token generation limit incoherent: the **unsteered base** got 4/10 on dev
(4/5 complete answers coherent, 0/5 cut ones). The coherence question now says the reply may be cut by a length limit
and that this does not count against it (generations unchanged). Validity check fixed with it: the base must get >= 9/10
on dev, otherwise the judge is unusable and the study stops. Result: base 9/10 (valid); 0.125 10/10, 0.25 10/10,
0.5 8/10, 1.0 0/10 (2.0 and 4.0 degenerate) -> rule picks **0.25**. Judge-v1/v2 files kept in lab/runs/happy/judge_v*/.
C1-C5 unchanged; the outcome is reported as "after two measurement fixes".
