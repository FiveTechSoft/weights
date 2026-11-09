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
