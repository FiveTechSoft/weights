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

## Repeat with the corrected op (addendum 2026-11-12, before any run)
Why: `contrast_vector` took the last row of D from transformers' `hidden_states[-1]`, which is after the final norm, so
`direction.31` of `hf_cache/cc/cc.gguf` was the norm's rescaling, not a residual change (13.2 against 9.1 at layer 30,
median 1.9; fixed in 4aeb453, results 45.2-45.3). The happy repeat with the fix failed (45.3), so 43.17-43.18 are
re-tested, not assumed.

Same traces, rows, items, scales, rules and criterion as above; only the op changes. Runs in `lab/runs/cc_v2`
(`CC_RUNS`), split / pos / neg copied from `lab/runs/cc`.
1. Vector: `programs/cc_vector_v2.wspl` -> `hf_cache/cc/cc_v2.gguf`, placebo seed 2026.
2. Mechanical check (stop if it fails): direction.1..30 equal to `cc.gguf` (max relative diff < 1e-3); report norm and
   cosine of direction.31 old / new.
3. Screen: base re-run as an identity check, then scales 0.25 / 0.5 / 1 / 2, same choose rule. If `screen_base` does not
   match `lab/runs/cc/screen_base.json` item by item, the dev and sealed base arms are re-run too; else they are reused
   (the setup reproduced 150/150 sealed items in 43.18).
4. Dev: vector and placebo at the chosen scale, C1-C3 as above.
5. Sealed confirmation, only if dev passes: `data/sealed_confirm.json` (third use; its items touched no decision of this
   repeat, the vector differs from 43.18's only in direction.31), vector v2 and placebo v2 at the dev scale, base
   reused under step 3's rule, criterion of `lab/probes/sealed/cc_confirm.md`, in `lab/runs/sealed_cc_v2`.

Prediction: passes dev and sealed; scale 1.0 again. A fail is reported as "43.17-43.18 held only with the old op", as
with 43.13. The compiled-weights results (43.19-43.21, 43.25) also used `cc.gguf`; they are re-tested only after this,
with their own addendum.
Cost: ~6 GPU hours (op 0.5 h, screen 2.3 h, dev 1.7 h, sealed 1.7 h).

### Extra arm, descriptive only (added 2026-11-12, before any cc_v2 result)
`hf_cache/cc/cc_no31.gguf` = `cc.gguf` without `direction.31` (layers 1-30 are the same in both ops). Run on dev at the
scale chosen by the cc_v2 screen (1.0, 43.17's, if it picks none), after the cc_v2 chain ends, once. Not a criterion and not run on the sealed set: it
only says whether the old layer 31 added to, or took from, 43.17's gain (compare dev_vec of cc, cc_v2 and this arm).

## Name test: does "conclude" behave as its name outside maths? (added 2026-11-12, before any run)
The direction was found on maths traces and named by us. It earns the name if it shortens reasoning in domains that
were never used to find, scale or name it, and the cost is bounded. Known so far (43.21, old vector): HumanEval-20
median tokens 650 -> 201 and correct 20 -> 15. Descriptive: not a criterion of the cc study, reported whatever it says.
Runs after the cc_v2 chain and the no31 arm, with the cc_v2 vector and placebo at the scale cc_v2 chose (1.0), only if
cc_v2 passed dev (otherwise the vector itself is in question first). Code: `lab/probes/cc/name_test.py` (tests first),
chain `lab/runs/cc_v2/chain_name.sh`, runs in `lab/runs/cc_v2/name/`.

Domains (both outside maths, thinking as the template does it, greedy, 2,048-token cap, same server flags as dev):
- D1 HumanEval first 20 (`lab/probes/close_probe/he_eval.py`): closed / correct / median tokens. Arms: base, vector, placebo.
- D2 Horn logic, the 120 cases of `rig/logic_probe.make_cases` (imported, not modified; its own runner uses 256
  tokens and no thinking, so it is not reused): correct = `extract_answer` on the text after `</think>` (whole text if
  no think block), cut = finish != stop, tokens. Arms: base, vector, placebo.

Readings (pre-registered, N = "name"):
- N1 shortens: vector median tokens <= 0.85 x base in both domains.
- N2 bounded cost: vector correct >= base - 2 on HumanEval (/20) and >= base - 6 on logic (/120).
- N3 closes: in a domain where the base cuts >= 10 % of answers, the vector cuts at most half as many. Not applicable
  if the base cuts < 10 % in both.
- Control: placebo median tokens within +-10 % of base and correct within +-2 (HumanEval) / +-6 (logic) in both domains.
Name holds: N1 in both and N2 in both. "Concludes at a cost": N1 in both, N2 fails somewhere. Name fails (the direction
is maths-specific or a format effect): N1 fails in either domain.

Prediction (ours, low confidence), arm and control, premises marked:
- D1: N1 passes (tokens ~200-350), N2 fails (correct 14-17 of 20), as with the old vector. Placebo 19-20, tokens within 10 %.
- D2: N1 passes (tokens -30 to -60 %); N2 passes (base and vector within 6); base cuts < 10 %, so N3 does not apply.
  Placebo within the control bands. Overall: "concludes at a cost".
- Premises assumed, not yet measured: (p1) the base closes most logic cases in 2,048 tokens with thinking on; (p2)
  HumanEval base reproduces 20/20/650 with these flags; (p3) the corrected layer 31 does not change the out-of-maths cost
  direction. If p1 fails the D2 reading is about closing (N3), not about shortening.
- "We would be surprised if": the vector lengthens either domain (> 1.1 x base), or the placebo shortens by >= 15 %.
Cost: ~1.5 GPU hours (HumanEval 3 x ~5 min, logic 3 x ~25 min).
