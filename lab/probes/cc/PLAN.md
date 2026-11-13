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

## Composition: does "conclude" add to the Jackrong adapter? (added 2026-11-12, before any run)
Two edits that each work on the base (adapter r16 x0.5, results 43.15; conclude vector, 43.18 / cc_v2 pending): if we
understand them, we predict their sum. Both act on closing, so the prior is redundancy. Descriptive; runs after the
name test, only if cc_v2 passed dev. Code `lab/probes/cc/comp.py` (tests first), chain `lab/runs/cc_v2/chain_comp.sh`,
runs in `lab/runs/cc_v2/comp/`. Dev set (`bench_runner` defaults), 2,048 tokens, greedy, 2 slots, one run per arm.
Server: `--lora hf_cache/close_lora/demo_jack_r16.gguf --lora hf_cache/close_lora/placebo_demo_jack_r16.gguf
--lora-init-without-apply`, adapter scale per request; control vector by server flag.

Arms: base (cc_v2's `dev_base`, identical to 43.17's); C = cc_v2 x SC alone (cc_v2's `dev_vec`); J = adapter x0.5 alone
(new); J+C = both (new); J+P = adapter x0.5 + cc_v2 placebo x SC (new; "any push of this size on top of J").
Smoke first: 2 GSM8K items with J+C to confirm the two flags work together; stop and report if the server rejects them.

Readings (exact McNemar, grader 1, each benchmark):
- K1 J+C vs J: "adds" if J+C > J with p < 0.05 on either benchmark and not below on the other; "interferes" if J+C < J
  with p < 0.05 on either; else "redundant" (within noise).
- K2 tokens: mean tokens J+C / J per benchmark.
- K3 cost: cut answers and loop tails of J+C against J.
- Control: J+P within +-3 of J on both benchmarks (p >= 0.05); if J+P is below J, the vector's push is not free on top
  of J either, which bounds K1's reading.
Prediction (ours, low confidence), arm and control, premises marked:
- J alone on dev: GSM8K 86-93, MATH 31-37 (sealed gave 91 / 35).
- J+C: redundant on GSM8K (within +-3 of J); MATH 2-6 below J (the vector's loop tails cost more where J already closes);
  tokens 10-25 % below J; loop tails above J's. Overall K1: "redundant", leaning "interferes" on MATH.
- J+P: 3-10 below J on GSM8K (the placebo hurt the base by 13-20; J buffers part), MATH within +-3.
- Premises: (p1) llama-server applies LoRA and control vector together (smoke test); (p2) J on dev is within the sealed
  range; (p3) SC = 1.0.
- "We would be surprised if": J+C beats J by > 5 on either benchmark (would mean two different mechanisms of closing), or
  J+P is above J.
Cost: 3 arms x ~35 min + smoke ~ 1.9 GPU hours.

## Prompt control: does an instruction to close do what the direction does? (added 2026-11-13, before any run)
Outside review (2026-11-13, point 2) and our own gap: the base fails on dev mostly by not finishing inside 2,048 tokens
(cut 51/100, 45/50; results 43.2: with 8,192 tokens it still cuts 30/50 and never answers wrong). "conclude" is read as
closing, not capability (T2). Missing arm: the same closing asked for **in the prompt**. If a sentence does what the
direction does, the direction's value is "no prompt, compilable into the weights", not the number. Descriptive; runs
after the recompilation chain (`lab/runs/cc_v2/chain_prompt.sh`, one GPU job at a time). `bench_runner.run(extra=...)`
appends the sentence after the usual suffix (test first); everything else as dev (§43.17 settings, 2,048, greedy, 2
slots, dev items, base A0, no vector, no adapter).

Arms (new): **P_close** = base + " Once you have the answer, stop reasoning and give it: no re-checking, no restating."
**P_neutral** (control: "any extra sentence") = base + " Use clear notation and define every variable you introduce
before using it." Reused: base and cc_v2 x1.0 from `lab/runs/cc_v2/dev_{base,vec}.json`.

Readings (exact McNemar, grader 1, each benchmark; grader 2 reported):
- R1 P_close vs base: does the sentence help at all (p < 0.05 on either)?
- R2 P_close vs cc_v2: "prompt-equivalent" if within +-3 on both and p >= 0.05 on both; "the direction does more than a
  prompt" if cc_v2 > P_close with p < 0.05 on either; "the prompt does more" if the reverse.
- R3 cost: tokens, cut answers, loop tails of P_close against base and against cc_v2.
- Control: P_neutral within +-3 of base on both (p >= 0.05) and cut answers within +-5.
Prediction (ours, low confidence), arm and control, premises marked:
- P_close: GSM8K 62-75, MATH 12-22; cut 20-35 / 30-40; loop tails >= base's (1 / 1): the farewell loop ("I will write
  the response", 43.2) is not an instruction-following failure, so a sentence shortens the honest cases and leaves the
  loops. Tokens GSM8K 1,100-1,400 (base 1,689).
- R2: "the direction does more than a prompt" (cc_v2 83 / 35 above P_close with p < 0.05 on GSM8K at least).
- P_neutral: within +-3 of 50 / 6, cuts within +-5 of 51 / 45.
- Premises: (p1) `extra` reaches the server unchanged (test); (p2) base dev is reusable (deterministic, shown three
  times); (p3) the model follows a closing instruction at all (if P_close == base on tokens too, the arm says nothing
  about the direction and is reported as such).
- "We would be surprised if": P_close >= cc_v2 - 3 on both benchmarks (the direction would be instruction following
  written into the stream, and the name test's cross-domain shortening would be re-read as the same); P_close below
  base on either.
Cost: 2 arms x ~50 min (long answers) ~ 1.8 GPU hours.

## Placebo distribution: is "a same-norm random direction does not help" robust to the seed? (added 2026-11-13, before any run)
Every placebo in this line (vector §43.17-43.18, §45.4; compiled §43.19-43.21, §45.8; composition §45.7) had **one
seed, 2026**. Outside review (2026-11-13, point 4): one seed is one draw; the claim "a random direction of the same norm
does not help, and tends to hurt" (T6, T3) rests on it. This arm measures the distribution. Descriptive; nothing already
confirmed is re-decided by it.

**What runs** (`lab/runs/cc_v2/chain_placebo.sh`, after the prompt-control chain; one GPU job at a time):
1. `lab/probes/cc/placebo_seeds.py check`: the placebo of seed 2026 regenerated from `cc_v2.gguf` (vector read back
   from the GGUF in layer order, `contrast_vector.placebo`) must be **identical** to `cc_v2.placebo2026.gguf`. If not,
   stop (the generator would not be the one used so far). Done before writing this plan: identical, max |diff| 0.
2. `make 1..8`: eight placebos `hf_cache/cc/cc_v2.placebo{1..8}.gguf` (same norm per layer as cc_v2, random direction).
3. Each at **scale 1.0** on the **screen set** (`split.json` screen_math, 64 MATH-500 items, not dev, not sealed; 2,048
   tokens, greedy, 2 slots), reports `lab/runs/cc_v2/screen_pl{seed}.json`. Reused: `screen_base.json` (8/64, 57 cut)
   and `screen_s1.0.json` (38/64, 30 cut, 20 loop tails), both from §45.4 on the same items.
4. `readings 1..8` -> `lab/runs/cc_v2/placebo_seeds/readings.json`.

Readings (grader 1; grader 2 reported):
- D1 distribution: per seed correct/64, cut, loop tails, mean tokens; median / min / max over the 8 seeds.
- D2 "no seed helps": number of seeds above base with exact McNemar p < 0.05, and number at or above **half the
  vector's gain** (base + ceil((38 - 8) / 2) = 23/64).
- D3 "harm": median of the seeds at or below base; number of seeds below base with p < 0.05.
- Vector rank among {8 placebos, vector}: 1 means no placebo ties or beats it (descriptive, 1/9 under the null).
Prediction (ours, low confidence), arm and control, premises marked:
- D1: median 5-9 correct (base 8), min 2-5, max <= 12; cut 52-63 per seed (base 57); loop tails 2-12; mean tokens
  1,850-2,000 (base 1,987).
- D2: 0 seeds above base with p < 0.05; 0 at or above 23. D3: median <= 8; 0-3 seeds significantly below base (on dev
  seed 2026 was 26 vs 50 on GSM8K, p < 1e-4, but only 5 vs 6 on MATH: MATH at this budget is at the floor, so little
  room to harm with significance). Vector rank 1.
- Premises: (p1) the generator reproduces seed 2026 exactly (checked, see step 1); (p2) screen base is reusable (the
  identity check of §45.4 matched §43.17 item by item); (p3) scale 1.0 puts each placebo at the vector's norm, as on dev.
- "We would be surprised if": any seed >= 20/64; median >= 12; vector rank > 1 (a random direction of that norm would
  sometimes do what the direction does, and §43.17-45.8 would need a different control).
Cost: 8 arms x ~25 min (screen arms ran 1,200-1,700 s) ~ 3.5 GPU hours.

## Token-budget control: the base at 8,192 (added 2026-11-13, before any run; prediction row 31)

**Question.** Is "conclude" an accuracy gain or a token-budget effect? Between finished answers the base is 49/49
and the vector 79/92 on GSM8K dev (45.9). Reading the 45.8 arms directly (cap8192/thirteen.json,
cap8192/early_wrong.json): every GSM8K gain of the vector over the base (36 items) is a base token cut; the vector
closes 13 finished-but-wrong answers, 10 of them also cut in the base and 3 that the base had right (early-close
cost). In MATH the vector cuts 17 vs 45. Everything so far is consistent with "closes instead of cutting".

**Arms** (`lab/runs/cc_v2/chain_cap8192.sh`, after the placebo and gate-3b chains; one GPU job at a time;
idempotent markers; GSM8K dev 100, the same items as 45.8, greedy, 2 slots):
1. base at 8,192 (new run);
2. vec at 8,192 (`cc_v2` control vector at scale 1.0);
3. merged at 8,192 (the compiled Q6_K; consistency with the vector arm).

**Readings** (grader 1, grader 2 reported): ok/100, cuts, finished-wrong, mean tokens per arm; per-item on the
36 base-cut items (how many the base now closes ok), on the 13 vector early-wrong items (does it still close
them wrong at 8,192?), and the 3 base-ok items (early-close cost persists?).

**Prediction (ours).**
- Base at 8,192: ok 82-92/100, cuts <= 8. If so, the GSM8K effect of conclude is finishing-in-budget (saving,
  0.4x tokens), not new capability, and the placebo's 28-34 at 2,048 is prompt fragility of the base, not noise.
- Vec at 8,192: ok = its 2,048 value (83) +/- 2; it still closes early (its 13 finished-wrong stay mostly wrong);
  mean tokens far below base.
- Merged = vec within 2.
- We would be surprised if: base at 8,192 < 70 (then conclude is a real accuracy gain, the framing changes);
  vec at 8,192 beats base at 8,192 by more than 5 (effect beyond finishing); base's 3 early-close-ok items
  (50, 63, 82) also ok under the vector at 8,192 (early-close cost would be an artifact of the 2,048 budget).
Cost: 3 arms x ~2 GPU h (8,192-token cap doubles-to-quadruples arm time) ~ 5-6 GPU hours.

[rev 2026-11-13 21:20, by the session that integrates this section, before any arm ran; prediction left untouched.]
- The base at 8,192 on these GSM8K items was already measured: results §39.6, `lab/runs/h4/h4_A0.json`, same A0
  Q6_K, greedy, 8,192: **72/100, 24 cut**, mean 3,638 tokens (MATH: §43.1, 20/50, 30 cut). Paired with the vector at
  2,048 it is 17 won / 6 lost (results §45.9, note). The new base arm is therefore a repeat under this chain's server
  settings (`-c 40960`); its value is the consistency check with the vector and merged arms run the same way.
- The chain's GPU guard (`pgrep -x llama-server`) does not exist on this system (`pgrep: command not found`), so it
  only waits for "PLACEBO DONE". Sequencing is done by the gate 3b chain instead (`lab/runs/act_map/cc08_chain.sh`
  waits for this chain's `CHAIN DONE` / `CHAIN EXITED` marker and for no llama-server alive). Order: row 29 → row 31 →
  row 30. Prediction row 31 in `predictions.md`.
