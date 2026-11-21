# Editing a decompiled `.wspngram` program: criterion fixed before measuring (2026-11-05)

Question: can a decompiled program be edited as text, recompiled without gradients, and change one behaviour
of the model without touching the rest? This is weight-space programming at the source level.

## Setup
- Program: `lab/runs/decompiled/nanogpt_10.7M_order5.wspngram`, the behavioral decompile of the trained
  nanoGPT (val 1.4573). Compiled at the exact 6L/384/1536 shape, seed 1337 (val 1.7810 unedited).
- Val = last 10% of Shakespeare char. Every position is scored with at least 128 characters of context
  (windows of 256, stride 128).
- "Target positions" = val positions whose last 3 characters are the edited context.
  "Outside" = every other position.

## Edits (one text change each, to a single `unit` line)
- **E1 unlearn "lord":** in unit `"lor"`, entry `d` -= 12 nats. `Lord` (capital L) has another context and
  must not change.
- **E2 swap "king" / "kind":** in unit `"kin"`, entries `g` and `d` are changed so that the summed
  log-score of `g` and `d` at context `kin` are exchanged.

## Controls
- **C0, no edit:** the unedited program, compiled the same way.
- **C1, wrong place:** the E1 change (`d` -= 12) applied instead to 5 random units of similar frequency
  (count within 0.5x-2x of `lor`). It must not change P(d | lor).
- **C2, output-level bias:** -12 on the `d` logit at every position (a decoding hack, not a weight edit).
  It meets the target with no context specificity.
- **C3, gradient fine-tuning of the trained nanoGPT:** 300 AdamW steps, lr 1e-4, batch 32x256, train split.
  - E1: unlikelihood -log(1 - p(d)) at target positions, cross-entropy elsewhere.
  - E2: targets relabelled g<->d at target positions.
  - Settings fixed now and not tuned. It is a reference, with no pass/fail.
- **C4, direct weight edit:** apply the same change to the compiled weights by hand (one `mlp.c_proj` column),
  without recompiling. Also count how many weights differ between the C0 and E1 compiled models.

## Pass criteria (on the compiled models)
- **E1 passes if all hold:**
  1. Mean P(d) at target positions whose next char is `d` is <= 0.01.
  2. Mean |delta NLL| outside is <= 0.001 nats/char.
  3. In 20k sampled characters (16 streams, temperature 1, fixed seed), lowercase `lord` drops >= 95%
     versus C0.
- **E2 passes if:**
  1. At target positions followed by `g`, P(d) > P(g) in >= 95% of them.
  2. At positions followed by `d`, P(g) > P(d) in >= 95% of them.
  3. Mean |delta NLL| outside is <= 0.001.
- **C4 passes if** the logits of the direct edit match the recompiled edit within 1e-3, and the recompiled
  E1 differs from C0 in a single `c_proj` column.
- **C1 must change P(d | lor) by < 0.01.** C2 is expected to cost far more than 0.001 outside.

If E1 or E2 fails, the result is reported as a failure. Thresholds are not changed after seeing numbers.

## Addendum, written after E2 failed and before running E2b (same day)
E2 edited the wrong context. "kin" is shared with making, taking and lacking: their child units `akin` and
`ckin` add +2.4 and +3.8 to `g`, which outweighs the 1.85 swap. E2 condition 2 was also impossible for any
order-5 program: at ` kin` the base model already prefers `g` at every "kind" position, because 4 characters
cannot tell the words apart. E2 stays a failure.

**E2b:** the same swap, computed for unit `" kin"` (word-initial). It passes if all hold:
1. At val positions with context ` kin` (last 4 characters), the model's preference flips: P(d) > P(g) in
   >= 95% of the positions where the base model had P(g) > P(d).
2. At positions whose last 3 characters are `kin` but whose last 4 are not ` kin` (making, taking, ...),
   max |delta log p| <= 1e-6.
3. Mean |delta NLL| at every other position is <= 0.001.

## Outcome (measured 2026-11-05, `lab/runs/wspngram_edit.json`)
Base for E/C0-C2/C4: the unedited compiled program, val 1.7721 when every position has >= 128 characters of
context. Base for C3: the trained nanoGPT, val 1.4423. Every key number was checked a second way with
`UnitTable.log_prob` on the edited program, without compiling; both ways agree to 4 decimals.

| Run | Target | Outside: mean abs dNLL / max abs dlogp | val | Verdict |
|---|---|---|---|---|
| E1 unlearn `lord` | P(d after lor) 0.999 -> 0.0086 (13 positions); samples `lord` 15 -> 0, `Lord` 5 -> 7 | 0.0 / 0.0 | 1.7722 | **passes** |
| C4 direct weight edit | recompiled E1 differs from C0 only in `h.0.mlp.c_proj[:, 728]` (384 weights); direct edit = recompile within 1.9e-6 | - | - | **passes** |
| C1 same change in 5 random units | P(d after lor) change 0.0 in all 5 | 0.0 / - | - | as expected |
| C2 -12 on `d` at every position | P(d after lor) 0.0086 | 0.285 / 12.0 | 2.0264 | as expected: no specificity |
| C3 gradient unlearn (trained model) | P(d after lor) 0.9995 -> 0.0142; samples `lord` 19 -> 0 | 0.319 / 12.48 | 1.4727 (+0.030) | reference |
| E2 swap at unit `kin` | flips 25/39 king positions (0.641); condition 2 impossible | 0.0 / 0.0 | 1.7723 | **fails** |
| C3 gradient swap (trained model) | P(d) > P(g) at king positions 0.974; kind positions 0.0 | 0.320 / 12.43 | 1.4731 (+0.031) | reference |
| E2b swap at unit ` kin` | flips 32/32; making/taking etc. max abs dlogp 0.0; samples king 8 -> 4, kind 2 -> 6 | 0.0 / 0.0 | 1.7721 | **passes** |

Reading:
- A line of the program is one MLP column: editing the text equals editing 384 weights by hand, and nothing
  else moves. The edit is exact and local because the program is a gated lookup table; that is the property
  being bought.
- The price is the model itself: the program keeps 83% of the trained model's gain (1.77 vs 1.44) and has no
  long context. The gradient edit keeps the better model but moves predictions everywhere (mean abs 0.32
  nats per position, net +0.03).
- E2 failed because a 3-character context is not a word: "kin" also ends making, taking and lacking.
  Choosing the context is the hard part of the edit, and a measured check caught the mistake.
- Sample counts are small (tens of words in 20k characters); the per-position numbers carry the result.
