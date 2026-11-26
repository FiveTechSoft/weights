# V10 — coprocessor with an explicit interface trained over the target numeral range

V6's success was in-distribution only (0..49): 100% on the training surface,
10.5% on novel 100..199 pairs (V8, my V10 measurement). The same interface
direction was trained on 1-2 digit numerals and evaluated on novel 3-digit
numerals — a mismatch the architecture cannot absorb.

V9 showed the same failure with a last-layer native expansion (fail-closed).

V10 is diagnostic on the coprocessor architecture, not a new native edit.

## Contract
- Frozen Qwen2.5-0.5B (in-distribution semantics unchanged).
- Skip branch (V6 architecture): hidden state after layer 2, categorical
  readouts for a mod 10 and b mod 2, trigger, 20-rule circuit, logit injection.
- Training range: MC pairs with operands 0..99 (2-digit numerals); disjoint
  from the test.
- Test 1 (in-distribution): 200 novel pairs from the same 0..99 range (the
  eval set). 
- Test 2 (out-of-distribution): the novel 100..199 pair set from V8 (3-digit
  numerals), seed SEED+8, 200 pairs.
- Dose selection by dev: dev skill >= 0.95, preservation (arithmetic and NLL,
  relative tolerances as in V6). Report both sealed arms.
- Preservation gates unchanged from V6: original weights intact, zero parity,
  arithmetic within 2pp, prose within 0.1 nats.
- Pre-registration here; no V9-style native edit, no re-interpretation of any
  earlier result.

## Honest claims
- V10 tests whether the coprocessor's interface can generalize when trained
  over the target numeral surface.
- Failure modes: a ridge op over 2-digit+ numerals will still not transfer to
  3-digit numerals; that would confirm the coprocessor needs an explicit
  token/char interface, not a hidden-state readout.
- This is a diagnostic. No "skill" claim accompanies this unless the 100..199
  sealed arm clears the pre-registered gates.
