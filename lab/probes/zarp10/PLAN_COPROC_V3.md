# Coprocessor V3 — generic register-compiler integration

V2 is hardcoded: NAMES / TEMPLATES / SOURCES are three literals; the expected
answers are hand-coded per skill; the source files exist but the code does not
compile them — it asserts they match the literals.

V3 makes the coprocessor a true register compiler:

## Architecture (unchanged)
- Frozen Qwen2.5-0.5B. Skip branch at output: prompt tokens -> one-hot circuit
  -> digit-parallel residual injected onto the digit logits.
- Compiled weights: exact zero-dose parity, original weights intact, dose 4.

## What changes
- New spec file: programs/register/coproc_v3.json
  * skills: [name, source, templates] — any .wspreg file, any template pair.
  * The spec is the only source of truth for what the coprocessor supports.
- Generic reference evaluator computes labels from the wspreg source itself
  (same modulo-10 semantics), not from hardcoded formulas.
- Generic pattern generation: one pattern set per skill × template.
- Runtime loads the spec, compiles each program via the same one-hot SSA
  circuit (the register-compiler semantics), evaluates, reports.
- One new skill: SPC10, 2x = ADD x x / 2x + y = ADD a y / RETURN b — fresh,
  with no hardcoded label formula.

## Evaluation
- Sealed: 200 pairs per skill (100..999), both templates; mid: 100 pairs (50..99).
- Circuit-exact + accuracy gates as in V2 (>=95% per group, >=99% exact).
- Preservation gates unchanged from V2: zero parity, arithmetic retention,
  NLL retention, original weights intact, negatives no-activation, dose 4.
- Replacement: compile a NEW program COPY10-with-a-new-instruction mid-run
  (same runtime) and show 100% on changed skill, 100% exact unchanged.
- Honest non-claims: no semantics routing; no free-form skill acquisition.
