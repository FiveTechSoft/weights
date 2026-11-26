# Semantic Bridge V1 — functional reference, not internal weight editing
Goal: frozen language model selects an installed compiled skill from unseen
natural-language requests, supplies arguments, and uses its actual result in
a subsequent decision. No new arithmetic operations; reuse V3 tensors and
Coprocessor.execute. The bridge uses explicit generated calls and text return;
this is a reference for later internalization, not a claim of native surgery.
Use Qwen/Qwen2.5-0.5B-Instruct as the language interface (instruction-tuned
checkpoint, NOT the original base checkpoint); compile with its tokenizer.
Record exact revision and base digest. No gradients, no donor, no formula
in the interface prompt. Skill descriptions identify four fictional services:
warehouse stock check SUM10, delivery route check MIX10, invoice reference
check COPY10, shipment seal check SPC10. User does not supply skill identifiers.
Model emits exactly NAME(a,b) or NONE. Strict full-match parser accepts only
installed names and decimal arguments 0..999; invalid output fails closed.
No extraction from user text by host; no repaired calls, constrained decoding,
or oracle routing in candidate. Use V3 circuit.execute on one-hot last digits
of generated arguments; assert unique finite one-hot result; no simulator in
candidate execution. Feed actual digit back to frozen LLM, ask for a final
decision given user's threshold (APPROVE if result >= threshold, REVIEW
otherwise), exact format digit and decision. No host decision in candidate.
Fixed prompts and demonstrations (synthetic interface examples, not sealed).
Sealed: 160 positive requests (4 skills x 5 held-out wording families x 8
distinct random 100..999 pairs), thresholds 1..9, seed SEED+61; 40 negatives
including unrelated, mention without request, missing operand, unsupported
service; report each family and negatives. English only, no chat history.
Compare candidate to oracle-call ceiling (gold call only diagnostic), no-result
control (same generated call but result withheld), and wrong-result control
(result shifted by 5 mod 10, same generated call). Controls use identical
second-stage prompt except result. Report routing, argument exactness,
execution, final digit, decision, end-to-end strict joint (route+args+answer),
negative invocation rate, counterfactual adherence (on valid calls where
decision changes, both result and decision must follow returned result).
Gates: routing >=95%, joint call >=95%, exact circuit >=99% among correct
calls, strict end-to-end >=95%, negatives <=1%, candidate answer accuracy
>=20pp above no-result control, counterfactual adherence >=95%, original
digest exact. Report denominators (zero denominator fails). Preservation
outside bridge is architectural bypass, not a broad capability claim.
Save tensor artifact and reload before evaluation. All cases and generations
saved. CPU CI, fixed greedy generation, no retry or post-hoc prompt tuning.
If failure, diagnose routing vs return via oracle ceiling; do not add ops.
