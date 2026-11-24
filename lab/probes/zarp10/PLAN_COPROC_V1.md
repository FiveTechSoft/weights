# Coprocessor V1 — explicit token interface, not native weight surgery
V8 failed. Do not rerun native residual transport. Use frozen Qwen plus persistent
tensor-only token-level coprocessor with direct digit logit output. No runtime
regex, integer parsing, answer lookup by operand pair, gradient, donor, hidden
state fitting, or external tool call. This is an explicit finite grammar, NOT
free-form language comprehension or standard Qwen checkpoint compatibility.
Compile two commands ZARP10(a,b) = (3a+5b+7)%10 and
DAX10(a,b) = (7a+3b+2)%10, rendered as "NAME(a,b) = " or
"Apply NAME to a and b. Answer: ". Decimal widths 1..3, tokenizer-derived fixed
token pattern, digit slots use ten one-hot digit embeddings. The only input
interface is input_ids/attention_mask. At every position, match complete
prefix from first unmasked token, using tensor products of fixed token and
digit indicators; route final digits into 10x10 conjunctions, compile rule
matrix analytically. No arbitrary prefix accepted; no learned semantic routing.
Patterns derived offline from digit-1 exemplars; verify exact tokenizer coverage
on dev/test, report failures rather than repair on sealed. Store pattern tensors,
digit lookup and rule in safetensors; runtime only tensor operations. Cache
disabled; evaluate next-token prediction, not multi-token generation.
Dev seed SEED+20: 100 random distinct pairs 0..49 per skill (both templates);
choose common dose [0,4,8,16,32,64] maximizing worst skill/template accuracy
subject to relative arithmetic loss <=2pp and prose NLL increase <=.1.
Sealed seed SEED+21: 200 random distinct pairs 100..999 per skill, both
templates (400 per skill, 800 total), unseen numeric values, no fit. Also 100
pairs 50..99 per skill, both templates, separate diagnostic. Negative commands
OTHER10, NOT prefix, malformed delimiter, trailing suffix, ordinary arithmetic,
prose: measure exact zero residual on every valid position, compare base logits
on negative commands. Scope grammar explicitly; no claim arbitrary language.
Compare base, candidate, cyclic wrong-output placebo (same router/dose), reload
from separate base plus tensor artifact. Gates simultaneous: every skill/template
sealed >=95%, aggregate >=20pp vs base and placebo, every sealed router correct
>=99%, no wrong-command activation, arithmetic loss <=2pp, NLL increase <=.1,
nonzero dose, zero-dose parity exact, immutable base digest. Report per-skill,
per-template, operand widths, raw counts. Full original model remains frozen.
Success only means executable tensor coprocessor for explicit token grammar.
