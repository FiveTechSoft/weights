# Coprocessor V2
Compile .wspreg sources using COPY ADD SUB (modulo 10), RETURN, x/y last
decimal digit inputs. Reuse existing SSA parser; LOOKUP explicitly unsupported
until a record interface exists. Each operation becomes a dense tensor
projection on one-hot register banks. ADD/SUB use pair outer products and
fixed arithmetic maps, COPY uses identity. No whole-program input/output
table, no runtime opcode dispatch, no source parsing at inference.
Fixed grammar NAME(a,b) = or Apply NAME to a and b. Answer: ; widths 1..3.
Compile three new programs from files; independent reference expressions.
Artifact stores grammar patterns, register circuit matrices and output selector.
One generic runtime loads all programs; verify replacement of one program
under the same command without changing runtime (independent test).
Base frozen, no gradients/donor. Dose 4 fixed from V1, no retuning.
200 distinct pairs 100..999 per program, both templates, seed SEED+31.
All 6 groups >=95% unrestricted next-token accuracy and >=99% exact circuit
output; mean >=20pp above base and shifted-output placebo. Replacement test
100 distinct pairs on changed command >=95%, unchanged commands circuit exact.
No activation on unknown command, NOT prefix, malformed format, arithmetic
and prose (final position for suffix cases). Exact zero-dose parity, base hash
unchanged, arithmetic loss <=2pp, NLL increase <=.1. Save/reload tensors.
Report independent per-operation ablation on circuit only (zero one stage),
runtime source hash, artifact size, per-group results. Diagnostic ablation
must reduce at least one nontrivial program by >=20pp. This is fixed grammar
tensor programming, not native surgery or arbitrary semantic routing.
