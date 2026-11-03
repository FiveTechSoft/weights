# Measurement correction, not a model improvement

Base: FiveTechSoft/weights 5a42cf21092827317d0e3c9d23a97e6685b221d1.
No model was loaded, edited, trained or queried. No new benchmark score was produced.

Changes:
- Preserve grouping when flattening LaTeX fractions, including nested fractions.
- Preserve root index. Cube and square roots are no longer equated.
- Do not remove comma separators inside tuples/lists as if they were thousands.
- Read final Respuesta: lines, as requested by the legacy runner's own prompt.
- Use the shared scorer in the legacy runner; use repository-relative paths.
- Replace a fabricated model-alias hash with optional full local-file SHA-256.
  This identifies the file only, not what a server actually loaded.
- Use MATH subject metadata rather than categorizing everything as GSM8K.
- Correct sampled-mode documentation: it does not perform three-seed evaluation.

Executed:
    python rig/math500.py --selftest
    python -m unittest discover -s tests -v
Result: 44 existing assertions and six regression test methods pass.

Previously published r0-orig-math.json contains 12 answer records and one summary
object, NOT 13 problems. Four of the 12 were scored correct by the old scorer.
Replaying only published final-answer tails now scores ten of 12; the other two
have empty tails. This is an audit of existing answers, not a fresh baseline,
not a model improvement and not a usable estimate of general mathematical skill.
The original file remains untouched.

Limitations:
This normalizer is still not a full mathematical equivalence judge. Conservative
false negatives remain possible. Nested radicals, symbolic identities, ambiguous
bare comma lists and unit semantics need independent handling. Do not certify a
new model improvement solely from these unit tests. Compare the exact original,
candidate and requantization placebo on the same held-out problems and generation
settings. Audit disagreements. Report resource errors and truncation separately;
never silently drop them from the final comparison. The legacy runner's advertised
split/sealing and sampling behavior still needs verification before a full study.
The local-file hash does not authenticate the server's loaded weights. Existing
hardcoded quantization labels are not proof of a model's actual format.

No push or other remote mutation is part of this patch.
