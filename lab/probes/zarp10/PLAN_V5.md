# V5 — operand interface diagnostic, not a new skill claim
Use Qwen2.5-0.5B frozen, resolved revision. No gradients.
Use 600 discovery pairs from permutation indices 700:1300, dev 1300:1500,
fresh sealed 2100:2300. Same formula and prompts as V4.
Inspect token offsets to locate last decimal digit of each operand; assert
each selected token encodes the expected digit. This is an offline diagnostic
using known positions, not an inference-time mechanism.
At embeddings and each transformer layer, fit categorical linear ridge
readouts of a mod 10 and b mod 2 separately at operand positions and at
final prompt position. Fit only discovery, regularization fixed .001 times
mean Gram diagonal. Also fit scalar readouts for comparison to V4.
Select best layer per interface by joint dev accuracy, ties earliest.
Evaluate selected interfaces on sealed 200 examples; report all dev curves,
raw sealed predictions, tokenization coverage, joint and formula accuracy.
Include analytic embedding decoder from ten digit token embeddings, no
examples; validate that decoder on actual operand tokens. This establishes
whether token-local data are recoverable, not whether Qwen can route them.
Unit-test V4 SwiGLU arithmetic with exact synthetic operand values for all
20 residue pairs and report correct output and off-trigger residual.
Gates: local joint >=99%, synthetic arithmetic >=99%; final-position joint
>=95% is a separate transport gate, not required for local success.
No changes to Qwen, no claims of new integrated capability. If local succeeds
but transport fails, next build dedicated attention routing; do not repeat
blind last-layer editing. No answer labels in fitted readouts.
