# `.wspngram`: n-gram programs and the nanoGPT decompiler

A `.wspngram` file is a WSPLang program that `step synthesize method=ngram table=<file>` compiles into nanoGPT
weights without gradients, and that `step decompile` produces from weights.

```
# WSPLang n-gram program: compile with step synthesize method=ngram table=<this file>
ngram vocab=65 order=4
provenance distilled_from:gradient_descent
chars "\n !$&',-.3:;?ABC...xyz"
unit "" -4.1 -1.9 ...                 # top: " " +2.10, "e" +1.62, "t" +1.03
unit "th" ...                          # top: "e" +2.83, "a" +1.40, "i" +0.95
```

- `unit "<context>" <vocab numbers>`: log-probability correction added on top of the unit for the context one
  character shorter. A context's prediction is the softmax of the sum along its suffix chain, so an unseen context
  backs off to its longest listed suffix. The list must be suffix-closed.
- Units are in priority order. When the architecture has fewer MLP units than the program, the first ones are kept.
- The `# top:` comment is for people; the parser ignores it.
- `provenance` says where the numbers came from: `closed_form` for tables from data, the source checkpoint's
  provenance for a structural decompile, `distilled_from:<provenance>` for a behavioral one. `synthesize` writes
  `closed_form` only when the program is `closed_form`; otherwise `closed_form_compile_of:<provenance>`, so a
  model distilled from a gradient-trained one never passes as gradient-free.

## Two decompilers

| `method=` | Input | Output | Exact? |
|---|---|---|---|
| `structural` | Weights written by `synthesize` | The program that was compiled | Yes: recompiling with the same seed gives the same weights within 1e-6 (`tests/test_wsp_decompile.py`). No data is used. |
| `behavior` | Any nanoGPT, including gradient-trained ones | For each context of fewer than `order` characters, the model's average prediction over its occurrences in train | No: anything the model computes from older context is lost |

`structural` refuses weights that do not follow the compiler's layout, such as any trained model, with a
`DecompileError` naming the first check that failed.

## How much of a trained model survives

Shakespeare char, full val, unigram entropy 3.3373; retained = share of the teacher's gain over unigram.
`exact` is the 6L/384/1536 shape (10.7M parameters); `wide` holds every unit in one wide MLP.

| Teacher (val) | order 4 exact | order 5 exact | order 5 wide | Gate |
|---|---|---|---|---|
| nanoGPT 10.7M (1.4573) | 1.8064 (81%) | 1.7810 (83%) | 1.6347 (91%) | FAILS (L2 0/3) |
| AdamW 2L/d64, 0.1M (1.7153) | 1.8178 (94%) | 1.7936 (95%) | 1.7149 (100%) | FAILS (L2 0/3) |
| Witten-Bell from data (control) | 1.8005 | 1.7941 | 1.6776 | FAILS |

Reproduce with `python lab/verify_wsp_decompile.py` (results in `lab/runs/wsp_decompile_round_trip.json`).
