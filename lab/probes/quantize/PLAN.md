# What 2 bits cost the 9B, and whether the conclude edit survives (roadmap gap 31) — plan fixed before the run

**Op.** `quantize` (WSP-Lang 0.8.3, `engine/quantize_op.py`, 13 tests first; real smoke on the 0.8B: Q6_K → Q2_K in
seconds, manifest written). Source: our Q6_K files with `allow_requantize=true` (no BF16 GGUF of the 9B is local; the
manifest records the caveat). Importance matrix for the IQ2 types: `llama-imatrix` on WikiText-2 **train**, 100 chunks
of 512, computed on each source model.

**Arms** (outputs in `hf_cache/quant/`, manifests beside them):
| arm | source | type | imatrix |
|---|---|---|---|
| `base_q2k` | `hf_cache/h4/A0-Q6_K.gguf` | Q2_K | none |
| `base_iq2xs` | A0 | IQ2_XS | auto |
| `compiled_iq2xs` | `hf_cache/cc_op_v2/merged-Q6_K.gguf` | IQ2_XS | auto (on the compiled model) |

**Measures.** `dist-shift` (WikiText-2 test ppl, KL and top-1 against the A0 base's saved logits, HellaSwag /
Winogrande / MMLU 400) for the three; `rank-2048` (GSM8K, MATH-500, HumanEval 164, TriviaQA 200, compliance) for the
three, with `provenance=` the quantize manifest. References: A0 base and the compiled Q6_K runs already on both
suites (§45.31, §45.36). About 8 GPU h. Chain `lab/runs/quant/chain.sh`, queue 9 after the HumanEval-recovery chain.

**Readings (fixed).**
- **Q1 cost of 2 bits on the base**, per axis: Δ against A0 base (ppl ratio, KL, top-1; GSM / MATH / HumanEval /
  TriviaQA / compliance with item-paired p).
- **Q2 imatrix helps:** `base_iq2xs` (≈ 2.3 bpw) has lower KL than `base_q2k` (≈ 2.6-3 bpw mixture) despite fewer bits.
- **Q3 the edit survives:** survival = (compiled_iq2xs − base_iq2xs) / (compiled Q6 − base Q6) on GSM8K; the edit
  "survives" if ≥ 0.5 and the difference is p < 0.05 on both graders.

**Prediction:** `predictions.md` row 53, written before the run.

**Not done here.** Quantising from BF16 (needs the official BF16 GGUF, ~18 GB, driver's call); per-tensor protection
(`tensor_type=` Q6 on the edited `down_proj` rows) and re-injecting the edit after quantising: next arms if Q3 fails.
