# Row 54: why 2 bits loop, and what the conclude edit adds beyond breaking loops — plan fixed before the run

Follows row 53 (§45.38): from Q6_K, Q2_K without imatrix does not answer; IQ2_XS loops (GSM 4/100, 96 cuts); the
compiled conclude edit at IQ2_XS gives GSM 36 (survival 0.91) but every gain is a recovered cut, as in Q6. Three
questions left open there, plus a control for the decoding change.

**Arms** (all `rank-2048`, one server per arm, port 8095; chain `lab/runs/quant/chain54.sh`, log `chain54.log`):
| arm | model | runtime | question |
|---|---|---|---|
| `q2k_imx` | `hf_cache/quant/qwen9b-Q2_K-imx.gguf` (built 23:22: Q2_K + the A0 imatrix of row 53; per-tensor type mix identical to row 53 `base_q2k`, 442/442 tensors, 3.405 bpw) | — | QD; also `dist-shift` |
| `q6_lora` | A0 Q6_K | `--lora hf_cache/cc_op_v2/vector/lora_r1.gguf`, scale 1.0 per request | denominator for QC |
| `iq2_lora` | `qwen9b-IQ2_XS.gguf` (row 53) | same LoRA, scale 1.0 | QC: the edit re-injected in fp16 after quantising |
| `iq2_dry` | `qwen9b-IQ2_XS.gguf` | DRY 0.8 / 1.75 / 2 in `extra_body` | QA: loop control without the edit |
| `compiled_iq2_dry` | `compiled-IQ2_XS.gguf` (row 53) | same DRY | QB: the edit under the same loop control |
| `q6_dry` | A0 Q6_K | same DRY | cost of the decoding change at full precision |

DRY values are the sampler author's defaults (multiplier 0.8, base 1.75, allowed length 2), fixed a priori, not tuned
on any scored item. Temperature 0 as the suite. **Guards run before this plan (23:30):** on an invented word problem
(not in any suite), the IQ2_XS server with the adapter gives different outputs at LoRA scale 0 and 1 (adapter live)
and DRY turns a 1 500-token cut into a 983-token stop with `\boxed{5}` (the penalty reaches the sampler).

**Dose caveat.** The Q6 merge of the edit lost part of it (merge rel err 0.33-0.60 per layer, `cc_op_v2/report.md`);
the fp16 LoRA is the fitted edit before that loss, so `iq2_lora` carries a larger dose than `compiled_iq2xs`. QC is
read against `q6_lora` (same dose, full precision), not against the merged Q6.

**Readings (fixed; GSM8K, item-paired, exact binomial, both graders must agree):**
- **QA loop control alone:** `iq2_dry` vs row 53 `base_iq2xs`. If `iq2_dry` GSM ≥ 36 (the compiled 2-bit score), loop
  breaking by decoding recovers at least what the edit recovered.
- **QB the edit beyond loop control:** `compiled_iq2_dry` vs `iq2_dry`. The edit "adds" only if more GSM items are
  won than lost with p < 0.05 on both graders; otherwise at 2 bits the edit is a loop breaker and nothing more.
- **QC re-injection:** survival_lora = (`iq2_lora` − `base_iq2xs`) / (`q6_lora` − base Q6) on GSM; and `iq2_lora` vs
  `compiled_iq2xs` (row 53) paired: re-injection "beats merging then quantising" if p < 0.05 on both graders.
- **QD imatrix isolated:** `q2k_imx` vs row 53 `base_q2k` on `dist-shift` KL (same type mix, so the imatrix is the
  only change). "Helps" if KL is lower by more than 5 standard errors.
- **Decoding cost:** `q6_dry` vs base Q6 on GSM, MATH, HumanEval, TriviaQA (paired). If `q6_dry` loses GSM with
  p < 0.05, QA and QB are read as "loop control at a cost", not as free.
- Accuracy among finished items reported for every arm (the row 53 lesson: gains can be recovered cuts only).

**Prediction:** `predictions.md` row 54, written before the run.

**Order** (information per GPU hour): `q2k_imx` dist-shift (minutes), `q6_lora`, `iq2_lora`, `iq2_dry`,
`compiled_iq2_dry`, `q6_dry`, `q2k_imx` rank. About 10 GPU h.

**Not done here.** `dist-shift` for the LoRA and DRY arms (`dist-shift` has no `lora=` path, and DRY is a sampler,
not a distribution change). Quantising from BF16 (arm 4 of the proposal): `hf_cache/qwen35-9b-hf` (19 GB safetensors)
can be converted locally, no download, but the BF16 GGUF is about 18 GB more on a disk with 47 GB free and a past
commit-limit failure (memory note); deferred to after this chain, the user's call on disk. Protecting the edited
`down_proj` rows at Q6 inside the 2-bit file: next if QC shows re-injection helps.

**Machine note.** Row 53's chain was killed at 09:35 on 2026-11-20 by a bugcheck 0x70C under GPU load (minidump
`C:\WINDOWS\Minidump\112026-33781-01.dmp`), not by an update; chains resume from `.partial.jsonl`.
