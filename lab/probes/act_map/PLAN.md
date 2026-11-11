# Gate 1 of cross-model porting: is there a linear map between two models' residual streams? (2026-11-11; drafted by the user's outside reviewer, approved by the user, reviewed here before any run; amendments marked [rev])

**Why.** `subspace_transfer` and `delta_to_lora` move pieces between models with the same base (same coordinates).
Between different models the weights have no shared coordinates (results §23: Procrustes on token embeddings,
rejected). The remaining route is activations: both models read the same tokens, so their residual streams can be
regressed on each other. If a linear map exists and carries contrasts, a steering direction measured in one model
can be written in the other (gate 3), and later the residual side of a low-rank adapter (gate 4).

**Scope of this gate.** No benchmark, no edit, forwards only. Qwen3.5-9B (`hf_cache/qwen35-9b-hf`) and Qwen3.5-0.8B
(`hf_cache/qwen35-08b-hf`): same tokenizer, checked by the script (identical token ids or it stops). [rev] Checked
before writing: the two `tokenizer.json` files are byte-identical (sha256 5f9e4d4901a92b99…), vocab 248 320 both.

## Data
GSM8K-train rows 1000-1239 (question + solution, raw text, first 128 tokens): not the retention rows (0-299), not
cc's discovery rows (0-63), not any dev or sealed item (those are GSM8K test / MATH-500). Every token but the first of
each text is stored, per hidden state (0 = embeddings; the last one is after the final norm), float16.
Split by whole texts: last 40 = test, last 10 % of the rest = validation (ridge strength), the rest = fit.

## Measures (`act_map.py fit`)
1. **CKA table:** linear CKA between every hidden state of one model and every one of the other (2 000 fit tokens).
2. **Maps:** affine ridge, strength from {1e-4, 1e-3, 1e-2, 1e-1} · tr(XᵀX)/d on validation. For each target hidden
   state ≥ 1, three sources: the highest-CKA state, the depth-matched state, and the source **embeddings**
   (control: what the token's identity alone explains). Score: held-out R² over all dimensions.
3. **Shuffle control:** one map fitted on permuted rows, middle target state.
4. **Contrast transport:** on held-out tokens, the mean difference D between two groups, measured in the source,
   mapped (D·W; the intercepts cancel), against the same difference measured in the target; cosine. Groups:
   *late* (last quarter of its text vs the rest: context only) and *digit* (digit tokens vs the rest: mostly token
   identity, reported, not a criterion). Each also through the embeddings map.

Both directions are fitted. The one that matters for gate 3 is **9B → 0.8B**.

## Pass (9B → 0.8B, target hidden states 1..last; fixed now)
1. median held-out R² ≥ 0.5;
2. R² exceeds the embeddings control by ≥ 0.1 in at least 2/3 of the target states;
3. shuffle control R² ≤ 0.05;
4. *late* contrast cosine ≥ 0.7 in at least half of the target states.

All four: the map exists and carries a context contrast → gate 2 (run the 0.8B up to state i, map, continue in the
9B from state j, NLL against the 9B alone). Criteria 1-3 without 4: representations are similar but mean differences
do not transfer; gate 3 is not justified by this result. 1 or 2 fail: no linear map at this data size; before
concluding, one rerun with 4× the texts (stated now, the only rerun allowed).

## Prediction
Drafter's (low confidence): passes 1-3 in the middle of the stack (R² 0.6-0.8), weaker in the last states; *late*
cosine above 0.7 in more than half. The 0.8B → 9B direction scores lower R² (1 024 dimensions cannot span 4 096).
[rev] Ours: the same, plus the embeddings control is high in the first states (criterion 2 decided by the middle and
late states), and the CKA-best source tracks relative depth closely (|best − depth-matched| ≤ 2 states for most).

## Limits known now
- One text domain (arithmetic word problems). A map fitted here says nothing about reasoning traces; gate 3 needs a
  capture on traces (`--texts`).
- At most 23 k fit tokens against 4 096 source dimensions (n/d ≈ 5.6): the ridge strength matters; R² may rise with more text.
- A high R² does not make the map causal (results §23, point 4): that is what gate 2 tests.
- Same family and tokenizer. Another family needs token alignment, not written.
- [rev] R² over all dimensions can be dominated by a few high-variance residual channels; a pass driven by them would
  say little about the rest. Check the per-dimension picture before reading a pass as "the spaces correspond".
- [rev] **Gate 3 has no target yet:** cc08 (§43.22) could not re-derive "conclude" in the 0.8B (1/256 traces close),
  so a ported "conclude" has nothing to be compared with in the destination. Gate 3 needs another direction the
  0.8B does have, or a different pair of models.

## Cost
Capture: one forward pass over 240 short texts per model (the 9B streams layer by layer through the GPU, as
`contrast_vector`); disk up to 1.5 GB (0.8B) + 7.7 GB (9B). Fit: CPU, minutes to tens of minutes.
[rev] On 2026-11-11 C: had 1.6 GB free: the captures wait for disk space (user's decision on what to remove).

## Not verified when drafting
`fit` and its pieces are tested (12 tests, including an end-to-end run on synthetic captures where related models
pass and unrelated ones fail; they pass on this machine too). `capture` has not run yet.
