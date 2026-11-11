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
- [rev] **Gate 3 has no target yet** (see the gate 3 addendum: "happy" chosen): cc08 (§43.22) could not re-derive "conclude" in the 0.8B (1/256 traces close),
  so a ported "conclude" has nothing to be compared with in the destination. Gate 3 needs another direction the
  0.8B does have, or a different pair of models.

## Cost
Capture: one forward pass over 240 short texts per model (the 9B streams layer by layer through the GPU, as
`contrast_vector`); disk up to 1.5 GB (0.8B) + 7.7 GB (9B). Fit: CPU, minutes to tens of minutes.
[rev] On 2026-11-11 C: had 1.6 GB free: the captures wait for disk space (user's decision on what to remove).

## Not verified when drafting
`fit` and its pieces are tested (12 tests, including an end-to-end run on synthetic captures where related models
pass and unrelated ones fail; they pass on this machine too). `capture` has not run yet.

## Gate 2 (addendum 2026-11-11, written after gate 1 passed and before any gate 2 code; user: "adelante con la puerta 2")
**Question.** Gate 1 says the map predicts the target's states (R²). Gate 2 asks whether the target can *use* the
mapped states: run the source up to hidden state i, map it to the target's hidden state j with the gate 1 map
(saved, not refitted), put it into the target's residual stream at j and let the target finish. If the map only fits
the variance and misses what later layers read, the stitched model's next-token loss is no better than feeding the
target just the token identity.

**Setup.** Same 40 test texts as gate 1 (never used to fit any map), same token ids (re-tokenized and checked against
the gate 1 hash). Source states come from the gate 1 captures (float16). The stitch replaces the output of target
layer j−1 (= hidden state j) at every position but the first (gate 1 does not store position 0; there the target
keeps its own state: one token of ≤ 128, the same in every arm). Depths, fixed: j at ¼, ½, ¾ of the target's
states: 9B j ∈ {8, 16, 24} (33 states), 0.8B j ∈ {6, 12, 18} (25 states); source i = the gate 1 CKA-best state.

**Arms per (direction, j)**, scored by mean next-token NLL over the stitched positions of the test texts:
- `native`: the target alone (once per direction);
- `stitch`: the gate 1 map from the CKA-best source state;
- `emb`: control, a map from the source's embeddings (token identity only), same ridge rule, fitted now on the
  gate 1 fit split;
- `shuffle`: control, the same source state mapped by a ridge fitted on permuted rows;
- `mean`: control, the target's fit-split mean state at every position (context erased at depth j);
- `self` (mechanics check, second method): the target's own captured state j (float16) injected through the same
  hook. Must give the native NLL within 0.01, or the hook is wrong and nothing else is read.
Also reported: the source model alone (its own NLL) and the share of positions whose top-1 token equals the
native target's (KL dropped before coding: storing native distributions over a 248 k vocabulary is ~2.5 GB).

**Score.** Gap recovered G = (NLL_emb − NLL_stitch) / (NLL_emb − NLL_native): the share of what the target's context
adds at depth j (over token identity) that the mapped source state carries. 95 % interval by bootstrap over test
texts (2 000 resamples, seed 2026).

**Pass rule per direction (fixed now).** At ≥ 2 of the 3 depths: G ≥ 0.5, the bootstrap lower bound of
NLL_emb − NLL_stitch > 0, and NLL_stitch < NLL_shuffle and < NLL_mean. Both directions are run.
- 0.8B → 9B is the direction this plan named for gate 2 ("run the 0.8B up to state i … continue in the 9B").
- 9B → 0.8B is the one gate 3 needs (a direction measured in the 9B written into the 0.8B). Gate 3 is only planned
  if this direction passes.
- Prediction (ours, low confidence): G 0.3-0.6, best at ¼ depth and worst at ¾ (gate 1's R² gain over embeddings
  is ~0.3 and later states carry more of what the next token needs); 0.8B → 9B passes at most at one depth.

**Limits.** Same domain as gate 1 (GSM8K text, 128 tokens). Stitching a whole state is a harder test than gate 3
needs (gate 3 moves one mean difference, which gate 1 already transports with cos ≈ 0.99, likely partly position);
a gate 2 failure does not by itself kill gate 3, but then gate 3 needs its own content contrast and a causal check.

**Cost.** GPU, one job: the 9B streams layer by layer (12 GB card), ~13 forwards over 40 × 128 tokens per
direction; the 0.8B fits. Code `lab/probes/act_map/stitch.py`, tests in `tests/test_act_map.py` first.

## Gate 3 (addendum 2026-11-11, written after gate 2 passed and before any gate 3 code; user: "a")
**Question.** Can a steering direction measured in the 9B be written into the 0.8B through the gate 1 maps and steer
it, about as well as the 0.8B's own direction for the same thing? Direction: **"happy"** (results §43.13: passes on the
9B with judge and lexicon agreeing, placebo flat). "Conclude" is out: the 0.8B has no own version to compare with
(§43.22).

**Vectors (all from the same 70 pair texts, `lab/runs/happy/pairs/{pos,neg}.jsonl`, written by the 9B in §43.13;
response tokens averaged per hidden state, as `engine/contrast_vector.py`):**
- `own`: the 0.8B's own contrast D₀.₈(j), j = 0..24, from its own forwards over those texts (the reference). Same
  texts in both models, so own and ported measure the same token-level difference; the 0.8B's own *answers* are not
  used (stated limit, below).
- `port`: the 9B's contrast D₉(i) (recomputed: §43.13 stored only increments from layer 1, which lose D(0), D(1)),
  mapped per target state j with the saved gate 1 map from its CKA-best source state: D_port(j) = D₉(i(j)) · W_j
  (intercepts cancel in a difference). Not refitted on chat text: the maps were fitted on GSM8K raw text, so this
  also tests whether they hold out of their domain.
- `embport`: control, the 9B's **embedding** contrast D₉(0) mapped through an embeddings ridge per target state j
  (same rule as gate 2's `emb` arm, gate 1 fit split): what token identity alone carries across. Happy text differs
  from neutral largely in word choice, so this control can be strong; it separates "the map carries context" from
  "the map carries which words were used".
- `placebo`: `port` with the same norm per layer in random directions (`contrast_vector.placebo`, seed 2026).
All become per-layer increments (`contrast_vector.increments`) in a llama.cpp control vector for the 0.8B Q6_K
(`hf_cache/qwen35-08b-gguf`).

**Step 0 (forwards only, no generation; reported, not a criterion).** Cosine per target state j between D_port(j)
and D₀.₈(j), and between D_embport(j) and D₀.₈(j). The gate 1 analogue on chat text.

**Step 1 (behavior), protocol of §43.13 unchanged** except the model: 0.8B Q6_K, no system prompt, thinking off,
greedy, 200 tokens; the same 10 dev and 40 test prompts; judge MiMo (tone 1-5, coherent yes/no, verdict read after
`</think>`, truncation announced); second method the same lexicon (positive-affect words + "!" per 100 words).
- Judge validity on the 0.8B, fixed now: base coherent in ≥ 9/10 dev. Otherwise the judge is not valid for this model
  and gate 3 stops before reading any arm (no retuning of the judge on test).
- Scale per arm (`own`, `port`, `embport`) chosen on dev by the §43.13 rule: largest of 0.125 / 0.25 / 0.5 / 1 / 2 / 4
  with ≥ 9/10 coherent; none → that arm FAILS. `placebo` runs at `port`'s scale.
- Test arms: base, own, port, embport, placebo.

**Reference first.** `own` must pass §43.13's C1-C4 on the 0.8B (tone +1.0 over base with sign p < 0.01; lexicon
+2/100 words with p < 0.01; coherent ≥ 36/40). If it does not, the 0.8B has no usable "happy" direction of its own,
there is nothing to compare with, and gate 3 is **not decided** (written as such, not as a fail).

**Pass rule for `port` (fixed now; all must hold):**
- P1 tone ≥ +1.0 over base, paired sign test p < 0.01;
- P2 lexicon ≥ +2 per 100 words over base, p < 0.01 (second method agrees);
- P3 `placebo` tone gain < +0.5;
- P4 coherent ≥ 36/40;
- P5 `port` tone gain ≥ 0.5 × `own` tone gain (it does at least half of what the native direction does).
P1-P5 → **gate 3 passes**: a direction measured in one model steers another of the same family through the map.
Then, read separately, P6: `port` tone gain exceeds `embport`'s by ≥ +0.5. P6 fails → the transfer works but token
identity alone already carries it; "the map carries context" is then not shown for this direction.
Reported, not criteria: GSM8K fast set (48) with each vector against the 0.8B base, cut answers, step 0 cosines.

**Prediction (ours, low confidence).** Step 0: median cosine port/own ≥ 0.7; embport close to it in early states and
lower after ½ depth. `own` passes. `port` passes P1-P5 at a smaller scale than `own`. P6 fails: "happy" is mostly
words.

**Limits.** One direction, one model pair, one family and tokenizer. Tone is not a capability. Own and ported are
measured on 9B-written texts; the 0.8B answering with a "happy" system prompt might give another direction (not
tested). Maps fitted on 128-token GSM8K text, used on chat responses up to 384 tokens. llama.cpp steers layers 1..23
only (layer 0 cannot be steered), same for every arm.

**Cost.** Forwards: the 9B streams over 140 texts once, the 0.8B fits (minutes). Generation: 0.8B llama-server, dev
3 arms × 6 scales × 10 + base 10, test 5 × 40; judge MiMo over ~420 answers; one GPU job at a time, chained by
script. Code `lab/probes/act_map/port.py` (vectors, step 0) with tests first; `happy.py` gets a runs-directory option
so the 0.8B results go to `lab/runs/act_map/happy08/` and §43.13's files stay untouched.
