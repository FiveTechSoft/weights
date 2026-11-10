# Is the "conclude" direction made of a few neurons? (cc in the neuron basis) — plan fixed 2026-11-11, before any run

**Why.** The donor-free "conclude" vector (results §43.17-18) works, and compiled as a rank-1 write per layer into
`down_proj` it keeps working (§43.19). Goodfire, *Neurons as an Interpretable Basis* (10 Nov 2026), reports on
Qwen3.6-35B-A3B that most MLP neurons have one dominant role, and that ablating weak activations makes reasoning loop
and stop concluding. Question: **is our vector's MLP part carried by a small set of `down_proj` columns (neurons), or
spread over thousands?** If few, the edit could be smaller and readable: scale or write a handful of columns instead of
a dense rank-1 term. If spread, the rank-1 write is the right object and the neuron story does not apply here.

## Phase A — exact decomposition (no benchmark; one GPU pass, after the sealed chain and the Turbo forensic end)

`hf_cache/cc/cc.gguf` holds v_N = D_{N+1} − D_N (`engine/contrast_vector.py`), D = mean residual over the `pos` rows
minus over the `neg` rows (`lab/runs/cc/{pos,neg}.jsonl`: last 48 reasoning tokens before `</think>` vs the earlier
≤ 1 536). Layer N's update is attention + MLP, and the MLP output is `W_N a` (W_N = `down_proj`, a = its input), so
**exactly**

    v_N = Δattn_N + W_N Δa_N,   Δa_N = mean_pos(a) − mean_neg(a)   (12 288 neurons per layer)

- Re-run the same pass as `contrast_vector` (same rows, same `max_resp`, same tokenizer and template, bf16) with hooks
  that also average the attention output and the `down_proj` input per layer. **Two-method check first:** the
  re-computed v_N must match `cc.gguf` (cosine ≥ 0.999 per layer) and Δattn_N + W_N Δa_N must equal v_N (relative
  error ≤ 1e-2, bf16). If either fails, stop and fix before reading anything.
- Per layer, report:
  - **MLP share** m_N = ⟨W_N Δa_N, v_N⟩ / ‖v_N‖² (the attention share is 1 − m_N).
  - **Concentration:** neurons ranked by their contribution c_j = Δa_j ⟨w_j, v_N⟩ / ‖v_N‖² (w_j = column j);
    share of m_N carried by the top k = 1, 8, 32, 128, 512, and k₅₀ / k₉₀ = smallest k reaching 50 % / 90 % of m_N
    (sum of positive c_j, sorted).
  - **Null for concentration:** the same numbers with Δa_N replaced by Δa computed on a random split of the same
    tokens into two groups of the same sizes (seed 2026) — what "concentrated" looks like by chance for this layer.
- Also reported (not criteria): the top-10 neurons of the 4-6 layers with the largest ‖v_N‖, with their Δa, mean
  activation on pos vs neg, and the 20 tokens (from the 28 held-out traces of §43.19) where they fire most.

**Reading rule (fixed now).** *Concentrated* if, over the layers that carry ≥ 70 % of Σ_N m_N ‖v_N‖², the median k₅₀
is ≤ 32 (0.26 % of 12 288) and at least 5× below the null's median k₅₀. *Spread* if the median k₅₀ ≥ 512. In between:
written as partial; no phase B.

## Phase B — only if *concentrated*: a few columns instead of a dense rank-1 term

Edit (same matrices, nothing added): for each layer, the top-k₅₀ neurons S_N of phase A; write ΔW_N = v_N u_Nᵀ with u_N
supported **only on S_N**, fitted exactly like `lab/probes/cc_weights/compile.py` (ridge to u ᵀx ≈ 1, same λ rule,
same 100/28 calibration split, layers in order). G0 as in cc_weights (median held-out |u ᵀx − 1| ≤ 0.5), else stop.
Placebo: same fit on |S_N| neurons drawn at random (seed 2026) per layer. Merged into Q6_K as in §43.19.

Dev (same items/server/budget as §43.17-19; base, vector and dense merged re-used): arms **sparse merged**, **random
sparse merged**.
- **B1** sparse merged > base, exact McNemar p < 0.05 with both graders on GSM8K or on MATH-500.
- **B2** random sparse merged not better than base.
- **B3** sparse merged not below the dense merged (§43.19) with p < 0.05 (grader 1) on either benchmark.
- If B1-B3 pass: a fresh sealed set 3 (made before that run, excluding everything in `lab/runs/`), arms base / dense
  merged / sparse merged / random sparse.

## Not done

No search over k, layers, λ or scale; no column *scaling* edits (scaling neurons that barely fire did nothing, README
"Evil Qwen" note). One model (Qwen3.5-9B, dense: 32 layers × 12 288 neurons, unlike Goodfire's MoE), one direction.

## Limits known now

Δa is a mean over tokens: a neuron can matter through rare large firings and still show a small mean (it would count as
spread here). The attention share is not decomposed further. SwiGLU "neurons" here are `down_proj` input coordinates
(act(gate) · up), the same unit as Goodfire's.

## Clarifications written before running (2026-11-11, with the script `decompose.py`)

- `cc.gguf`'s last increment (layer 31) is taken against HF's last hidden state, which is after the final norm; the
  attention + MLP identity holds for the raw layer outputs, so check 2 uses the layer outputs (hooks) and check 1 the
  hidden states (as `contrast_vector`). Layer 31's share against `cc.gguf` is reported but may not add up.
- The null's concentration is measured against its own MLP direction (W Δa_null), not against v_N: a random split has
  no reason to point along v_N, and the question is only how many neurons it takes to make *a* mean difference.
- Tokens assigned to the null groups independently with probability n_pos / (n_pos + n_neg) (expected sizes equal to
  the real groups, not exact).
