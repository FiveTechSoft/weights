# "Conclude" direction on a second family: Phi-4-mini-reasoning — plan fixed 2026-11-15 before any run (user: "adelante")

**Why.** Everything on "conclude" (vector §45.4, compiled weights §45.8, 8,192 budget §45.11/§45.13, nine placebos §45.10,
prompt control §45.9) is one model family (Qwen3.5-9B, greedy). FOUNDATIONS T8 lists "single model family, greedy only"
as the open limit that weighs most. Same recipe, other lab, other architecture.

**Model.** `microsoft/Phi-4-mini-reasoning` (3.8B, `Phi3ForCausalLM`, 32 layers, d 3,072, fused `gate_up_proj`, tied
embeddings; thinking as `<think> … </think>` emitted by the model after `<|assistant|>`). HF bf16 in
`hf_cache/phi4mr-hf` (7.2 GB, downloaded 2026-11-15), GGUF `hf_cache/phi4mr-gguf/Phi-4-mini-reasoning-Q6_K.gguf`
(unsloth, 3.1 GB). Served by the same llama-server as the Qwen runs.

## Stage 0: premise probe (≈ 20 min GPU; decides whether the rest runs; results reported either way)
P1. **Reasoning is parsed:** 5 GSM8K dev items + 5 MATH dev items through `bench_runner.ask` at 2,048: `reasoning` non-empty
    and `content` without `<think>` tags. If llama-server does not split them for this template, `bench_runner` gets a
    fallback (split `content` on `</think>`), with a unit test, **before** any arm; noted here as [rev].
P2. **The base over-thinks at the budget:** of those 10 items, ≥ 3 cut at 2,048 (`finish == "length"`). If ≤ 2, repeat the
    10 at 1,024; if still ≤ 2, **stop**: "Phi-4-mini-reasoning does not run out of budget at 1-2k tokens on GSM8K/MATH;
    the conclude question needs a model that does" (reported, no arms). The budget that shows ≥ 3/10 cuts is the budget
    of the whole study (2,048 preferred).
P3. **llama.cpp applies control vectors to this architecture:** a same-norm random vector (the placebo of step 3 below, or
    a throwaway vector built from 2 rows if the op has not run yet) at scale 8 changes the greedy output of 3 prompts
    against the base. If outputs are identical, stop: control vectors are not applied to `phi3` by this build.
P4. Tokenizer offsets (`return_offsets_mapping`) work for this tokenizer (the rows step needs them).

## Data (no download; nothing from the dev set or the sealed set)
`cc.py split` with `CC_RUNS=lab/runs/cc_phi`: same seed 20261110, so the same 64 discovery / 64 screen MATH items and
GSM8K **train** 0-63 as the Qwen study (none of them in dev or sealed sets; sharing them across models leaks nothing).

## Steps (mirror of `lab/probes/cc/PLAN.md`; the op and the server are the same code)
1. **Traces** (`cc.py traces`, `CC_HF=hf_cache/phi4mr-hf`): base Q6_K, greedy, 4,096-token cap, 128 discovery problems.
   Keep traces that close `</think>` **and** end with `finish == "stop"`. Need ≥ 30, else `extra` (GSM8K train 64-191), else stop.
2. **Rows** (`lab/probes/cc_phi/rows.py`, a thin wrapper over `cc.rows`): pos = last 48 reasoning tokens before
   `</think>`, neg = the reasoning before them (first 1,536 at most). **Phi difference:** the model writes the `<think>\n`
   tag itself (Qwen's template writes it), so the row's `response` is `"<think>\n" + reasoning` and the focus offsets are
   shifted by that prefix; the `think` flag stays (the op's `enable_thinking` kwarg is ignored by this template).
3. **Vector:** `contrast_vector` (`programs/cc_phi_vector.wspl`), placebo seed 2026, forwards on the GPU (3.8B bf16
   fits; the op streams layers anyway). [rev 2026-11-15, before any run: the WSP-Lang step has no `model_hint` parameter,
   so the GGUF carries the op's default hint string "qwen35"; it is metadata llama.cpp does not check. Left as is rather
   than changing the engine for a label.] Report per-layer increment norms and the last-row norm (Qwen: last row ≈ 2×
   the one before; if Phi's last row is > 4× or < 0.5× the one before, report as a profile difference, not a stop).
4. **Screen** (scale; screen MATH items, 2,048 or the stage-0 budget): base, vector at 0.25 / 0.5 / 1 / 2. Rule of
   `cc.py choose` (most correct, grader 1; ties → smaller; floor rule: if no scale beats the base in correct, most
   finished). No scale beating the base in correct **or** finished → **FAIL**, nothing goes to dev.
5. **Dev** (`bench_runner` defaults, GSM8K 0-99 + MATH select_indices(2026, 50); never used for any Phi decision): base,
   vector at the chosen scale, placebo at the same scale. One run per arm. **C1-C3 of `lab/probes/cc/PLAN.md`** verbatim
   (`cc.py verdict`).
6. **Sealed**, only if dev passes: `data/sealed_confirm.json` (its items touched no Phi decision; for this model it is a
   first use), base / vector / placebo, criterion of `lab/probes/sealed/cc_confirm.md`. Then stop; compiling into weights
   is a separate plan (Phi's `down_proj` exists; the fused `gate_up_proj` does not matter for a `down_proj` write).

## Prediction (ours, low confidence; arm and control; premises marked)
- Stage 0: P1 holds or needs the fallback (50/50); **P2 holds at 2,048** (the model's own card reports long traces;
  3-7 of 10 cut); P3 holds; P4 holds.
- Traces: 50-90 of 128 close within 4,096.
- Screen: base 15-30/64 MATH; best scale 0.5 or 1.0; vector at best scale base + 3 to + 12.
- Dev (2,048): base GSM8K 55-80 (cut 15-40), MATH 10-25 (cut 25-40); vector GSM8K **+5 to +15**, MATH **+5 to +12**,
  tokens −20 to −40 % on both, cuts at least halved; placebo within ± 3 of base on both, tokens within ± 10 %.
  C1 passes (MATH both graders or GSM8K both), C2 passes, C3 passes → PASS; sealed confirms.
- Premises: (p1) the GGUF is the same checkpoint as the HF weights (both from the release of 2026-11; `general.name`
  checked); (p2) llama.cpp's template for this GGUF matches the HF tokenizer's (`<|user|> … <|end|><|assistant|>`);
  (p3) grader 1 reads `\boxed{}` from `content` (if the answer lands inside the thinking only, the grader misses it —
  reported if ≥ 5 % of finished answers have no boxed content).
- **We would be surprised if:** the vector is below base by > 3 on GSM8K at the chosen scale (the direction hurts
  here); the placebo reaches vector − 3 on either benchmark (the push, not the direction); the base cuts < 10 % on both
  benchmarks at 2,048 (P2 fails: no over-thinking to fix); the chosen scale is 2.0 (Qwen needed 1.0 both times).

## Limits known now
One greedy budget, one second model, Q6_K; traces and dev items share the Qwen study's item lists (fine) but the
tokenizer and template differ (rows re-made); if P1 needs the fallback, the "reasoning" field is our split, not the
server's. Nothing here says anything about sampling.

## Cost
Stage 0 ≈ 20 min; traces ≈ 40 min (128 × 4,096 at 3.8B, 2 slots); op ≈ 10 min; screen 5 × 64 ≈ 1 h; dev 3 × 150 ≈ 1 h;
sealed 3 × 150 ≈ 1 h. ≈ 4.5 GPU h, after `cc08b_chain.sh`. Chain `lab/runs/cc_phi/chain.sh` with `set -e`, ERR trap,
smoke per arm, server-log check, GGUF sha256 in the log.
