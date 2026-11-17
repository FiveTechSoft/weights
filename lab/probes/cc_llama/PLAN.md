# "Conclude" direction on a third family: DeepSeek-R1-Distill-Llama-8B — plan fixed 2026-11-17 before any run (user: "adelante")

**Why.** The compile primitive (`contrast_vector` → `compile_control_vector` into `down_proj`) holds on two families:
Qwen3.5-9B (§43.17-43.21, §45.8-45.13) and Phi-4-mini-reasoning 3.8B (§45.15-45.17). FOUNDATIONS T8 lists a third family
as the open limit after sampling. User (2026-11-16): verify the main advances on another LLM "so that we have verification
in three". Same recipe, same code, a third lab and a third lineage (Llama), with the two studies of the Phi plan
(vector, then compile) in one plan and one chain, each gated on the previous verdict.

**Model.** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (revision `6a6f4aa4`, `LlamaForCausalLM`, 32 layers, d 4,096,
`intermediate_size` 14,336, untied embeddings, bf16, 2 shards ≈ 16 GB → `hf_cache/r1llama8b-hf`). GGUF
`unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF` revision `615f8936`, file `DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf` (≈ 6.6 GB →
`hf_cache/r1llama8b-gguf`). Chat template checked on the HF `tokenizer_config.json` (2026-11-17): the generation prompt
ends with `<｜Assistant｜><think>\n`, so **the template writes the think tag** (Qwen's rule, not Phi's): pos / neg rows
come from `cc.py rows` unchanged, calibration rows carry no prefix (`lab/probes/cc_llama/rows.py`, 2 unit tests). Card
defaults: `do_sample`, T 0.6, top-p 0.95, "avoid system prompts". **Protocol stays the one of the other two families**
(greedy, `bench_runner` defaults with its system prompt, `-np 2`, `-c 16384`, llama-server `--jinja`); the deviation
from the card is a declared limit, not a knob. Why this model and not Gemma: Gemma's `post_feedforward_layernorm` sits
after `down_proj` and would attenuate the write (Kocabay's Muse case, roadmap note 2026-11-16); Llama is pre-norm, the
`down_proj` output lands on the residual as in Qwen and Phi. Why not SmolLM3-3B: smaller, lineage less distinct.

## Stage 0: premise probe (≈ 30 min GPU; decides budget and whether the rest runs; reported either way)
P1. **Reasoning is parsed:** 5 GSM8K + 5 MATH dev items through `bench_runner.ask` at 2,048: `reasoning` non-empty and
    `content` without `<think>` tags. llama-server knows the DeepSeek-R1 format; if it does not split for this GGUF, the
    `split_think` fallback of §45.15 covers an open tag (the template pre-fills `<think>\n`, so the output may begin
    without the tag and end with `</think>`: the fallback treats text before `</think>` as reasoning; checked in
    `tests/test_bench_runner.py` already). Report which path was taken.
P2. **The base over-thinks at the budget:** of those 10 items, ≥ 3 cut at 2,048 (`finish == "length"`) → budget 2,048. If
    ≤ 2, repeat at 1,024; if still ≤ 2, **stop** (no over-thinking to fix at 1-2k; reported, no arms).
    `lab/runs/cc_llama/budget.txt` holds the budget of the whole study.
P3. **llama.cpp applies control vectors to `llama`:** a same-norm random vector (d 4,096, layers 1..31, norm 1.0 per
    layer; `probe.py randvec`, 1 unit test) at scale 8 changes the greedy output of the 10 prompts at 512 tokens against
    the base. All identical → stop.
P4. Tokenizer offsets (`return_offsets_mapping`) work (CPU; `probe.py offsets`).
P5. `rows.py template` on the HF tokenizer prints `template_writes_think: true` (CPU). If false (template differs from
    the one read today), the Phi rows rule applies instead and this plan says so as [rev] before any arm.
P6. GGUF ↔ HF: `general.name` / `general.basename` name this model, `llama` arch, 32 `blk`, and **every `blk.N.ffn_down`
    is Q6_K** (merge_q6k refuses anything else; unsloth's Q6_K keeps `ffn_down` at Q6_K in the files we know, the
    embeddings may be Q8_0, which does not matter). If some `ffn_down` is not Q6_K, the compile part needs another quant
    file (noted here as [rev], not improvised).

[rev 2026-11-17 00:45, after stage 0's CPU checks, before any arm.] **P4 failed under the plan's loader and is fixed in
code, not in the plan's criteria:** transformers 5.12's `AutoTokenizer` hands this directory to the slow `LlamaTokenizer`,
which splits "We need" into byte tokens (`W`, `ene`, `ed`) and returns offsets that do not point at the tokens;
`PreTrainedTokenizerFast.from_pretrained` on the same directory agrees with tokenizer.json (the `tokenizers` library,
what llama.cpp's converter read) and keeps the chat template. New `engine/tok.py` (`load_tokenizer`: AutoTokenizer if it
reproduces tokenizer.json on a sample, else the fast loader, else error; 4 unit tests written first) now used by
`contrast_vector`, `compile_control_vector`, `cc.py rows` and this probe. Qwen3.5 and Phi agree under AutoTokenizer
(checked on the three directories) and are returned unchanged, so nothing earlier moves. Without this the HF forwards
of the op would have seen other tokens than the served model. P4 re-run: `TokenizersBackend`, offsets monotone, ok.
P5 true (template ends with `<think>\n`), P6 true (`llama`, 32 blocks, 32/32 `ffn_down` Q6_K, GGUF sha256
`403a545b2dce6dfb`). P2: budget **2,048** by the rule (cut count in `lab/runs/cc_llama/probe.log`).

## Data (no download; nothing from the dev set or the sealed sets)
`cc.py split` with `CC_RUNS=lab/runs/cc_llama`, seed 20261110: the same 64 discovery / 64 screen MATH items and GSM8K
**train** 0-63 as the Qwen and Phi studies (none in dev or sealed sets).

## Part A — the direction as a control vector (prediction row 37; mirror of `lab/probes/cc_phi/PLAN.md` steps 1-6)
1. **Traces** (`cc.py traces`, `CC_HF=hf_cache/r1llama8b-hf`): base Q6_K, greedy, 4,096 cap, 128 discovery problems. Keep
   traces that close `</think>` and finish with `stop`. ≥ 30, else `extra` (GSM8K train 64-191), else stop.
2. **Rows** (`cc.py rows`): pos = last 48 reasoning tokens before `</think>`, neg = the reasoning before them (first 1,536
   at most); `think` flag; the template writes the tag (P5).
3. **Vector:** `contrast_vector` (`programs/cc_llama_vector.wspl`), placebo seed 2026, GPU (8B bf16 streams layer by
   layer). Report per-layer increment norms and the last-row ratio (Qwen ≈ 2×, Phi 0.87×).
4. **Screen** (scale; screen MATH items at the budget): base, vector at 0.25 / 0.5 / 1 / 2; `cc.py choose` (most correct,
   grader 1; ties → smaller; floor rule). No scale beating the base in correct **or** finished → **FAIL**, nothing to dev.
5. **Dev** (`bench_runner` defaults, GSM8K 0-99 + MATH select_indices(2026, 50)): base, vector at the chosen scale,
   placebo at the same scale; one run per arm; **C1-C3 of `lab/probes/cc/PLAN.md`** verbatim (`cc.py verdict`).
6. **Sealed** only if dev passes: `data/sealed_confirm.json` (first use for this model), base / vector / placebo,
   criterion of `lab/probes/sealed/cc_confirm.md` (`cc_verdict.py`, `SEALED_CC_RUNS=lab/runs/sealed_cc_llama`).

## Part B — the direction compiled into `down_proj` (prediction row 38; runs only if Part A's dev passes C1-C3)
Same op and criteria as `lab/probes/cc_phi/PLAN.md` "Compile the Phi direction": vector and placebo scaled to the chosen
scale (`lab/probes/cc_phi/scale_vector.py`, generic; skipped when the scale is 1.0), calib = `rows.py calib` (finished
traces, no prefix), `programs/cc_llama_compile.wspl` (`lora=0`, heldout 28, seed 20261111, max_len 4,096), out
`hf_cache/cc_llama_op/`. **G0** (held-out median of per-layer medians |uᵀx − 1| ≤ 0.5) gates the merge. Dev: merged and
placebo merged served as `-m`, no flags, **C1-C4 of `lab/probes/cc_weights/PLAN.md`** (`verdict.py`, `CCW_CC=lab/runs/cc_llama`,
`CCW_RUNS=lab/runs/cc_llama/weights`); references = Part A's `dev_base` / `dev_vec`. **Sealed set 2** only if PASS:
`data/sealed_confirm2.json` (first use for this model): base, vector at the scale, merged, placebo merged, same C1-C4
(`CCW_SEALED=lab/runs/sealed_ccw_llama`). Reported, not a criterion: HumanEval first 20 (`he_eval.py`, 2,048 cap) on base
and merged. Part B does not run if Part A's sealed does not confirm? **It does run** (dev PASS is the gate, as for Phi);
the sealed outcome of A is reported next to B's.

## Chain (`lab/runs/cc_llama/chain.sh`, one script, both parts, resumable)
`set -eu`, ERR trap writing FAILED, `chain_guard` config (GGUF sha, vector sha, budget, scale, commit) in
`lab/runs/cc_llama/` and `.../weights/`, 2-item smoke per served configuration, "no valid control vector" check on every
vector arm, sha256 of every served GGUF in the log, `served` recorded in every JSON, port 8097, one GPU job at a time
(waits for any llama-server to exit before starting). Not edited while running.

## Prediction (ours, low confidence; arm and control; premises marked)
- **Stage 0:** P1 holds via the server's split (60 %) or the fallback (40 %); **P2 holds at 2,048** (R1 distills think
  long: 3-7 of 10 cut; budget 2,048) — if it needs 1,024, every band below moves down as Phi's did and is read as such;
  P3-P6 hold.
- **Traces:** 60-100 of 128 close within 4,096 (Qwen 83, Phi 107).
- **Vector profile:** last-row increment 0.5-2.5× the one before; median per-layer norm within 0.3-3× Qwen's.
- **Screen (2,048):** base 12-30/64 MATH; best scale **0.5 or 1.0**; vector at the best scale base + 3 to + 12.
- **Dev (2,048):** base GSM8K **55-80** (cuts 10-35), MATH **12-28** (cuts 18-36); vector GSM8K **+4 to +14**, MATH **+3 to +10**,
  tokens −15 to −40 % on both, cuts at least halved on GSM8K; placebo within ± 3 of base on both, tokens within ± 10 %.
  C1-C3 PASS; sealed confirms (≥ one benchmark significant with both graders; placebo null; no damage).
- **Compile:** G0 passes, median 0.02-0.15; cos(written, dW) median 0.6-0.9; ‖dW‖/‖W‖ median 0.005-0.03. Dev merged within
  **5 of the vector on GSM8K and 4 on MATH**, above base by at least half the vector's GSM8K gain with p < 0.05; placebo
  merged within ± 3 of base on both, tokens ± 10 %; C1-C4 PASS; sealed set 2 confirms. HumanEval-20 merged within ± 3 of
  base. No arm with > 2 % of finished answers under 50 characters of thinking.
- **Family reading we expect (not a band):** the effect beyond the budget sits between Qwen's and Phi's; the tally after
  rows 37-38 moves by two scorable rows.
- Premises: (p1) GGUF and HF are the same checkpoint (P6); (p2) llama-server's template for the GGUF matches the HF
  one incl. the `<think>\n` pre-fill (P1 tail inspected); (p3) grader 1 reads `\boxed{}` from `content` (reported if ≥ 5 %
  of finished answers have no boxed content); (p4) greedy on a model whose card says T 0.6: same as both families, a
  limit; (p5) the system prompt of `bench_runner` is rendered by this template (it is: the template has a system branch);
  (p6) `-np 2` nondeterminism small; (p7) the op streams 8B bf16 through 12 GB (Qwen 9B did; commit ≈ 17 GB, disk has
  93 GB free, pagefile can grow).
- **We would be surprised if:** the vector is below base by > 3 on GSM8K at the chosen scale; the placebo reaches
  vector − 3 on either benchmark; the base cuts < 10 % on both benchmarks at 2,048 **and** at 1,024; the chosen scale is
  2.0; G0 fails; cos median < 0.5; merged below base on GSM8K; merged > vector + 5 on either benchmark; > 10 % of a
  served arm's finished answers without thinking (the stop regime of §45.14); llama.cpp rejects or ignores the control
  vector on `llama` (P3 fails).

## Part C — dose (added 2026-11-17 16:20, after Parts A-B were scored, before any run; prediction row 40)
**Question.** §45.19 found the vector at 0.5 loses as many GSM8K items as it recovers, by premature closure on an
intermediate result (14 lost on sealed set 1, 0 of them cut). 0.5 is already half the scale that breaks the model
(1.0: 4/64). Does a smaller dose keep the saving and the recoveries while cutting that cost?
**Arms (dev, same 150 items, reference `dev_base.json` of Part A, premise p3 as in §45.17):** vector at **0.25** and
**0.35** (`dev_vec_s0.25.json`, `dev_vec_s0.35.json`). No placebo on dev (null at 0.5 on dev and both sealed sets; a
weaker placebo is not more informative). Chain `lab/runs/cc_llama/chain_c.sh`, guard in `lab/runs/cc_llama/dose/`.
**Gate (`lab/probes/cc_llama/dose_gate.py`, 2 unit tests written first), fixed now:** chosen = the scale with the larger
GSM8K + MATH correct on dev (tie: the smaller); sealed set 3 runs only if, at that scale, GSM8K ≥ base + 3 (≥ 77),
lost (base correct and finished → arm wrong) ≤ 6, MATH ≥ base + 4 (≥ 26) and GSM8K tokens ≤ −20 %. Otherwise the
result is reported and sealed set 3 is **not** spent. **Sealed set 3** (`data/sealed_confirm3.json`, 200 GSM8K + 100
MATH-500, single use, first use of any kind): base, vector at the chosen scale, same-norm placebo at that scale;
`cc_verdict.py` criterion (C1 significant with both graders on one benchmark, C2 placebo null, C3 GSM8K ≥ base − 2).
**Prediction (low confidence; arm and control):**
- 0.25: GSM8K **77-82** (base 74; dev 0.5 gave 76), lost **3-7** (0.5: 9), cuts 6-14 (base 22), tokens −18 to −32 %;
  MATH **26-31** (base 22; 0.5: 32), cuts 16-22.
- 0.35: GSM8K 76-81, lost 5-9, cuts 3-9, tokens −28 to −40 %; MATH 28-33, cuts 12-18.
- Gate passes at one of the two (55 %). If sealed set 3 runs: base GSM8K 150-170/200 and MATH 44-60/100; vector within
  ± 4 of base on GSM8K and + 0 to + 8 on MATH, tokens −20 to −35 %; placebo within ± 3 of base; **does not confirm C1**
  (p ≥ 0.05 on both benchmarks, 60 %), confirms (25 %). Overall: we expect the lower dose to trade saving for safety
  (fewer losses, fewer recoveries) with the net accuracy still ≈ base; a confirmation would mean the Llama failure was
  dose, not direction.
- Premises: p3 reused `dev_base` (same server flags, `-np 2`); the gate thresholds are the ones above and nothing
  else; sealed set 3's hash check passes.
- **We would be surprised if:** at 0.25 GSM8K < base − 3 or lost ≥ 10 or token saving < 10 %; any arm with > 2 % of
  finished answers without thinking; sealed set 3 vector > base + 10 on GSM8K; placebo significant on either benchmark.
**Cost.** Dev 2 × 150 ≈ 1 GPU h; sealed set 3 if run 3 × 300 ≈ 3 GPU h.

## Limits known now
One budget, greedy (card says sample), one quant (Q6_K), one third model; traces and dev items shared with the other two
studies (fine: no item from dev or sealed sets); system prompt against the card's advice, kept for comparability; the
sealed sets are used for the first time on this model but for the third time overall (their items are public GSM8K /
MATH-500 items; the protection is against **our** decisions, which never saw them). Nothing here says anything about
sampling (roadmap gap 1 stays open).

## Cost
Download ≈ 23 GB (approved 2026-11-16). 8B Q6_K on the 3060 ≈ half Phi's speed. Stage 0 ≈ 30 min; traces ≈ 1.5-2.5 h;
op vector ≈ 20 min; screen 5 × 64 at 2,048 ≈ 1.5 h; dev 3 × 150 ≈ 2 h; sealed 3 × 150 ≈ 2 h; compile op ≈ 1 h; dev 2 × 150
≈ 1.3 h; sealed 4 × 150 ≈ 2.7 h; HE 2 × 20 ≈ 10 min. **≈ 12-14 GPU h**, two nights, one chain.

[post-data 2026-11-17 15:40; results §45.19-45.20, rows 37 and 38 both ❌.] Chain 00:36-12:19 (11.7 GPU h, no failure).
Part A: screen 25 / 32 / 35 / 4 / 0 of 64, scale 0.5; dev base 74 / 22, vector 76 / 32 (MATH 13/3, p 0.021 with both
graders), placebo 76 / 23, tokens −46 % / −26 %: C1-C3 PASS. **Sealed set 1 does not confirm**: base 83 / 28, vector 76 /
26 (7/14, 7/9), placebo 82 / 23; pooled over both sealed sets the vector is 24/29 on GSM8K and 17/14 on MATH. The losses
are items the base finished right and the vector finishes wrong without cutting (14 of 14 on sealed 1 GSM8K). Two
declared surprises fired (vector < base − 3 on GSM8K; placebo ≥ vector − 3). Part B: G0 0.030, cos 0.74, ‖dW‖/‖W‖ 0.012,
dev merged 74 / 35 (C1-C4 PASS), sealed set 2 base 75 / 23 → merged 77 / 27 (FAIL on C1), vector 77 / 28 on the same
items (7/7, 4/5: equal), compiled placebo 79 / 20, HumanEval-20 14 → 13. What the plan got right: stage 0, screen, dev
base, dose window (0.5, 1.0 breaks), the op's numbers, merged = vector. What it got wrong: the accuracy gain (none
beyond dev), the vector profile (last row 3.2× the one before), traces closed (103 > 100), the token saving (larger).
Reading: in this family the primitive is a token saving at equal accuracy. Untested here: other budgets, sampling
(the card asks for T 0.6), scales between 0.25 and 0.5. First dev PASS of the series that the sealed set rejected.
