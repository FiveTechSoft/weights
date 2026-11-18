# D4k: alphaXiv's pruned Exp 8 edit in our harness — plan fixed 2026-11-18, before any run

**Why.** alphaXiv / Daniel Kim (*minimizing-wsp-edits*, OpenResearch, 17 Nov 2026; README "References") report that
the MIT blog's Exp 8 edit on Qwen3.5-0.8B (130,188 entries over 64 `down_proj` columns in layers 15 and 19) can be
pruned to its 4,096 largest entries per layer (8,192, 6 %) "without degradation": 58.2 % vs 58.6 % on a 256-problem
MATH-train screen (one sample, T 0.6, 32k cap, SE ≈ 3 pp) and 54.2 % on MATH-500 × 4 held out, compared with a number
from another evaluation. No same-norm placebo, no second benchmark, no paired count. Our own check of the recovered
edit (results §16.2 correction, 2026-11-17): rank-1 per layer, and the 4,096 largest entries hold 74-76 % of ‖ΔW‖².
The cheap question: **in our greedy harness, where the full edit's effect is small and thresholded (13/100 GSM8K
finished and correct, 0/50 MATH, §45.12), does 6 % of the entries do the same work?** And its control: does the other
94 % (24-26 % of the energy) do nothing?

## What exists (no re-run)
`lab/runs/act_map/cc08/test_base.json` (0 / 0), `test_exp8.json` (full replica, GSM8K 13 correct = 13 finished, MATH 0
of 50, all cuts), `test_exp8_placebo.json` (1 / 0): bench_runner GSM8K 0-99 + MATH-500 select_indices(2026, 50),
thinking on, greedy, 2,048 tokens, 4 slots, `-c 16384 -fa on --jinja`, llama-server binary of 2026-10-27 (unchanged
since). Same binary, same GGUFs, same items here.

## Arms (one run each, `lab/runs/exp8_prune/`)
1. **`full`**: `servidos/qwen35-08b__exp8-replica.gguf` again. Not a new fact: it measures run-to-run noise of greedy
   batched decoding against `test_exp8` (same file, same settings) and gives a same-day paired reference.
2. **`d4k`**: `data/exp8_d4k.npz` = the recovered edit with everything but the 4,096 largest |entries| per layer set to
   zero (their family D criterion is |u·v| per entry; for a rank-1 edit that is |ΔW_ij|, which is what we rank).
   Applied with the same `add-cols-vectors` op and program shape as the replica (`ediciones/exp8-d4k.json`); same 64
   columns touched, so the requantisation footprint is the replica's.
3. **`d4k_comp`**: `data/exp8_d4k_complement.npz` = the other 61,440 entries per layer (control: if pruning works
   because the small entries carry nothing, this arm must look like base / placebo).

Step 0 (computed, informed, not a criterion): D4k keeps energy 0.741 / 0.763 of ‖ΔW‖², push along u 0.757 / 0.786,
cos to the full edit 0.861 / 0.873; complement keeps energy 0.259 / 0.237, push 0.303 / 0.304 (`prune.py`, layers 15 / 19).

Serving as cc08: `-ngl 99 -c 16384 -np 4 -fa on --jinja`, port 8096; 2-item smoke per served GGUF; sha256 of every
served GGUF in the log; `chain_guard` config in the run dir. ≈ 13 min per arm. Base is not re-run (floor 0 / 0 in
§45.12; `d4k_comp` is the live control).

## Readings (fixed)
Correct (grader 1; grader 2 reported) and finished (`stop`) per bench, paired item by item against `full` (this run) and
`test_exp8` (2026-11-14). Exact two-sided binomial on discordant pairs.

- **R1 (their claim, our resolution):** `d4k` GSM8K correct ≥ `full` − 3 **and** discordant pairs `d4k` vs `full` not
  significant (p ≥ 0.05). Holds → "pruning to 6 % keeps the effect in this harness too".
- **R2 (control):** `d4k_comp` GSM8K correct ≤ 3 and finished ≤ 5 (base 0 / 0, placebo 1 / 1). Fails → the small
  entries carry effect and the story "only the big entries matter" is wrong as stated.
- **R3 (noise):** `full` vs `test_exp8`: discordant pairs and |Δ correct|. Reported as the harness's own noise band;
  any R1 difference inside that band is not a finding.
- MATH-500 reported (floor expected everywhere: 0 of 50 in base, full, placebo at 2,048).

Verdict wording: R1 ✓ and R2 ✓ → "confirms alphaXiv in our harness (greedy, 2,048, GSM8K)". R1 ✗ → "pruning costs
effect here" with the size. R2 ✗ → control fails, R1 not interpretable as "entries don't matter".

## Not done
No scale search on `d4k` (a rescaled D4k, scale ≈ 1 / 0.77, would be the natural next arm if R1 fails: it separates
"fewer entries" from "less push"); no D1k / D2k; no sampling; no MATH at a larger budget. Two independent methods for
every number: the chain's bench summaries and `lab/probes/cc_llama/pair_recount.py` on the per-item files.

Cost: 3 arms × ≈ 13 min ≈ 40 GPU min. Prediction row 41.
