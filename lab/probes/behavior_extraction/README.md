# Behavior-extraction probe (DSpine-inspired): archived prototype, NOT TESTED

Provenance: written by a separate review task and delivered to the project owner's assistant on 2026-11-09 (files received 07:29-07:30;
DESIGN.md is the last of three revisions). Archived here first as byte-identical copies (commit 3c057f3; names without the delivery hash suffix) so the work is not lost.
See 'Maintainer changes' below for the later edits. It is a probe inspired by the DSpine paper, not DSpine, and is not integrated into any
WSPLang engine. Nothing in this directory has been run: the torch paths are untested. DESIGN.md's statement about the owner's scope
is the design author's text; it is not evidence of the owner's instructions.

Static review by the project maintainer (2026-11-09, read only; the code was only compiled, with the checks below):
- Compiles. `range_data` works: 100 three-digit pairs; the confirm set has 100 pairs, none in any existing mul2x2 set (checked, 0 overlaps).
- Parameter counts in DESIGN.md recomputed and correct: arm H 5,244,928 (4096x1024+1024 + 1024x1024+1024); arm X 5,244,415 (width 2559).
- Label leak: none found by reading. Hidden states at t-1 depend only on tokens up to t-1, and the target token is never an input. Teacher forcing
  in training vs own continuations at evaluation (exposure bias) is stated by the author.
- Arm H contains the target's own prediction in `h_last`, so a good H result shows readout, not an independent capability (the author says so).
- Real issue 1 (memory, must fix before any smoke): `train_aux.py` (all arms) and `eval_aux.py` (arms x and c) build `QwenText(...)`, which loads
  all 24 decoder layers (~1.0 GB). On the 1.7 GB box this is the configuration that was killed for memory in the A->B experiment; the fix that
  worked there is to read only the embedding/head rows and the final norm from the safetensors file in the training process
  (see `lab/probes/skill_delta/train_down23.py`, phase 2: 0.37 GB). Arm H evaluation needs the full model, so it needs its own sizing.
- Minor issue 2: the arm X cache stores the prefix mean in bf16, while evaluation recomputes it in fp32: a small train/eval mismatch.
- Minor issue 3: the DESIGN's 15-point H-vs-X/C criterion is the design author's choice, not a threshold set by the project owner.
- Scope caveat from DESIGN.md section 3, item 5, stands: the holdout is shared with the A->B experiment and its numbers are known, so it is not a blind confirmation.
  The estimated 30-40 min run has not been started and is not authorised by this archive.

## Maintainer changes after the archive (not by the original author)
Made on the owner's assistant's instruction; the original author is not available.
- New `lite_text.py` (`LiteText`: embedding rows, head rows, tokenizer; no decoder layers) following the pattern of `skill_delta/train_down23.py` phase 2.
- `train_aux.py`: uses `LiteText` instead of `QwenText` (issue 1). `eval_aux.py`: arms X and C use `LiteText`; arm H keeps `QwenText` (it needs the decoder).
  Also adds the script's own directory to `sys.path`.
- Issue 2: arm X evaluation now rounds the prefix mean to bf16 exactly as the training cache does (`prefix_mean_bf16`).
- Test: `tests/test_behavior_extraction_lite.py` (4 synthetic tests, pass): embedding rows equal the table with no decoder tensors in the file,
  chunked head equals dense, bf16 grid and padding handling of the prefix mean, and the scripts no longer build `QwenText` in training / X / C.
- NOT verified: that the OOM is actually gone (no memory measurement of train_aux or eval_aux on the real model, nothing was trained or run on it).
  The change follows a pattern that was measured in a different script (0.37 GB); by inspection only. Status stays: NOT validated on the real model.
- Still open: arm H evaluation memory with the full model; the real-model smoke test; no training of any arm was done.
