# Loop fix by weight editing: criterion fixed before any edit (2026-11-05)

A small replica of exp. 8 of "Sparks of Weight-Space Programming" (Takehi et al., 2026), with controls they did
not run.

## Defect
The trained nanoGPT (`models/nanogpt_shakespeare_char`, val 1.4423) loops under greedy decoding.
- On 64 dev prompts (seed 99, diagnosis only), 47% of 512-character continuations end in an exact loop, for
  example "That the strength of the state of the state,".
- With temperature 0.5 the loop rate is 0%.
- "Loop" = the last 200 characters are exactly periodic with a period of 1-100 characters
  (`harness.loops`).

## Editor
A Claude subagent with Python, the weights and the train split.
- **Behaviour budget:** 50 calls to `harness.dev_eval` (64 dev prompts, seed 1) or `harness.forward_probe`,
  logged in `lab/runs/loopfix/dev_calls.jsonl`. It sees no test prompts.
- **No gradients:** no backward, optimizers or autograd. Its scripts stay in `lab/runs/loopfix/editor/` and are
  checked afterwards.
- **The final edit is a `.wspl` program** run through the WSPLang CLI with
  `max_modified_weights = 107000` (1% of the parameters), so the write guard checks it.

## Test
256 prompts from the second half of val (seed 2), greedy, 512 new characters. Run once per model by
`final_test.py` after the editor has finished.

## The edit passes if all hold
1. **Loop rate:** test loop rate is <= 50% of the unedited model's.
2. **Quality:** val NLL rises by <= 0.01 (every position scored with >= 128 characters of context).
3. **Applied by WSPLang:** the guard accepted it within budget.
4. **Beats a random edit:** the same tensors, the same number of changed weights and the same Frobenius norm
   per tensor, but random direction (3 seeds), must reduce the loop rate by less than half of the editor's
   reduction.

## References (reported, no pass/fail)
- **No-repeat-ngram decoding,** n = 20 characters: a decoding fix that touches no weights.
- **Temperature 0.5 sampling.**
- **Gradient control:** sequence-level unlikelihood (Welleck et al. 2019).
  - 300 AdamW steps, lr 1e-4.
  - Each step, 32 greedy 256-character continuations from train prompts; unlikelihood on every character that
    completes a 16-character n-gram already in its continuation, plus cross-entropy on 32 train windows.
  - Settings fixed now.

Thresholds are not changed after seeing numbers. If the edit fails, that is the result.

## Outcome (2026-11-06): FAILS criterion 4
Editor: two `project_out` steps on rows 960-1023 of `h.5.attn.c_attn` (24,576 weights, Frobenius 3.54), found with
18 of the 50 budgeted calls. Data: `lab/runs/loopfix/final_test.json`.

| Model | Test loop rate | Loops /256 | Val NLL |
|---|---|---|---|
| Base | 0.434 | 111 | 1.44229 |
| **Editor** | **0.102** | **26** | **1.44629 (+0.004)** |
| Random edit, seed 0 | 0.410 | 105 | 1.44272 |
| Random edit, seed 1 | 0.422 | 108 | 1.44293 |
| Random edit, seed 2 | 0.102 | 26 | 1.44290 |
| No-repeat-ngram 20 | 0 | 0 | 1.44229 (unchanged) |
| Temperature 0.5 | 0 | 0 | - |
| Unlikelihood fine-tune | 0 | 0 | 1.48541 (+0.043) |

- Criteria 1-3 hold. Criterion 4 fails: random seed 2 matches the editor's reduction.
- **The 26 = 26 tie is chance.** Only 3 of the 26 looping prompts are shared; no continuation is identical.
- **Post-hoc, outside the criterion:** 7 more random seeds (3-9) gave 115, 115, 103, 108, 64, 102 and 90 loops. So
  over 10 random seeds: 1 equals the editor, 1 halves the loops (seed 7, 64), 8 do little. The editor is the best or
  tied best of 11, which is weak evidence (about 2 in 11 by chance) that its direction matters beyond "perturb
  this head".
- The edit moves loops more than it removes them: 11 of the base's 111 looping prompts still loop, and 15 new ones
  appear.
- Check: loop counts recomputed with an independent periodicity check on regenerated text (base 111, editor 26,
  seed 2 26, seed 7 64); both agree.
- Reading: a decoding rule (no-repeat-ngram 20) removes every loop at zero NLL cost and touches no weights. This
  replica does not support weight editing as the better fix for loops in this model.
