# Loop fix by weight editing: editor report

## Mechanism
- Copying over long range is weak in this model. Repeating a 60-character train segment 126 characters later gains only 0.055 nats per character (probe 3, matched pairs). There is no strong classic induction head: attention at the exact offsets t-P / t-P+1 is under 2% for every head.
- The greedy loops are attractor cycles of the model's own output. The main one is "That the strength of the state of the state,\n" (period 45), which shows up in all 3 baseline examples. Inside the cycle many greedy decisions win by tiny margins (0.03 to 0.3 logits). What holds the loop in place is that it repeats: on the same phrase, per-character agreement with greedy goes from 0.83 with 1 cycle in context to 0.96 with 5 cycles (probe 4).
- I attributed the repetition-driven gain by head, using the direct logit of the correct character with 1 cycle vs 5 cycles in context. Layer 5 head 3 (L5H3) dominates at +0.52; the next largest are L4H0 +0.17 and L5H0 +0.12. L5H3 attends to the matching place in earlier cycles (for example, from "state," to the "T" that followed ",\n" at distances 43, 88, 133...) and pushes the same continuation, so it works as a fuzzy induction head. On natural text its effect barely depends on repetition (probe 3: 0.186 vs 0.181).
- In L5H3's 64-dimensional value space, I took the eigenvectors of the first-order "help the loop character" matrix d logp/d y · y, computed over 4 attractor phrases (probe 5) plus probe 4. I ranked them against a Fisher (second-order) NLL cost on natural train text. The top 2 directions are stable across attractor phrases (|cos| 0.985 to 0.986 against directions from a single phrase) and have a high benefit/cost ratio.

## Causal evidence (dev, 64 prompts)
| edit (L5H3 value) | loop_rate | distinct4 | val_nll |
|---|---|---|---|
| none (baseline) | 0.484 | 0.0433 | 1.44229 |
| whole head x0.5 | 0.375 | 0.0446 | 1.44292 |
| whole head x0 | 0.438 | 0.0437 | 1.44504 |
| whole head x-0.25 | 0.156 | 0.0464 | 1.44672 |
| whole head x-0.5 | 0.141 | 0.0449 | 1.44883 |
| top-2 loop dirs x0 (ablate) | 0.141 | 0.0448 | 1.44305 |
| top-2 loop dirs x-1 | 0.109 | 0.0493 | 1.44438 |
| top-2 loop dirs x-1.5 | 0.0625 | 0.0514 | 1.44524 |
| **top-2 loop dirs x-2 (final)** | **0.078** | **0.0545** | **1.44629** |
| top-2 loop dirs x-3 | 0.078 | 0.055 | 1.44889 |
| control: 2 random dirs x-2 | 0.531 | 0.0453 | 1.44234 |
| top-1 dir of L5H3, L5H1, L4H1, L4H4 (x-1 each) | 0.25 | 0.0504 | 1.44464 |

Random directions do nothing, and the chosen directions do. Flipping their sign, which turns copying into anti-copying, works better than ablating them. Results are flat from x-1.5 to x-3, and I picked x-2 from the middle of that range.

## The edit (`fix.wspl`)
- 2 steps of `project_out` on `transformer.h.5.attn.c_attn.weight`. Each u is a unit vector supported on the V rows of head 3 (rows 960 to 1023), and each step uses `alpha=3 top_rows=64 top_cols=1.0`. The update is W <- W - 3 u u^T W, so the head's value output along each of the 2 orthonormal loop directions is scaled by -2. The vectors are in `vecs/fix_L5H3_V_loopdir{0,1}.npz`.
- Weights changed: **24,576** (64 rows x 384, all in one tensor), against a limit of 107,000. No protected tensor was touched. I checked that the output is bit-identical to the candidate I evaluated.

## Final dev numbers
- Unedited: loop_rate 0.484, distinct4 0.0433, val_nll 1.44229
- Final `edited/model.safetensors`: loop_rate **0.078**, distinct4 0.0545, val_nll **1.44629** (+0.0040)

## Budget
I used 18 of 50 calls: 13 dev_eval (including the baseline and the final model) and 5 forward_probe.

## Caveats and limits
- The analysis of head 3's output (and of the L3/L4 heads) is direct-path only and approximates the ln_f scale. It ignores indirect effects, which may be why the multi-head mix did worse.
- Greedy dynamics are chaotic, so loop rate on 64 prompts carries roughly ±0.06 noise per edit. The size of the drop, the plateau across strengths and the null random control make me fairly confident, but I have not seen the test prompts.
- Many texts that no longer loop are still repetitive in vocabulary ("the seas of the strength of the world"); they just aren't exactly periodic. distinct4 rose only from 0.043 to 0.055.
- Housekeeping: probe 1 first wrote its text list to the repo root; I moved it into this directory. The WSPLang CLI writes `challenges/wsplang/<experiment>.manifest.json` for every run, so I moved those into `manifests/` here and the repo is unchanged apart from this directory.
