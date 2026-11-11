# H6: in a training we control, how small a slice of ΔW carries the learned skill, and do the directions recur across seeds? (pre-registration, approved 2026-11-07)

Status: **APPROVED by the user 2026-11-07 ("exacto, así cada fotograma del proceso nos irá contando cómo el modelo se va transformando"); nothing run.** Runs **before H5** (user decision 2026-11-07). Order: stage 0
(trajectory of small trainings), then the rank/transfer arms, then stage 2 (synthetic). Thresholds below are fixed
before any transfer arm is evaluated. Once approved, this file is committed and is not edited after the first
transfer result. Changes go in a dated addendum. Gates G1/G2 may change the task or training settings in an addendum,
because they run before any transfer result exists.

## Why
- H4 used MiMo, which has three problems as a test bed:
  - its training data is unknown;
  - the metric is noisy (24 vs 36 base cuts depending on quantization, only 16 shared);
  - each arm costs 80 min.
- **A pair we train ourselves fixes all three.** We know the skill. A short-answer test takes minutes, so whole rank
  sweeps are cheap. Several seeds separate what the data teaches from optimizer and data-order noise.
- **The question that matters for WSPLang** is not only "can ΔW be compressed". That is already known for fine-tune
  deltas (LoRA extraction, DARE, TIES, BitDelta). It is:
  1. how small a slice of ΔW keeps the skill, measured against a random slice of the same size;
  2. whether energy (singular values) and function line up;
  3. whether the directions that carry the skill are found again by independent trainings. If they are, they can be
     looked up once and reused, which is the first step toward building such updates without training.

## Models and training
- **M0:** `hf_cache/qwen35-08b-hf` (Qwen3.5-0.8B, open, local).
  - 24 layers (18 Gated DeltaNet + 6 full attention), hidden 1024, tied embeddings.
  - Largest matrix rank: 1024.
- **What trains:** only the 2-D matrices in `model.layers.*`. Frozen: embeddings/lm_head (tied), all 1-D tensors
  (norms, A_log, dt_bias), conv1d. So ΔW lives exactly where `subspace_transfer` acts, and "transfer everything"
  equals M1 by construction (a check, not a result).
- **Full fine-tune, no LoRA.** A LoRA ΔW is low-rank by construction and would make the rank result trivial.
- **fp32 master weights** for the trained matrices. bf16 weights would round away updates smaller than one bf16 step,
  and the rounding pattern would show up as fake structure in ΔW.
- **Optimizer:** AdamW, lr 1e-5, cosine decay, 1 epoch, batch 32.
  - Memory estimate: ~0.55 B trainable × 16 bytes ≈ 9 GB plus the frozen embedding; it should fit 12 GB with short
    sequences.
  - A smoke run checks this. Fallback (addendum): Adafactor.
- **Tokenizer:** `tokenizers.Tokenizer.from_file(tokenizer.json)`, not AutoTokenizer. transformers 5.12.1 in the
  venv has the wrong Qwen pre-tokenizer (findings F37).
- **Seeds:** 3 trainings (s = 1, 2, 3), same data, different data order (and dropout if any), giving ΔW_1, ΔW_2,
  ΔW_3.

## Stage 0 (first, the core of H6): the effect of each small training, one after another
User direction 2026-11-07: understand the change made by one small training, then apply a second small training and
analyse again, and so on. The run is therefore cut into **10 chunks of 2 000 examples**; chunk t continues from
checkpoint W_{t-1}. Everything below is measured at each boundary, for each seed. Increment ΔW_t = W_t − W_{t-1};
cumulative C_t = W_t − W_0.

**Fingerprint of each small training** (computed during training, online; snapshots are kept on the CPU; analysis
code `lab/probes/h6/fingerprint.py`, later a WSPLang op):
1. **Effect:** test_id accuracy, both OOD sets and retention NLL after each chunk (about 3 min).
2. **Where:** ‖ΔW_t‖/‖W‖ per layer and matrix kind, and each one's share of the chunk's total change.
3. **How many dimensions:** effective rank of ΔW_t (entropy of the singular values), and the energy in the top 1/5/10 %.
4. **Which directions:**
   - principal-angle overlap of the top-r subspace of ΔW_t with that of ΔW_{t-1} (does each small training push
     the same way?);
   - overlap with C_{t-1} (is it still in the subspace that earlier chunks opened, or new?);
   - overlap with the same chunk of the other seeds.
5. **Readable meaning:** for down_proj / out_proj, the logit lens of the top directions of ΔW_t (exploratory, as in
   `semantic_change.py`).

**Questions stage 0 answers (descriptive, reported as curves, no threshold):**
- Does the update settle into a fixed subspace after a few chunks (overlap of ΔW_t with C_{t-1} → 1), or does it
  keep opening new directions?
- Does the accuracy gain per chunk follow the size of the change, or do some chunks move a lot and teach little?
- Is the first chunk's change the same across seeds (a property of the data) or different (noise)?

**Synthetic continuation (pass/fail, fixed now):** if the subspace settles, the rest of the training is moving inside a
known subspace, and can be imitated without training it.
- Arm E_k: W_0 + α · C_k, the first k chunks extrapolated, k ∈ {1, 2}. α is chosen on the dev split (200 items),
  α ∈ {1, 2, 3, 5, 8, 10}. Zero further backward passes.
- **Passes** if E_k reaches recovery ≥ 0.7 against the full 10-chunk M1, averaged over seeds, with the cost of k/10
  of the training. It must also beat M3 at the same norm (p < 0.01 in each seed).
- **Prediction:** E_2 recovers about 0.4-0.7, so it fails at 0.7. Overlap of ΔW_t with C_{t-1} rises to about 0.6-0.8
  by chunk 3. Confidence: low.

**After stage 0, a second skill (stage 0b, written as an addendum before it runs):** continue M1 with 10 chunks of
a different small task, for example 4-digit addition, and fingerprint again.
- Does it use new directions, or reuse the multiplication subspace?
- What does it do to multiplication accuracy (interference)?

**Storage:** fp32 snapshots of the trained matrices (~2.2 GB each). Kept: chunks 1, 2 and 10 of each seed
(~20 GB in total). The other boundaries are analysed online and dropped.

## Task and data (`lab/probes/h6/make_data.py`, deterministic)
- **Skill:** multiplication of a 3-digit by a 2-digit number, answer only. Prompt `"{a}*{b}="`, target the product
  digits, then a newline.
- **Prompt for evaluating every model:** the same 3 fixed solved examples before the item (3-shot). Then M0 knows the
  format, and the measured gain is arithmetic, not format learning.
- **Splits:** train 20 000 pairs; test_id 1 000 unseen 3×2 pairs; test_ood_easy 500 pairs 2×2; test_ood_hard 500
  pairs 3×3. No (a, b) pair is shared across splits.
- **Retention:** mean token NLL on 300 fixed GSM8K-train solutions (`data/gsm8k_train.jsonl`, first 300). Lower is
  better; reported for every arm.
- **Graders:** exact string match of the generated digits, and an independent one (first integer parsed from the
  output). Disagreements are listed.
- **Decoding:** greedy, max 8 new tokens, in-process with HF transformers on the GPU (no GGUF; the evaluation
  harness differs from H4 on purpose).

## Arms (per seed s; ΔW = ΔW_s, all 2-D layer matrices, α = 1 unless said)
- Grid of ranks, as % of the largest rank 1024: **r ∈ {10, 20, 51, 102, 256}** (1, 2, 5, 10, 25 %).
- `basis=svd` with rank r gives exactly the rank-r truncation of ΔW (Eckart-Young). Projecting on either side gives the
  same result.

| Arm | Weights | Purpose |
|---|---|---|
| M0 | base | reference |
| M1 | base + ΔW_s (`basis=full`) | full skill; must equal the trained model (check) |
| M2_r | base + rank-r SVD of ΔW_s (`basis=svd`, k=r) | how small a slice keeps the skill |
| M3_r | base + ΔW_s on r random directions, rescaled per tensor to M2_r's norm (`basis=random norm_like=svd`) | control: same source, same size, random directions |
| X_r | base + ΔW_s projected on the top-r directions of the **other two seeds'** deltas (`basis=deltas:<M1_t>,<M1_u>`) | do independent trainings find the same directions? |
| α sweep | M2_r at r = r90, α ∈ {0.5, 1.5, 2} | stability |

- **Recovery:** R(X) = (acc(X) − acc(M0)) / (acc(M1) − acc(M0)) on test_id.
- **Energy share:** E_r = ‖P(ΔW)‖² / ‖ΔW‖², reported next to R so energy and function can be compared directly.

## Code needed before running (tests first)
1. `subspace_transfer basis=svd` (top-k singular directions of D itself) and `basis=full`. These are shared with H5.
   `norm_like=svd` must work, so `basis_for` receives D.
2. `subspace_transfer basis=deltas:<dirA>,<dirB>`: the top-k directions, on D's smaller side, of the concatenated
   deltas of models A and B against the same base.
3. `subspace_transfer basis=acts:<file>` (stage 2): stored activation principal directions for each matrix.
4. `lab/probes/h6/`:
   - `make_data.py`;
   - `train.py` (trains only the 2-D layer matrices, fp32 master, saves an HF folder with provenance);
   - `eval_arith.py` (3-shot, both graders, retention NLL);
   - `sweep.py`: builds the arms in memory with `engine.subspace_ops` (the same functions as the op), writes
     `lab/runs/h6/seed{s}.json`.
   Stage 2 adds `acts.py` (activation principal directions on M0, forward only) and `grad_once.py` (the summed
   gradient at M0, no optimizer step).
5. Headline arms are also built once through the WSPLang op and checked equal to the in-memory build. Two methods.

## Gates (before any transfer arm; may trigger an addendum)
- **G1, headroom:** M0 3-shot accuracy on test_id must be between 5 % and 50 %. If not, change the digit counts
  first (more digits if too high), before training.
- **G2, the skill is learned:** acc(M1) − acc(M0) ≥ 30 points on test_id for each seed, and M1 equals the trained
  model on 50 items. If not, change lr/epochs (addendum) before any transfer arm.
- **G3, the evaluation is deterministic:** running M0 twice gives identical outputs on test_id.

## Decision rules (fixed now)
- **H6a, size (a measurement, not pass/fail):** r90 = the smallest r in the grid whose recovery, averaged over the 3
  seeds, is ≥ 0.9. Reported with the per-seed values and E_r. If no r reaches 0.9, r90 = "> 256".
- **H6b, specificity:** at r* = r90 (or 102 if r90 > 256), M2_r* beats M3_r* by ≥ 20 points on test_id in **each**
  seed, with exact McNemar p < 0.01 in each. Pass/fail.
- **H6c, recurrence across seeds (the WSPLang question):** at r*, both must hold:
  1. the recovery of X_r* is ≥ 0.8 × that of M2_r*, averaged over seeds;
  2. X_r* beats M3_r* with McNemar p < 0.01 in each seed.

  Pass/fail.
- **Also reported, no threshold:**
  - recovery on test_ood_easy and test_ood_hard;
  - retention NLL for every arm;
  - the α sweep;
  - energy against recovery, as a curve.
- If M3 does about as well as M2 at some r, that rank is "any perturbation of this size", the failure mode of
  results §36.
- Whatever the result, it is written up in results.md and findings.md.

## Cost and order
- GPU, one job at a time: H6 starts after the H4 A2 run ends.
- Code and tests: a few hours.
- Each training: estimated 30-60 min (the DeltaNet layers use the slow torch path, because fla and causal_conv1d are
  not installed; sequences are short).
- Each evaluated arm: about 2-3 min.
- Per seed: about 18 arms, about 1 h.
- Total: about 5-6 h of GPU.
- Disk: about 3.2 GB per fp32 model folder (3 seeds); arms are built in memory.

## Prediction (written before running)
- **H6a:** r90 between 51 and 102 (5-10 %). Confidence: medium; a narrow skill should be fairly low-rank.
- **H6b:** passes. A norm-matched random slice should do little or harm. Confidence: medium.
- **H6c:** fails on criterion 1. Seeds share the leading directions but not enough: recovery of X at about 0.5-0.8 of
  M2. Confidence: low. This is the most informative outcome either way.
- **Energy and recovery:** recovery rises faster than energy, so the first directions carry more function than their
  energy share. Confidence: low.

## Stage 2: from "what changed" to a synthetic update (the user's original goal)
The aim: analyse what a small training changed, then build that change ourselves without running the training.
Stage 1 above says *how much* of ΔW is needed. Stage 2 asks whether ΔW can be *predicted* from things that are cheap
to compute on M0. It runs on the same seeds and test sets.

**What changed (analysis, reported, no threshold):**
- per layer and matrix kind: ‖ΔW‖/‖W‖, effective rank, and the share of the total change;
- per seed, whether the same layers and matrix kinds carry the change.

**H6d, predictability without training (forward passes only).** For each matrix whose smaller side is its input
(q/k/v, gate/up, DeltaNet in_proj), let K be the top-r principal directions of that matrix's input activations on
512 training prompts, run through M0 (no backward pass).
- Overlap: share of ΔW_s's energy whose input side lies in span(K), against a random r-dim subspace.
- Arm P_r: ΔW_s projected on span(K) (`basis=acts:<file>`). This still uses ΔW. It tests whether the directions
  can be predicted, not a synthesis.
- **Passes** if, at r*, the recovery of P_r ≥ 0.8 × that of M2_r, averaged over seeds, and P_r beats M3_r with
  McNemar p < 0.01 in each seed.

**H6e, synthetic update (no optimizer run).**
- G_r: the gradient of the training loss at M0, summed over the same 512 prompts (one backward pass each, no
  weight step), rank-r per matrix. A single scale α is chosen by accuracy on a 200-item dev split (separate from
  test_id): α ∈ {0.5, 1, 2, 5, 10, 20, 50} × lr·steps.
- Provenance is written honestly: `backward_calls = 512`, no optimizer steps. It is cheaper than training (about
  1 pass over 2.5 % of the data), not gradient-free.
- **Passes** if, at r*, the recovery of G_r ≥ 0.5 averaged over seeds and G_r beats M3_r with McNemar p < 0.01 in
  each seed.
- If H6d passes, G_r is also tried restricted to span(K): synthesis guided by the analysis.

**Prediction:**
- H6d passes partly. The input side of ΔW is pulled toward the task's activation directions, so the overlap is well
  above chance; recovery is about 0.5-0.8 × M2. Confidence: low.
- H6e fails the 0.5 threshold, landing around 0.2-0.4. One gradient step does not reach where 600 steps went.
  Confidence: low.

## Later (not part of H6)
- Composition and interference, S(ΔW_A + ΔW_B), with a second skill.
- Choosing directions by activations or gradients instead of energy.
- Applying what works back to MiMo (H5).

## Addendum 2026-11-07 (before any code ran on the GPU; nothing trained)
**Inside one frame (user, 2026-11-07: "es posible que incluso tengamos que revisar paso a paso lo que llega a
formar un único fotograma").** Each chunk is made of optimizer steps, so every step is logged too.
- **Every step, every chunk** (scalars, small): loss; per matrix |g| (gradient), |u| (the applied update),
  cos(g, u) (how much AdamW bends the gradient) and cos(u_t, u_{t-1}) (does the step keep its heading).
- **Chunks 1 and 10:** a fixed random sketch of each step's update, u · R with R a Gaussian matrix of shape
  (larger side × 8), kept per step in fp16. This allows, afterwards, the overlap of each step with the chunk's final
  increment and with its top-r subspace (inner products preserved approximately, Johnson-Lindenstrauss). The sketch is
  checked against exact values in a test.
- To measure u exactly without keeping a 2 GB copy, AdamW is implemented in the script, one parameter at a time. A
  test checks it against `torch.optim.AdamW`.

**Batch:** 40 instead of 32, so a chunk of 2 000 examples is exactly 50 steps (500 in total). One cosine schedule
runs over the whole 500 steps: chunks are observation points, not restarts.

**Model as loaded:** transformers builds `Qwen3_5ForCausalLM` (text only): 320 tensors, 497.6 M parameters in the
2-D layer matrices (the trained set), 752.4 M in total.

## Addendum 2026-11-07 b (gates G1/G2, before any trained result; H4 A2 stopped at 36/100 by the user)
**G1, task changed.** M0 (3-shot, greedy) on 100 random items per size:

| Size | M0 accuracy |
|---|---|
| 3×2 | 80 % |
| 3×3 | 28 % |
| 4×2 | 66 % |
| 4×3 | 9 % |
| 5×2 | 54 % |

3×2 is above the 50 % ceiling, so the task is now **3×3**: shots, train, dev and test_id are 3×3, ood_easy is 3×2
and ood_hard is 4×3. With 3×3 shots, M0 gets 28 % in fp32 and 26 % with the bf16 embedding used for training.

**Training settings changed:**
- One AdamW step at lr 1e-5 took a fixed batch's loss from 0.34 to 1.20, and 1.14 on another batch. The first
  Adam step is close to sign descent over 498 M coordinates.
- The gradient is correct: a small SGD step along it lowers the loss, 0.338 → 0.319.
- With lr 3e-6 the loss goes 0.34 → 0.28, and 0.42 → 0.45 on another batch.
- So: **peak lr 3e-6, 25-step linear warmup, then cosine** over the 500 steps.

**Memory:** the frozen tied embedding is kept in bf16, gradient checkpointing is on, and each step accumulates
4 micro-batches of 10 (same update as a batch of 40). Peak 8.9 GB.

**Fingerprint SVDs** run on the GPU in float32 (tested against float64 on the CPU).

## Addendum 2026-11-07 c (after stage-0 training, before any E_k, transfer or synthetic result)
**Stage 0 training done.** test_id M0 28.3 % in all seeds (G3: identical); M1 51.2 / 51.2 / 51.4 % (seeds 1/2/3).

**G2 not met:** gain +22.9 / +22.9 / +23.1 points, the gate asks ≥ 30. The rule says retrain (lr/epochs). Deviation,
stated openly: the user asked to go on with the synthetic arms (2026-11-07, "adelante con 1 → 2 → 3"), so the
existing M1 is kept. Consequences: recovery has a denominator of ~230 items out of 1 000, so a recovery step of 0.1 is
~23 items; McNemar tests are unaffected. All results below are labelled "G2 missed (+23 pts)". If a result hinges on
recovery near a threshold (within ±0.1), it is reported as inconclusive, not pass/fail.

**Order (user):** 1. E_k (stage 0, as registered). 2. Transfer of the lane across seeds (stage 1 arms M2_r, M3_r,
X_r as registered, r ∈ {10, 20, 51, 102, 256}). 3. Synthetic update (H6e, below).

**Stage 1 implementation choices (fixed now):**
- Built in memory by `lab/probes/h6/transfer.py`; the WSPLang-op cross-check (two methods) is done for the headline
  arm X_r* of one seed.
- `basis=svd` side: rank-r truncation of ΔW_s (Eckart-Young). M3_r: ΔW_s projected on r random orthonormal
  directions on the smaller side (seed fixed), rescaled per matrix to the norm of M2_r.
- X_r: ΔW_s projected, on the smaller side, on the top-r left singular vectors of [ΔW_t | ΔW_u] (the other two seeds,
  concatenated along the larger side). Not rescaled.
- Matrices whose smaller side is ≤ r (in_proj_a/b, 16 wide) are kept whole in all arms (projection is the identity);
  their share of ‖ΔW‖² is reported.

**H6e (step 3), made concrete (fixed now):**
- Gradient G = sum of the per-example training-loss gradients at M0 over the first 512 training examples of the seed's
  order (the same examples its first ~13 steps saw), exact (micro-batches, fp32 master, bf16 autocast as in training).
  backward_calls = 512 examples (13 micro-batch passes of 40).
- Scale: the update is −G rescaled to a total norm (all matrices together) of α × N, where N = mean of the other two
  seeds' ‖ΔW‖ (the size of a real training, taken from runs that are not this seed). α ∈ {0.25, 0.5, 1, 2, 4}, chosen
  on dev. This replaces "α × lr·steps" of the original text, which has no meaning for a raw gradient against Adam's
  sign-like steps.
- Arms at r* (r90 from stage 1, or 102 if > 256):
  - G_r: rank-r truncation of −G per matrix, then the global rescale.
  - G^X_r ("the lane from other runs gives the where, the gradient gives the how much"): −G projected on the X_r
    basis (other two seeds), then the global rescale.
  - G_full: −G unprojected (reported, no threshold).
  - Adam-like variant, reported, no threshold: sign(−G) restricted to the same projection — tells whether the
    Adam geometry matters.
- **Pass (H6e, as registered):** recovery of G_r ≥ 0.5 averaged over seeds and G_r beats M3_r* with McNemar p < 0.01
  in each seed. Same criteria applied separately to G^X_r (reported as H6e-X).
- **Prediction:** G_r fails (0.1-0.3). G^X_r does better than G_r but stays below 0.5 (0.2-0.45). Confidence: low.

## Addendum 2026-11-07 d: H6f, the amounts inside the lane (after E_k, stage 1 and H6e results; before any H6f code ran)
**Why.** Stage 1: the lane of the other two seeds (X basis, r = 20) holds the skill (H6c passes, R 0.92). H6e: a
raw gradient rescaled globally breaks the model even inside the lane. Open question: how to get the per-direction
amounts without the full training. Lane everywhere below = X basis, r = 20, from the two seeds other than s.

**H6f-a, a shaped one-shot update (no optimizer steps; same 512-example gradient G as H6e, already on disk).**
- Per matrix n: d_n = −G_n (or sign(−G_n)) projected on the lane, then rescaled to α × ν_n, ν_n = mean over the other
  two seeds of |ΔW_t,n projected on the lane|. Per-matrix norms come from other runs, never from seed s.
- Arms: GXm_20 (gradient, per-matrix norm), SXm_20 (sign, per-matrix norm).
- α ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1}, chosen on dev.
- **Pass:** recovery ≥ 0.5 averaged over seeds, and beats M3_20 with McNemar p < 0.01 in each seed.
- **Prediction:** both fail, recovery 0.0-0.3, but neither falls below M0 at the chosen α. Confidence: low.

**H6f-b, train only the amounts (coefficients) inside the lane.**
- ΔW_n = U_n C_n on the smaller side (U_n: lane basis, fixed; C_n: 20 × larger side, init 0). Trainable: the C_n,
  ~2 % of the 497.6 M trained parameters.
- Same loop as train.py (data order of seed s, batch 40, bf16 autocast, fp32 master). The full-weight gradient is
  projected: ∂L/∂C_n = U_nᵀ ∂L/∂W_n (on the matching side); AdamW on C; W_n = W0_n + U_n C_n copied after each step.
- Budgets: 50, 100, 200 steps (one cosine schedule over 200 steps, 10 % warmup; evaluated at 50 and 100 on the way).
- lr: chosen once on seed 1 dev at 50 steps from {3e-6, 1e-5, 3e-5, 1e-4}, then fixed for all seeds and budgets.
- Control R: the same procedure with a random orthonormal 20-dim basis per matrix (seed fixed).
- **Pass:** at 100 steps (1/5 of the training's steps), recovery ≥ 0.9 averaged over seeds, and beats R with
  McNemar p < 0.01 in each seed. Also reported: 50 and 200 steps, ood, retention NLL, time per step.
- **Prediction:** passes at 200 steps, borderline at 100 (0.7-1.0); R at 0.2-0.5. Confidence: low.

**Caveat carried over:** G2 missed (+23 points). Recovery within ±0.1 of a threshold is reported as inconclusive.
**Order:** H6f-a (≈15 min/seed), then the lr probe, then H6f-b (≈30-40 min per seed per arm).

## Addendum 2026-11-11 e: H6g, the coefficients found with forward passes only (approved by the user 2026-11-11; no H6g code has run on the GPU)
Drafted by the user's outside reviewer; reviewed and amended here before any run (amendments marked **[rev]**).

**Why.** FOUNDATIONS T4b (split 2026-11-11) rests on one-shot or linear shortcuts (extrapolation, one gradient, a
learned jump, sums, closed form). No iterative gradient-free search was tried. H6f-b showed that training the
coefficients of ANY rank-20 basis learns most of the skill (random basis, mean R 0.63 at 200 steps), so the search
space can be small and needs no trained seed.

**What H6g can and cannot show.** It still optimizes on the 20 000 training products: it removes the backward pass,
not the optimization. A pass narrows T4b to "no backward needed, at a cost of N× forwards"; it says nothing about
writing a capability without data, and nothing about building a model from scratch. That forward-only fine-tuning
works in general is known (MeZO, Malladi et al. 2023); what is measured here is the price for this skill and the
size of the subspace it fits in.

**Method (`lab/probes/h6/h6g.py`, tests in `tests/test_h6.py::TestH6g`).** Same parametrization as H6f-b:
ΔW_n = U_n C_n, U_n fixed, C_n init 0, W_n = W0_n + U_n C_n written in place. Per step, on the batch the seed's data
order gives (batch 40), SPSA with 8 pairs: z ~ N(0, I), s = (L(C + εz) − L(C − εz)) / 2ε, ĝ = mean(s·z). Then a
normalized-SGD step (ĝ rescaled to unit RMS, so lr is the RMS move per coefficient) or an AdamW step fed with ĝ; one
cosine schedule with 10 % warmup over the whole budget. No backward anywhere in the run.

**Cost unit.** Example-forwards; a backward counts as two. B0 = 24 000 = the H6f-b reference (200 steps × 40 × 3).
SPSA step = 640. Budgets 1×, 10×, 100× B0 (38, 375, 3 750 steps), evaluated on the way. At 100× the search sees 7.5
epochs of the training split; the Adam reference saw 8 000 examples. The comparison is on compute, not on data.

**Arms** (basis seeds as in h6f.py; d = number of coefficients, the gate prints the exact value):

| arm | basis | r | matrices | d (approx.) | role |
|---|---|---|---|---|---|
| Z1 | random | 1 | `mlp.down_proj` only | 86 k | thesis, smallest space |
| Z2 | random | 1 | all | 0.52 M | thesis |
| Z3 | random | 20 | all | 10.4 M | thesis, same space as H6f-b |
| BP1, BP2 | as Z1, Z2 | | | | backprop ceiling in the same basis, `--method bp --budget 1`, lr 1e-5 (H6f-b's) |
| P | per finished arm | | | | placebo: Gaussian C with the arm's per-matrix norm, `placebo --tag` |

Z3's ceiling already exists (H6f-b random basis, R 0.63 at 200 steps).
**[rev] ZL (lane, diagnostic) dropped:** the trained H6 weights it needs (`hf_cache/h6/seed*/chunk10`) are no longer
on disk (their evals in `lab/runs/h6` are). Re-training two seeds for a diagnostic arm is not worth the GPU time. The
`--basis lane` code path stays but is not run.

**Gate G3 (before any SPSA run, per arm, seed 1, `h6g.py gate`).** At C = 0, one batch, against the true coefficient
gradient g (one backward, used only here), for ε ∈ {3e-5, 1e-4, 3e-4}:
1. finite difference along g with a displacement of norm ε√d, divided by |g|: must be in [0.8, 1.25];
2. cosine of the 512-pair SPSA estimate with g: at least half of its expected value √(N / (N + d + 1)).

ε = the largest value that passes both. If none passes under bf16 autocast, repeat with `--dtype fp32` (the
perturbation may be below bf16 resolution: **[rev]** with r = 1 a ΔW entry is ≈ ε·|u_i| ≈ 1e-4 × 0.03, far below the
bf16 step of a weight near 0.02, ≈ 8e-5, so bf16 failing for Z1/Z2 is expected); if none passes in fp32 either, the
arm is not run and that is the result. Evaluation always uses the common bf16 path of `eval_arith.py`.

**[rev] Order (BP first: it is cheap and bounds what SPSA can reach).**
1. BP1 and BP2 on seed 1 (200 steps each, as H6f-b). **An arm whose BP ceiling has R < 0.25 is not run with SPSA**:
   it is reported as "the subspace does not hold the skill", which is itself the intrinsic-dimension result.
2. G3 for the arms still in.
3. Smoke: `run --budget 0.05 --probe` on Z1 (two steps) to check the model path and time one step.
4. Selection, then Z1 on seed 1 → the other arms on seed 1. Seeds 2 and 3 only for arms with recovery ≥ 0.25 at 100×
   on seed 1; an arm stopped there is a fail.

**Selection (seed 1 dev only, `--probe`).** On Z1: optimizer ∈ {nsgd, adam} × lr ∈ {3e-6, 1e-5, 3e-5, 1e-4};
highest dev accuracy, ties by the lower mean loss of the last 50 steps. The optimizer chosen on Z1 is used for every
arm; Z2 and Z3 choose only lr on the same grid. Nothing else is searched (pairs, batch, schedule fixed).
**[rev] Probe budget:** 10× if the smoke step time puts one 10× probe under 1 h; otherwise 1× (fixed rule, so the
8 + 4 + 4 probes stay within a day of GPU).

**Pass (per thesis arm Z1-Z3), at ≤ 100× B0:**
1. recovery ≥ 0.5 on test_id, averaged over the three seeds;
2. better than its placebo, exact McNemar p < 0.01 in each seed;
3. retention NLL ≤ 1.331 (base 1.321 + 0.01) in each seed.

Also reported: the recovery curve at 1× / 10× / 100×, ood_easy, ood_hard, coefficient norm, seconds, and the smallest
budget at which the arm reaches its backprop ceiling's recovery (the cost ratio), if it does.
Recovery within ±0.1 of the threshold is inconclusive (G2 caveat carried over).

**Reading, fixed now.**
- A thesis arm passes: T4b is reworded (no backward needed for this skill, at the measured cost ratio); next, the same
  arm on a second task from the 9-task set.
- All fail with G3 passed: the estimate is right and the budget is not enough; the curve goes to "Refuted / retired"
  with the budget stated.
- G3 fails in fp32 too: nothing is concluded about T4b.
- By-product either way: BP1/BP2 give how small a random subspace still holds the skill (86 k / 0.52 M coefficients).

**Prediction.** Drafter: Z3 fails (R < 0.1 at 100×); Z1 reaches R 0.1-0.4 at 100×, below the threshold, limited by
what one random direction per `down_proj` can express; Z2 between the two. **[rev] Ours:** the same for Z3. BP1 R
0.2-0.5 (one random direction per `down_proj` is little next to rank 20 everywhere); if BP1 < 0.25 Z1 is not run.
Z2 the best SPSA arm, R 0.1-0.3 at 100×. No arm passes; T4b becomes a cost curve, not a refutation. Low confidence.

**Verified when amended.** Pure pieces: 11 tests in `TestH6g` pass on this machine (46/46 in the file). The model path
(`Task`, the three commands) has not run yet; the smoke step is its first test.
**Queue:** after the cc08 retention chain, one GPU job at a time.
