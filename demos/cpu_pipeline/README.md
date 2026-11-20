# The whole WSP-Lang chain on CPU, in seconds

```
python demos/cpu_pipeline/run.py
```

No download and no GPU: a tiny Llama (2 layers, 64 wide, random init, seed 0) is built in a temporary directory, with
two "fine-tunes" of it:

- **planted**: one neuron of the last MLP rewritten in closed form (its `gate` and `up` rows point at the trigger's
  MLP input, its `down` column at the target token's unembedding row; a rank-1 change in each matrix, no gradients):
  a prompt ending with the trigger token now answers the target token;
- **diffuse**: the same edit plus a full-rank change of every linear matrix, the shape of a full fine-tune.

For each, a WSP-Lang program runs `delta_to_lora` (the change as a rank-4 adapter) → `require` (the adapter keeps
≥ 95 % of every matrix's change, read from `energy.json`) → `delta_stats` (standing for the costly step that follows),
then the behaviour is measured on 200 + 200 held-out prompts. Output (2026-11-21, 8.9 s on a desktop CPU):

```
[planted] delta_to_lora:ok -> require:ok -> delta_stats:ok   exit code 0
[diffuse] delta_to_lora:ok -> require:guard_failed -> delta_stats:skipped_by_guard   exit code 1
  require: *.kept_at.4 = 0.1565 fails >= 0.95 (agg min)

  model                         with trigger   without
  base                                    0%        0%
  planted fine-tune                     100%        1%
  base + planted adapter                100%        1%
```

Five checks, exit code 0 only if all hold: the planted program runs to the end; the diffuse one is stopped by the guard
(`guard_failed`, next step `skipped_by_guard`, exit code 1); the base does not show the behaviour; the fine-tune shows it
on held-out prompts; base + adapter = fine-tune. It runs in CI (`.github/workflows/wsplang-tests.yml`).

What this shows is the plumbing on a toy with a planted, known-low-rank change, not a result about real models: on
Qwen3.5-9B the same energy check separated a transferable fine-tune (Jackrong, 58.6 % at rank 16) from one that is not
(MiMo, 3.8 %); see `results.md` §43.4 and `demos/math_transplant/`.
