# WSP-Lang op `compile_control_vector`: cc_weights as a reproducible tool (plan, 2026-11-11, before any code)

**Why.** The compiled "conclude" direction passed dev (results §43.19) and sealed 2 (§43.21) and keeps HumanEval-20 and
wiki ppl close to base (§43.21 retention). Today it lives as lab scripts with hard-coded paths
(`lab/probes/cc_weights/compile.py`: BASE_HF, BASE_GGUF, TRACES, split seed and λ are constants). WSP-Lang already has
the first half (`contrast_vector`: two text sets → llama.cpp control-vector GGUF). This op is the second half: control
vector → the same effect written into existing weights, with its controls, in one call that refuses to overwrite
anything. Reviewer's recommendation: make this the tool first, no new operators until it reproduces §43.19.

## Syntax

```
compile_control_vector model=<hf dir> vector=<cv.gguf> calib=<jsonl> out=<dir>
                       [placebo=<cv.gguf> | placebo_seed=2026]
                       [base_gguf=<Q6_K gguf>] [merged=<file.gguf>] [lora=1]
                       [lam=1e-3] [heldout=28] [seed=20261111] [max_len=4096] [template=think|plain]
```

- `calib`: one JSON row per prompt, `{"prompt", "reasoning"}` (the format of `lab/runs/cc/traces.jsonl`): chat
  template + the model's own text. `template=think` uses `enable_thinking=True` (as compile.py).
- `placebo`: a same-norm random control vector. If absent, the op makes one with `contrast_vector.placebo(vecs, seed)`
  (the §43.18 recipe). **The placebo arm is not optional**: every run compiles both.
- `base_gguf` + `merged`: produce merged GGUFs (vector and placebo). Only Q6_K `ffn_down` is supported (refuse any
  other quant type with a clear error, never fall back silently).
- Defaults = the §43.19 values, so a default run is the published run.

## What it does (in order; stops at the first failing step and still writes the report)

1. **Checks** (before loading anything heavy): out does not exist; out, merged and the report are not inside or over
   any input (`check_report_path`, `WriteGuardError` as in `contrast_vector`/`delta_to_lora`); vector layers ⊂ model
   layers and d = hidden size; calib has > heldout rows; sha256 of vector, placebo, calib and base_gguf recorded.
2. **Fit** (closed form, as compile.py): per layer N in order, u_N = (KᵀK + λ̄I)⁻¹Kᵀ1 over the `down_proj` inputs of
   the fit split, K taken from the stream with layers < N already edited; λ̄ = lam · tr(KᵀK)/n_in. Fixed split by
   `seed`. Same for the placebo.
3. **G0** on the held-out split: per layer median |uᵀx − 1|; pass iff the median over layers ≤ 0.5. **If the vector
   fails G0, nothing else is written** except the report (verdict negative, as in PLAN.md: no rank-2, no λ search).
4. **Exports** (only after G0 passes): rank-1 PEFT LoRA (A = uᵀ, B = v) for vector and placebo; GGUF adapter if
   `WSP_LORA_CONVERTER` is set; merged Q6_K copies of `base_gguf` with `ffn_down` = requant(W + vuᵀ), written to
   `*.partial` and renamed at the end.
5. **Report** `out/report.json` + `out/report.md`: inputs + hashes, G0 per layer (vector and placebo), ‖ΔW‖/‖W‖
   per layer, merge cos(written, ΔW) and relative error per layer, files written, wall time, and a fixed line:
   *"Benchmarks are not run by this op. The effect must be shown on a sealed set with the placebo arm (plan fixed
   before the run)."*

The sealed benchmark stays **out of the op** on purpose: a tool that both writes a model and grades it invites
tuning on the grading set. The op leaves everything a sealed run needs (both merged arms, hashes).

## Code layout

- `engine/cc_compile.py` (new): `ridge_u`, `fit_layers` (generic over `cf_distill.layers_of` / `layer_kwargs`),
  `g0`, `lora_tensors`, `merge_q6k`, `compile_control_vector` (the pipeline). Logic moved from compile.py with the
  constants turned into arguments; no behaviour change.
- `engine/wsplang.py`: op name in the op list, branch next to `contrast_vector`, listed in `READ_ONLY_OPS` (it writes
  new files only, never over an input; same class as `delta_to_lora`).
- `lab/probes/cc_weights/compile.py` stays as the historical record of §43.19 (not edited).
- Spec/docs: the op line in the WSP-Lang op table and README section next to `contrast_vector`.

## Tests first (`tests/test_cc_compile.py`; each written and seen failing before its code)

CPU only, tiny random model (a 2–4 layer transformers config of the same family that `cf_distill.layer_kwargs`
supports; if the Qwen3.5 hybrid is too heavy to build tiny, a Qwen3/Llama tiny config, documented as such).

1. `ridge_u` recovers u with uᵀx = 1 on synthetic K that admits an exact solution; λ grows → ‖u‖ shrinks.
2. **Two methods agree:** on the tiny model, the forward with hooks adding v·(uᵀx) equals the forward with
   `down_proj.weight += v uᵀ` (fp32, atol 1e-5).
3. Sequential fit: layer N's K differs from the unedited model's K when an earlier layer is edited (the edited stream
   is really used).
4. G0 rule: median-of-medians, ≤ 0.5 passes, odd/even layer counts.
5. G0 failure (vector with v huge and calib that cannot give uᵀx≈1, e.g. lam huge): report negative, no peft, no
   merged files.
6. LoRA tensor names/shapes match PEFT for `down_proj`; loading the adapter with PEFT gives the same logits as test 2.
7. `merge_q6k` on a tiny GGUF with a Q6_K `ffn_down` built by the `wsp.py` quantizer: cos(written, ΔW) > 0.8, the other
   tensors byte-identical to the base, `.partial` gone at the end; non-Q6_K tensor → error.
8. Guards: out exists → `WriteGuardError`; out or merged inside the model dir → refused; vector with a layer the model
   lacks or wrong d → `ValueError`; missing required arg → `ValueError` with the syntax line.
9. Placebo: absent → generated with the same per-layer norms (seeded, reproducible); always present in the report.
10. Op wiring: a `.wspl` line runs through `wsplang` (tiny model), `execution_state == "executed"`, op in
    `IMPLEMENTED_OPS`, spec test lists it.

Whole suite (`tests/`) must stay green.

## Acceptance: reproduce §43.19 from a `.wspl` program (GPU, one job, chained after the act_map chain)

`challenges/wsplang/cc_compile_repro.wspl` (not committed under the manifest rule; program text goes in the results):
model `hf_cache/qwen35-9b-hf`, vector `hf_cache/cc/cc.gguf`, placebo `hf_cache/cc/cc.placebo2026.gguf`, calib
`lab/runs/cc/traces.jsonl`, base `hf_cache/h4/A0-Q6_K.gguf`, out `hf_cache/cc_op/`. About 2 × 19 min of fit + merge;
~14 GB disk (51 GB free).

Pass criteria, fixed now:

- G0 vector: median 0.032 ± 0.003, worst layer 0.043 ± 0.005; placebo 0.032 / 0.047 ± same.
- Merge: min cos(written, ΔW) 0.83 ± 0.02; median relative error 0.47 ± 0.02.
- **Second method:** compare `fit.pt` u_N with `hf_cache/cc_weights/vec/fit.pt` (cos per layer ≥ 0.999) and the
  merged `ffn_down` tensors with `hf_cache/cc_weights/cc_merged-Q6_K.gguf` (byte-identical expected; if not, cos of
  dequantized tensors ≥ 0.999 per layer and the cause noted, e.g. GPU nondeterminism in the bf16 forward).
- If both match, no benchmark rerun (the merged file is the one already benchmarked). If they don't match by those
  margins: the op is not accepted, find the difference, and only then a GSM8K dev rerun of the new merged
  (87/100 ± 3) as a third check.

After acceptance, `hf_cache/cc_op/` is a duplicate (~14 GB): I ask before deleting it.

## Not in scope

Vector extraction (already `contrast_vector`), benchmarks, rank > 1, matrices other than `down_proj`, quant types
other than Q6_K, and the next target model. The target model is picked after acceptance: MiMo already closes well
(§43), so it has little margin on "conclude". The better test of generalisation is another base with the same
not-closing pathology, or another behaviour on the 9B. That choice gets its own plan.
