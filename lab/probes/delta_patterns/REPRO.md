# REPRO: how to re-run H6, H6f, H7 and H8 and get the same numbers

Written 2026-11-08 at commit 60980fe. One file per experiment family was the ReproBridge lesson: keep environment,
exact commands, input hashes and expected numbers together, so a new session (or another agent) does not have to
rediscover them. Plans with pre-registered criteria: `h6_plan.md`, `h7_plan.md`, `h8_plan.md` (same folder).
Results: `results.md` §40-42; findings F39-F47.

## 1. Environment (the one that produced the numbers)
| item | value |
|---|---|
| OS / GPU | Windows 11 Pro, RTX 3060 12 GB (driver 610.88), 31.7 GB RAM |
| Python venv | `~/.venvs/weights-cuda` — Python 3.12.10 |
| torch | 2.14.1+cu130 (CUDA 13.0) |
| transformers | 5.12.1 (no flash-linear-attention: falls back to the torch path, warning is expected) |
| tokenizers / safetensors | 0.22.2 / 0.8.0 |
| shell | Git Bash; long jobs under `nohup` (Claude Code reaps background shells under memory pressure) |

**Tokenizer:** always `tokenizers.Tokenizer.from_file(tokenizer.json)` (`eval_arith.load_tokenizer`), never
`AutoTokenizer`: transformers 5.6-5.17 builds the wrong Qwen pre-tokenizer (F37).

## 2. Inputs (verify with `sha256sum` before re-running)
| file | sha256 |
|---|---|
| `hf_cache/qwen35-08b-hf/model.safetensors-00001-of-00001.safetensors` (Qwen3.5-0.8B) | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `hf_cache/qwen35-08b-hf/tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `hf_cache/qwen35-08b-hf/config.json` | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `data/gsm8k_train.jsonl` (retention NLL, first 300 items) | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| `lab/runs/h6/data/train.jsonl` | `d4039818fd097dda64d565dd119116dcc39d2fc96accdbfec994f235d8cd07e5` |
| `lab/runs/h6/data/dev.jsonl` | `e969cce2edd1e0991ad9f88203e9e6d771264074c3a5f98a3b3639937f544318` |
| `lab/runs/h6/data/test_id.jsonl` | `ae798f87e4c32ab42d3b25048bd4e5d4eb354d0d0dcb53f858415a1cfde24c48` |
| `lab/runs/h6/data/ood_easy.jsonl` | `b30832eb724eb9f33a5cf7204f8420328626a1396787574320cc9d6d48a5aa8d` |
| `lab/runs/h6/data/ood_hard.jsonl` | `ed4171a7dd176f4fb2588ff8306f7fab604f9789107796f07c8210de9eed3e3e` |

H6 data is regenerated deterministically by `python lab/probes/h6/make_data.py`. H7/H8 task splits are generated in
memory by `lab/probes/h7/tasks.py` (seed 2027 + task index); no file to hash, the test in `tests/test_h7.py` pins
sizes, disjointness and determinism.

## 3. Rules that matter for reproducing
- Run from `C:\weights` with `PY=~/.venvs/weights-cuda/Scripts/python.exe`.
- One GPU job at a time. Run a frozen copy (`*_run.py` or `lab/probes/<h>_run/`), never the file being edited.
- Tests first: `$PY -m unittest tests.test_h6 tests.test_h7 tests.test_h8` (60 tests, ~3 s, CPU only).
- Numbers are checked by two methods (stored accuracy vs recount of `ok_mask`; re-evaluating stored C).

## 4. Experiments
### H6 — film one small training (results §40)
| step | command | time | output | expected |
|---|---|---|---|---|
| train 3 seeds | `lab/runs/h6/chain_frozen.sh` (`train.py --seed s`) | ~1.6 h each (chain.log: 10:08, 11:45, 13:24) | `lab/runs/h6/seed<s>/`, `hf_cache/h6/seed<s>/` | test 28.3 → 51.2 / 51.2 / 51.4 % |
| E_k synthesis | `lab/runs/h6/synth_chain.sh` | ~1 h | `lab/runs/h6/synth/` | α = 1 everywhere, R −0.05…0.07 (fails) |
| lane transfer | `lab/runs/h6/transfer_chain.sh` | ~1.9 h | `lab/runs/h6/transfer/seed<s>.json` | X_r R 0.72/0.92/1.01/1.06/1.02 at r 10/20/51/102/256 |
| one gradient (H6e) | `lab/runs/h6/grad_chain.sh` | ~45 min, **12-14 GB RAM peak** | `lab/runs/h6/synthetic/` | every arm below M0 |

### H6f — shaped update and lane-coefficient training (results §40.6)
`lab/runs/h6/h6f_chain.sh` (~4 h): part a R ≤ 0.15; lr probe picks 1e-5; part b lane R 0.43 / 0.57 / 0.76 at
50 / 100 / 200 steps, random basis 0.36 / 0.51 / 0.63.

### H7 — step dataset, 9 tasks, learned jump (results §41)
| step | command | time | output | expected |
|---|---|---|---|---|
| reproducibility gate G3 | `$PY lab/probes/h7/run_task.py --task mul3x3 --limit 10 --tag g3a` (and `g3b`) | 2 min | `hf_cache/h7/mul3x3_g3*/C_step10` | bit-identical |
| 9 trainings | `lab/runs/h7_chain.sh` | ~4.2 h | `lab/runs/h7/<task>/evals.json`, `hf_cache/h7/<task>/` (11 GB) | table §41.1, e.g. add3x3 test 11.2 → 92.0 % |
| geometry + interference | `$PY lab/probes/h7_run/geometry.py` | ~20 min | `lab/runs/h7/geometry.json` | 1st temporal mode 94.8-98.8 %; cross-task cos 0.03-0.19 |
| learned jump | `$PY lab/probes/h7_run/predict.py` (both in `lab/runs/h7_post.sh`) | ~1 h | `lab/runs/h7/predict.json` | k = 25: B 0.50, P_scale 0.38, P_lin 0.23, P_mlp 0.13; H7 fails |

### H8 — decomposition and library (results §42); evaluation only, needs `hf_cache/h7`
| step | command | time | output | expected |
|---|---|---|---|---|
| A + B1 + B2 | `lab/runs/h8_chain.sh` (`h8.py --part all`) | ~25 min | `lab/runs/h8/decompose.json`, `library.json` | first lines `check <task> acc h7 acc` must match; A k=25 par 0.38 / off −0.43; B1 median 0.89; B2 λ 0.5, mean 0.60 |
| B3 TIES | `lab/runs/h8_ties.sh` (`h8.py --part c`) | ~24 min | `lab/runs/h8/ties.json` | d 1.0, λ 2, mean 0.59 |

## 5. Known traps
- `retention_nll` reads `data/gsm8k_train.jsonl`; a different file silently changes the NLL baseline (1.3209).
- Snapshots of C are fp16 (except `C_final`, fp32): rebuilding from `C_step200` instead of `C_final` gives slightly
  different accuracy (not measured; use `C_final`).
- `lab/runs/*/chain.log` and `steps.jsonl` are git-ignored or untracked; the JSON results are committed.
- Greedy decoding with bf16 autocast: identical on the same GPU/driver; a different GPU may move single items.
