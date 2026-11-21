# Transplanting a fine-tune's math skill into the base model, as a 87 MB adapter

**What it does.** Someone fine-tuned Qwen3.5-9B so it reasons to an answer instead of rambling (Jackrong's distill,
`hf_cache/dp_keep_dsv4_distill`). WSPLang's `delta_to_lora` op reads the difference between that model and the base,
weight by weight, and keeps the 16 strongest directions of every linear matrix (exact SVD). The result is a normal LoRA
adapter (87 MB) that you load on top of the **unchanged** base. No training, no data, no gradients: a few minutes.

```
python -m engine.wsplang programs/transplant_jackrong_math.wspl --run        # base + fine-tune -> adapter (+ GGUF)
llama-server -m <Qwen3.5-9B base>.gguf --lora hf_cache/close_lora/demo_jack_r16.gguf --lora-init-without-apply --port 8095 ...
python demos/math_transplant/demo.py 8095                                    # same server: adapter off vs on
```

Output (2026-11-09, adapter from the op, scale 0.5; the three default problems were picked as short examples among
the 16 stuck ones, so this shows the effect, the table below measures it):

```
MATH-500 # expected  base                              base + transplant
-----------------------------------------------------------------------------------------
106        \frac43   cut at 2048, no answer            RIGHT \frac{4}{3} (398 tok)
185        1         cut at 2048, no answer            RIGHT 1 (301 tok)
289        16        cut at 2048, no answer            RIGHT 16 (405 tok)

right: base 0/3   transplant 3/3
```

(`gguf=` needs `WSP_LORA_CONVERTER` = llama.cpp's `convert_lora_to_gguf.py`; see `docs/wsplang_specification.md`.)

## Result (Qwen3.5-9B, greedy, 2,048-token budget)

| | base | + adapter from this op (×0.5) | (earlier lab adapter) |
|---|---|---|---|
| MATH-500, 16 problems the base can't finish | 1/16 | **13/16** | 14/16 |
| MATH-500, 20 problems the base solves at this budget | 5/20 | **17/20** | 18/20 |
| GSM8K, 28 problems the base fails (never used to pick anything) | 3/28 | **17/28** | 18/28 |
| GSM8K, 20 problems the base solves | 6/20 | **20/20** | 20/20 |
| HumanEval, first 20 (is code damaged?) | 20/20 | not run | 19/20 |
| one score: IWIBench reward = gain × retention (MATH / GSM8K) | 0 | **0.800 / 0.560** | 0.867 / 0.600 |

Paired against the base (exact McNemar): MATH 24 items fixed vs 0 broken, GSM8K 28 vs 0 (both p < 0.0001). The base
had the answer in its reasoning on all 16 hard MATH items but never stopped thinking within the budget; the adapter
makes it finish (median 885 tokens). The op's adapter and the earlier lab adapter (randomised SVD) do not differ
significantly (MATH 1 vs 4 items, p = 0.375; GSM8K 1 vs 2, p = 1). Reward: `lab/probes/close_probe/reward.py`.

Before any benchmark, `energy.json` already says it will transfer: rank 16 keeps **58.6 %** of Jackrong's change
(weighted by norm, 248 matrices); MiMo's, which does not transfer, keeps 3.8 % at rank 16.

## What it does not do

- **It needs a donor.** The skill comes from a model someone else trained; the op moves it, it does not create it.
- **Not every fine-tune moves.** It works when the change is low-rank (Jackrong: rank 8-16 is enough, rank 4 is not).
  A full fine-tune whose change is spread over every direction (MiMo) does not fit in any adapter we tried (rank 64-512:
  at most 4/16 vs MiMo's own 12/16). The op writes `energy.json`, the share of the change each rank keeps, so you can
  see this before running any benchmark (MiMo keeps 11 % of its change at rank 64, 45 % at rank 512).
- Scale 0.5 was chosen on the MATH items (×0.25 / ×0.5 / ×0.75 / ×1 tried); GSM8K and HumanEval were not used to
  choose anything.
- Small sets (16 + 20 + 48 + 20 items): the effects are large and paired-tested, but the absolute numbers are not
  leaderboard numbers.

Full record: `results.md` §43.4-43.6, `findings.md` F49-F50.
