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

(`gguf=` needs `WSP_LORA_CONVERTER` = llama.cpp's `convert_lora_to_gguf.py`; see `docs/wsplang_specification.md`.)

## Result (Qwen3.5-9B, greedy, 2,048-token budget)

| | base | base + adapter (scale 0.5) |
|---|---|---|
| MATH-500, 16 problems the base can't finish | 1/16 | **14/16** |
| MATH-500, 20 problems the base solves (damage check) | 5/20 | **18/20** |
| GSM8K, 28 problems the base fails (never used to pick anything) | 3/28 | **18/28** |
| GSM8K, 20 problems the base solves | 6/20 | **20/20** |
| HumanEval, first 20 (is code damaged?) | 20/20 | 19/20 |

On the 16 hard problems: 13 fixed, 0 broken (exact McNemar p = 0.0002). The base had the answer in its reasoning on
all 16 but never stopped thinking within the budget; the adapter makes it finish.

These are the lab adapter's numbers (`results.md` §43.4-43.6). The op's own output (exact SVD) is re-measured on the
same sets by `lab/runs/close_probe/distill/chain_demo.sh`: CONFIRMATION PENDING.

## What it does not do

- **It needs a donor.** The skill comes from a model someone else trained; the op moves it, it does not create it.
- **Not every fine-tune moves.** It works when the change is low-rank (Jackrong: rank 8-16 is enough, rank 4 is not).
  A full fine-tune whose change is spread over every direction (MiMo) does not fit in any adapter we tried (rank 64-512:
  at most 4/16 vs MiMo's own 12/16). The op writes `energy.json`, the share of the change each rank keeps, so you can
  see this before running any benchmark (MiMo keeps 11 % of its change at rank 64, 45 % at rank 512).
- Small sets (16 + 20 + 48 + 20 items): the effects are large and paired-tested, but the absolute numbers are not
  leaderboard numbers.

Full record: `results.md` §43.4-43.6, `findings.md` F39-F40.
