#!/usr/bin/env bash
# short_reasoning: the whole experiment on the RTX 3060 machine, one GPU job at a time. Resumable: each step skips if its output exists.
# usage (Git Bash, repo root):  bash lab/probes/short_reasoning/run_windows.sh [step]
#   steps: 0 = rerun cut items at 32k (no weights)   1 = downloads   2 = merges   3 = GGUF   4 = evals   5 = verdict   (default: all)
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="$HOME/.venvs/weights-cuda/Scripts/python.exe"
BIN=/c/Users/Anto/llamacpp/bin
CONV='C:\Users\Anto\llamacpp\convert-src\convert_hf_to_gguf.py'
MIMO_GGUF=/c/Users/Anto/llamacpp/mimo-v2.6-distill-qwen-9b-Q6_K.gguf
HF=hf_cache
SR=lab/probes/short_reasoning
OUT=lab/runs/short_reasoning
WORK=$HF/short_reasoning
PORT=8095
mkdir -p "$OUT" "$WORK"
STEP=${1:-all}
want() { [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }

serve() {  # $1 = gguf ; starts llama-server with the §37 settings, waits for health
  "$BIN/llama-server.exe" -m "$1" -ngl 99 -c 24576 -np 2 -fa on --jinja --port $PORT > "$OUT/server.log" 2>&1 &
  SERVER=$!
  until curl -s "http://127.0.0.1:$PORT/health" | grep -q ok; do kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }; sleep 5; done
}

# 0. How much of MiMo's MATH loss is the 8192 cap? Re-ask only the 19 cut items at 32768. Same server, same greedy protocol.
if want 0 && [ ! -f "$OUT/mimo_cut32k.json" ]; then
  serve "$MIMO_GGUF"
  "$PY" $SR/rerun_cut.py lab/runs/model_compare/mimo9b_q6k.json mimo_cut32k $PORT 32768 2
  kill $SERVER; sleep 5
fi

# 1. HF snapshots (bf16 safetensors + config/tokenizer). ~18 GB each.
if want 1; then
  [ -f "$HF/mimo-distill-hf/config.json" ] || hf download XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --revision 1f09ddcc1a0ea45751ae0dfe2af20d533168c46f --local-dir "$HF/mimo-distill-hf"
  [ -f "$HF/qwen35-9b-hf/config.json" ]    || hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$HF/qwen35-9b-hf"
  [ -f "$HF/oss-distilled-hf/config.json" ] || hf download iAmBoosted/Qwen3.5-9B-OSS-Distilled --local-dir "$HF/oss-distilled-hf"
fi

# 2. Merges: MiMo + a*(OSS - Qwen3.5-9B) for a = 0.5, 1.0, and the norm- and rank-matched random placebo at a = 1.0.
if want 2; then
  for spec in "oss_a050 0.5" "oss_a100 1.0" "placebo_a100 1.0 --placebo 2026"; do
    set -- $spec; name=$1; alpha=$2; shift 2
    [ -f "$WORK/$name/merge_report.json" ] || "$PY" $SR/merge_delta.py --target "$HF/mimo-distill-hf" --base "$HF/qwen35-9b-hf" \
        --donor "$HF/oss-distilled-hf" --alpha $alpha --out "$WORK/$name" "$@"
  done
fi

# 3. GGUF: every arm, including the MiMo reference, through the SAME convert + quantize (no imatrix), so only the weights differ.
if want 3; then
  for name in mimo_ref oss_a050 oss_a100 placebo_a100; do
    src="$WORK/$name"; [ "$name" = mimo_ref ] && src="$HF/mimo-distill-hf"
    [ -f "$WORK/$name.Q6_K.gguf" ] && continue
    MSYS_NO_PATHCONV=1 "$PY" "$CONV" "$(cygpath -w "$src")" --outtype bf16 --outfile "$(cygpath -w "$WORK/$name.bf16.gguf")"
    "$BIN/llama-quantize.exe" "$WORK/$name.bf16.gguf" "$WORK/$name.Q6_K.gguf" Q6_K
    rm -f "$WORK/$name.bf16.gguf"
  done
fi

# 4. Same protocol as results.md §37: GSM8K 100 + MATH-500 50, greedy, 8192 tokens. ~55 min per arm.
if want 4; then
  for name in mimo_ref oss_a100 placebo_a100 oss_a050; do
    [ -f "$OUT/$name.json" ] && continue
    serve "$WORK/$name.Q6_K.gguf"
    "$PY" -c "import sys; sys.path.insert(0,'.'); from engine import bench_runner as b; b.run('$name', $PORT, out_dir='$OUT', log=lambda s: print(s, flush=True))"
    kill $SERVER; sleep 5
  done
fi

if want 5; then "$PY" $SR/compare.py; fi
