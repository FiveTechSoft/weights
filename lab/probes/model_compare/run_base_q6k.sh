#!/usr/bin/env bash
# Base Qwen3.5-9B on the MiMo protocol (results.md §37): bartowski Q6_K (same quantizer as the MiMo GGUF),
# llama-server CUDA, -c 24576 -np 2 -fa on --jinja, greedy, 8192 new tokens, GSM8K 100 + MATH-500 50.
# Waits for the delta pipeline to finish first (one GPU job at a time).
# usage: bash lab/probes/model_compare/run_base_q6k.sh   (from the repo root)
set -u
cd "$(dirname "$0")/../../.."
PY="$HOME/.venvs/weights-cuda/Scripts/python.exe"
DIR=hf_cache/qwen35-9b-gguf
GGUF=$DIR/Qwen_Qwen3.5-9B-Q6_K.gguf
PORT=8093

echo "[base_q6k] waiting for lab/runs/delta_patterns/sig_dsv4_heretic.done"
until [ -f lab/runs/delta_patterns/sig_dsv4_heretic.done ]; do sleep 60; done
sleep 120  # the pipeline deletes its download after writing .done

if [ ! -f "$GGUF" ]; then
  hf download bartowski/Qwen_Qwen3.5-9B-GGUF Qwen_Qwen3.5-9B-Q6_K.gguf --local-dir "$DIR" --max-workers 2 || exit 1
fi
echo "[base_q6k] gguf ready: $(ls -la "$GGUF")"

/c/Users/Anto/llamacpp/bin/llama-server.exe -m "$GGUF" -ngl 99 -c 24576 -np 2 -fa on --jinja \
  --port $PORT > lab/runs/model_compare/qwen35_9b_q6k.server.log 2>&1 &
SERVER=$!
echo "[base_q6k] llama-server pid $SERVER"
until curl -s "http://127.0.0.1:$PORT/health" | grep -q ok; do
  kill -0 $SERVER 2>/dev/null || { echo "[base_q6k] server died"; exit 1; }
  sleep 5
done
echo "[base_q6k] server ready"
"$PY" lab/probes/model_compare/eval.py qwen35_9b_q6k $PORT
RC=$?
kill $SERVER
echo "[base_q6k] done rc=$RC"
