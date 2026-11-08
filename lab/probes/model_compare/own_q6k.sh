#!/usr/bin/env bash
# Own Q6_K (no imatrix, same recipe as H4 A0) of a HF bf16 folder, then the §39 protocol:
# llama-server CUDA -c 24576 -np 2 -fa on --jinja, greedy, 8192 new tokens, GSM8K first <gsm_n> + MATH-500 <math_n>.
# usage: bash lab/probes/model_compare/own_q6k.sh <name> <hf_dir> <gsm_n> <math_n>   (repo root; one at a time)
# The GGUF goes to hf_cache/own_q6k/<name>-Q6_K.gguf; results to lab/runs/model_compare/<name>.json.
set -u
cd /c/weights
NAME=$1; HF=$2; GSM=$3; MATH=$4
PY="$HOME/.venvs/weights-cuda/Scripts/python.exe"
CONV_PY="$HOME/.venvs/llama-convert/Scripts/python.exe"   # transformers 4.57.6 pinned by llama.cpp (F27)
LLAMA=/c/Users/Anto/llamacpp
Q=hf_cache/own_q6k
OUT=lab/runs/model_compare
PORT=8095
mkdir -p "$Q" "$OUT"
export PYTHONUTF8=1 GIT_TERMINAL_PROMPT=0
log() { echo "[own_q6k $NAME] $* ($(date +%H:%M:%S))"; }

GGUF=$Q/$NAME-Q6_K.gguf
[ "$NAME" = "base" ] && GGUF=hf_cache/h4/A0-Q6_K.gguf
if [ ! -f "$GGUF" ]; then
  log "convert"
  "$CONV_PY" "$LLAMA/convert-src/convert_hf_to_gguf.py" "$HF" --outtype bf16 --outfile "$Q/$NAME-bf16.gguf" \
    > "$OUT/$NAME.convert.log" 2>&1 || { log "convert failed"; tail -20 "$OUT/$NAME.convert.log"; exit 1; }
  PRE=$("$PY" -c "import gguf,sys; r=gguf.GGUFReader(sys.argv[1]); f=r.fields['tokenizer.ggml.pre']; print(bytes(f.parts[f.data[0]]).decode())" "$Q/$NAME-bf16.gguf")
  log "tokenizer.ggml.pre = $PRE"
  [ "$PRE" = "qwen35" ] || { log "wrong pre-tokenizer, stop"; rm -f "$Q/$NAME-bf16.gguf"; exit 2; }
  log "quantize"
  "$LLAMA/bin/llama-quantize.exe" "$Q/$NAME-bf16.gguf" "$GGUF" Q6_K > "$OUT/$NAME.quantize.log" 2>&1 \
    || { log "quantize failed"; exit 1; }
  rm -f "$Q/$NAME-bf16.gguf"
fi
log "serve $(ls -la "$GGUF")"
"$LLAMA/bin/llama-server.exe" -m "$GGUF" -ngl 99 -c 24576 -np 2 -fa on --jinja --port $PORT \
  > "$OUT/$NAME.server.log" 2>&1 &
SERVER=$!
until curl -s "http://127.0.0.1:$PORT/health" | grep -q ok; do
  kill -0 $SERVER 2>/dev/null || { log "server died"; exit 1; }
  sleep 5
done
log "eval"
"$PY" -c "
import sys; sys.path.insert(0, '.')
from engine import bench_runner as b
b.run('own_$NAME', $PORT, gsm_n=$GSM, math_n=$MATH, parallel=2, out_dir='$OUT', log=lambda s: print(s, flush=True))
"
RC=$?
kill $SERVER
log "done rc=$RC"
exit $RC
