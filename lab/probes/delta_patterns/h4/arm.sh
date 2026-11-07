#!/usr/bin/env bash
# One H4 arm (lab/probes/delta_patterns/h4_plan.md): build -> GGUF bf16 -> Q6_K -> llama-server -> GSM8K 100.
# usage: bash lab/probes/delta_patterns/h4/arm.sh A0|A1|A2      (from the repo root; one arm at a time)
set -u
cd "$(dirname "$0")/../../../.."
ARM=$1
PY="$HOME/.venvs/weights-cuda/Scripts/python.exe"
CONV_PY="$HOME/.venvs/llama-convert/Scripts/python.exe"   # transformers 4.57.6 pinned by llama.cpp (findings F27)
LLAMA=/c/Users/Anto/llamacpp
H4=hf_cache/h4
OUT=lab/runs/h4
PORT=8094
mkdir -p "$H4" "$OUT"
export PYTHONUTF8=1 GIT_TERMINAL_PROMPT=0
log() { echo "[h4 $ARM] $* ($(date +%H:%M:%S))"; }
[ -f "$OUT/HOLD" ] && { log "HOLD file present ($OUT/HOLD): not starting"; exit 1; }

if [ "$ARM" = "A0" ]; then
  HF=hf_cache/qwen35-9b-hf
else
  HF=$H4/$ARM-hf
  if [ ! -f "$H4/$ARM-Q6_K.gguf" ] && [ ! -f "$HF/config.json" ]; then
    log "build"
    "$PY" engine/wsplang.py lab/probes/delta_patterns/h4/$ARM.wspl --run --model hf_cache/qwen35-9b-hf \
      > "$OUT/$ARM.build.log" 2>&1 || { log "build failed"; tail -20 "$OUT/$ARM.build.log"; exit 1; }
    cp hf_cache/qwen35-9b-hf/*.txt "$HF/" 2>/dev/null
  fi
fi
if [ ! -f "$H4/$ARM-Q6_K.gguf" ]; then
  log "convert"
  "$CONV_PY" "$LLAMA/convert-src/convert_hf_to_gguf.py" "$HF" --outtype bf16 --outfile "$H4/$ARM-bf16.gguf" \
    > "$OUT/$ARM.convert.log" 2>&1 || { log "convert failed"; tail -20 "$OUT/$ARM.convert.log"; exit 1; }
  PRE=$("$PY" -c "import gguf,sys; r=gguf.GGUFReader(sys.argv[1]); f=r.fields['tokenizer.ggml.pre']; print(bytes(f.parts[f.data[0]]).decode())" "$H4/$ARM-bf16.gguf")
  log "tokenizer.ggml.pre = $PRE"
  [ "$PRE" = "qwen35" ] || { log "wrong pre-tokenizer, stop"; exit 1; }
  log "quantize"
  "$LLAMA/bin/llama-quantize.exe" "$H4/$ARM-bf16.gguf" "$H4/$ARM-Q6_K.gguf" Q6_K > "$OUT/$ARM.quantize.log" 2>&1 \
    || { log "quantize failed"; exit 1; }
  rm -f "$H4/$ARM-bf16.gguf"
  [ "$ARM" != "A0" ] && rm -rf "$HF"
fi
log "serve"
"$LLAMA/bin/llama-server.exe" -m "$H4/$ARM-Q6_K.gguf" -ngl 99 -c 24576 -np 2 -fa on --jinja --port $PORT \
  > "$OUT/$ARM.server.log" 2>&1 &
SERVER=$!
until curl -s "http://127.0.0.1:$PORT/health" | grep -q ok; do
  kill -0 $SERVER 2>/dev/null || { log "server died"; exit 1; }
  sleep 5
done
log "eval"
"$PY" -c "
import sys; sys.path.insert(0, '.')
from engine import bench_runner as b
b.run('h4_$ARM', $PORT, gsm_n=100, math_n=0, parallel=2, out_dir='$OUT', log=lambda s: print(s, flush=True))
"
RC=$?
kill $SERVER
log "done rc=$RC"
