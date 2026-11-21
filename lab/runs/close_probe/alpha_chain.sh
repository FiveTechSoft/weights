#!/usr/bin/env bash
# Task 2: base + alpha*(MiMo - base), alpha in 0.5 0.25. Per alpha: build HF (CPU) -> bf16 GGUF -> Q6_K -> delete HF.
# Then, after the GSM8K forced-close job ends, serve each and run MATH-500 n=50 naturally (no forcing), 4096 tokens.
set -u
cd /c/weights
PY="$HOME/.venvs/weights-cuda/Scripts/python.exe"
CONV_PY="$HOME/.venvs/llama-convert/Scripts/python.exe"
LLAMA=/c/Users/Anto/llamacpp
Q=hf_cache/own_q6k
OUT=lab/runs/close_probe
export PYTHONUTF8=1 GIT_TERMINAL_PROMPT=0
log() { echo "[alpha] $* ($(date +%H:%M:%S))"; }
for A in 0.5 0.25; do
  N=mimo_a${A/./}
  G=$Q/$N-Q6_K.gguf
  [ -f "$G" ] && continue
  HF=hf_cache/close_$N
  log "build $N"
  "$PY" lab/probes/close_probe_run/scale_delta.py hf_cache/qwen35-9b-hf hf_cache/mimo-distill-hf $A $HF > $OUT/$N.build.log 2>&1 || { log "build failed"; exit 1; }
  log "convert $N"
  "$CONV_PY" "$LLAMA/convert-src/convert_hf_to_gguf.py" $HF --outtype bf16 --outfile $Q/$N-bf16.gguf > $OUT/$N.convert.log 2>&1 || { log "convert failed"; exit 1; }
  PRE=$("$PY" -c "import gguf,sys; r=gguf.GGUFReader(sys.argv[1]); f=r.fields['tokenizer.ggml.pre']; print(bytes(f.parts[f.data[0]]).decode())" $Q/$N-bf16.gguf)
  [ "$PRE" = "qwen35" ] || { log "wrong pre $PRE"; exit 2; }
  "$LLAMA/bin/llama-quantize.exe" $Q/$N-bf16.gguf $G Q6_K > $OUT/$N.quantize.log 2>&1 || { log "quantize failed"; exit 1; }
  rm -f $Q/$N-bf16.gguf
  rm -rf $HF
  log "built $G"
done
log "waiting for GSM8K job"
until grep -qE '^\{"n"|Traceback' $OUT/base_gsm.log; do sleep 20; done
taskkill //IM llama-server.exe //F > /dev/null 2>&1
sleep 5
for A in 0.5 0.25; do
  N=mimo_a${A/./}
  "$LLAMA/bin/llama-server.exe" -m $Q/$N-Q6_K.gguf -ngl 99 -c 24576 -np 2 -fa on --jinja --port 8095 > $OUT/$N.server.log 2>&1 &
  S=$!
  until curl -s http://127.0.0.1:8095/health | grep -q ok; do kill -0 $S 2>/dev/null || { log "server died"; exit 1; }; sleep 5; done
  log "eval $N"
  "$PY" -c "
import sys; sys.path.insert(0, '.')
from engine import bench_runner as b
b.run('$N', 8095, gsm_n=0, math_n=50, parallel=2, max_tokens=4096, out_dir='$OUT', log=lambda s: print(s, flush=True))
" > $OUT/$N.eval.log 2>&1
  log "$N rc=$?"
  kill $S; sleep 5
done
log "ALL DONE"
