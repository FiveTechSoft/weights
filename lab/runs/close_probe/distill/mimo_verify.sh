#!/usr/bin/env bash
# mimo_verify — verifies the MiMo 2.6 → base Qwen transplants after the fact (driver request 2026-11-09).
# A) distill3_mimo_r16 (behavior transplant, verified 05:53: 9/16 MATH, 21/28 GSM, reward 0.533/0.720)
# B) mimo_r64 (WSPLang 0.7 op delta_to_lora r64 over the FULL MiMo HF delta: expected not to fit)
# Waits until no llama-server runs on the 12GB (one at a time, house rule). Resumable by marker files.
set -u
export PYTHONUTF8=1
cd /c/weights
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
LOG=lab/runs/close_probe/distill/mimo_verify.log
L=hf_cache/close_lora
F=lab/probes/close_probe_run/fast_eval.py
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
B=hf_cache/h4/A0-Q6_K.gguf
PORT=8097
log() { echo "[mimo-verify] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 60); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
serve() { nohup $S -m $B --lora "$1" --lora-init-without-apply -ngl 99 -c 32768 -np 6 -fa on --jinja --port $PORT >> $D/../ev_verify.server.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; kill $SP; exit 1; }; }
down() { kill $SP 2>/dev/null; sleep 5; }
battery() { log "battery $2"; $P $F "$1" $PORT --guard --parallel 6 >> $LOG 2>&1 || { log "MATH FAILED $1"; exit 1; }; $P $F "$2" $PORT --bench gsm8k --guard --parallel 6 >> $LOG 2>&1 || { log "GSM FAILED"; exit 1; }; }
gpu_free() { pgrep -x llama-server >/dev/null 2>&1; }
D=lab/runs/close_probe/distill

log "start"
while gpu_free; do sleep 90; done

# A) distill3 re-verify
if [ ! -f lab/runs/close_probe/fast/math_distill3_reverify.json ]; then
  sleep 20
  serve $L/distill3_mimo_r16.gguf
  battery math_distill3_reverify gsm_distill3_reverify
  down
fi
log "distill3 reverified"

# B) WSPLang-native r64 delta transplant re-verify
if [ ! -f lab/runs/close_probe/fast/math_mimo_r64_reverify.json ]; then
  serve $L/mimo_r64.gguf
  battery math_mimo_r64_reverify gsm_mimo_r64_reverify
  down
fi
log "mimo_r64 reverified"

$P lab/probes/close_probe/reward.py --all >> $LOG 2>&1
log "CHAIN DONE"
