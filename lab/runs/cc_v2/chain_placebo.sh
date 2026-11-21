#!/usr/bin/env bash
# Placebo distribution (lab/probes/cc/PLAN.md, "Placebo distribution", prediction row 29): eight same-norm random
# placebos of cc_v2 (seeds 1-8) at scale 1.0 on the screen set (64 MATH-500 items, 2,048 tokens, greedy).
# Base (screen_base.json) and vector (screen_s1.0.json) reused from 45.4. One GPU job at a time; idempotent.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; LOG=$OUT/chain.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
log() { echo "[cc_pl] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m $M "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $OUT/server_placebo.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {  # own server only: the Windows PID listening on 8095
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
bench() {  # name
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
s=json.load(open('$OUT/split.json'))
b.run('$1', 8095, out_dir='$OUT', max_tokens=2048, gsm_idx=[], math_idx=s['screen_math'], log=lambda m: print(m, flush=True))" >> $LOG 2>&1 || log "ARM FAILED $1"; }

# nothing else on the GPU
if netstat -ano | grep ':8095 ' | grep -q LISTENING; then log "port 8095 busy: not started"; exit 1; fi
for f in $OUT/screen_base.json $OUT/screen_s1.0.json $OUT/split.json; do [ -f $f ] || { log "missing $f"; exit 1; }; done

log "check: placebo seed 2026 regenerated from cc_v2.gguf"
"$P" lab/probes/cc/placebo_seeds.py check >> $LOG 2>&1 || { log "CHECK FAILED (generator differs): not run"; exit 1; }
log "make placebos seeds $SEEDS"
"$P" lab/probes/cc/placebo_seeds.py make $SEEDS >> $LOG 2>&1 || { log "MAKE FAILED"; exit 1; }

for sd in $SEEDS; do
  [ -f $OUT/screen_pl$sd.json ] && continue
  PL=hf_cache/cc/cc_v2.placebo$sd.gguf
  [ -f $PL ] || { log "missing $PL"; exit 1; }
  log "placebo seed $sd at scale 1.0 (screen)"
  serve --control-vector-scaled $PL:1.0
  bench screen_pl$sd
  stop
done

log "readings"
"$P" lab/probes/cc/placebo_seeds.py readings $SEEDS >> $LOG 2>&1 || log "READINGS FAILED"
log "PLACEBO DONE"
