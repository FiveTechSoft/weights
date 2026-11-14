#!/usr/bin/env bash
# "conclude" direction on Qwen3.5-9B without a donor (lab/probes/cc/PLAN.md, fixed before running). After chain_donors4.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
D=lab/runs/close_probe/distill; LOG=$D/chain_cc.log; OUT=lab/runs/cc; V=hf_cache/cc/cc.gguf; PL=hf_cache/cc/cc.placebo2026.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
log() { echo "[cc] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m $M "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $D/cc.server.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {  # own server only: the Windows PID listening on 8095 (kill $SP alone left it running before)
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
bench() {  # name lora-free; extra args: python kwargs
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
$2
b.run('$1', 8095, out_dir='$OUT', max_tokens=2048, log=lambda m: print(m, flush=True)$3)" >> $LOG 2>&1 || log "ARM FAILED $1"; }
log "waiting for chain_donors4"
until grep -q "CHAIN DONE" $D/chain_donors4.log 2>/dev/null; do sleep 60; done
mkdir -p $OUT hf_cache/cc
if [ ! -f $OUT/rows.done ]; then
  log "traces"; serve
  "$P" lab/probes/cc/cc.py traces 8095 >> $LOG 2>&1 || { log "TRACES FAILED"; stop; exit 1; }
  stop
  "$P" lab/probes/cc/cc.py rows >> $LOG 2>&1 || { log "ROWS FAILED"; exit 1; }
  touch $OUT/rows.done
fi
grep -q '"enough": true' $LOG || { log "STOP: fewer than 30 finished traces (PLAN step 1)"; exit 0; }
if [ ! -f $V ]; then
  log "op contrast_vector"
  "$P" -m engine.wsplang programs/cc_vector.wspl --run >> $LOG 2>&1 || { log "OP FAILED"; exit 1; }
fi
SCR="s=json.load(open('$OUT/split.json'))"
if [ ! -f $OUT/screen_base.json ]; then log "screen base"; serve; bench screen_base "$SCR" ", gsm_idx=[], math_idx=s['screen_math']"; stop; fi
for sc in 0.25 0.5 1.0 2.0; do
  [ -f $OUT/screen_s$sc.json ] && continue
  log "screen scale $sc"; serve --control-vector-scaled $V:$sc
  bench screen_s$sc "$SCR" ", gsm_idx=[], math_idx=s['screen_math']"; stop
done
"$P" lab/probes/cc/cc.py choose >> $LOG 2>&1
SC=$("$P" -c "import json;print(json.load(open('$OUT/scale.json'))['scale'])")
[ "$SC" = "None" ] && { log "STOP: no scale beats base on the screen (PLAN step 4) -> FAIL"; log "CHAIN DONE"; exit 0; }
log "dev at scale $SC"
[ -f $OUT/dev_base.json ] || { serve; bench dev_base "" ""; stop; }
[ -f $OUT/dev_vec.json ] || { serve --control-vector-scaled $V:$SC; bench dev_vec "" ""; stop; }
[ -f $OUT/dev_placebo.json ] || { serve --control-vector-scaled $PL:$SC; bench dev_placebo "" ""; stop; }
"$P" lab/probes/cc/cc.py verdict >> $LOG 2>&1
log "CHAIN DONE"
