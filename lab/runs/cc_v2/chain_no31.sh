#!/usr/bin/env bash
# Descriptive arm (lab/probes/cc/PLAN.md, "Extra arm"): old cc.gguf without direction.31, dev only, after chain.sh.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; LOG=$OUT/chain.log; V=hf_cache/cc/cc_no31.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
log() { echo "[cc_no31] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m $M "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $OUT/server_no31.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
until grep -q "^\[cc_v2\] CHAIN DONE" $LOG 2>/dev/null; do sleep 60; done
SC=$("$P" -c "import json;s=json.load(open('$OUT/scale.json'))['scale'];print(1.0 if s is None else s)")
[ -f $OUT/dev_no31.json ] && { log "dev_no31 exists"; exit 0; }
log "dev no31 at scale $SC"
serve --control-vector-scaled $V:$SC
"$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('dev_no31', 8095, out_dir='$OUT', max_tokens=2048, log=lambda m: print(m, flush=True))" >> $LOG 2>&1 || log "ARM FAILED dev_no31"
stop
log "NO31 DONE"
