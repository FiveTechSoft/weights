#!/usr/bin/env bash
# Sealed set 2 (lab/probes/cc_weights/PLAN.md, dev passed C1-C4): base / control vector / merged / placebo merged,
# data/sealed_confirm2.json, 2048 tokens, once per arm. Starts after the Turbo forensic (PID $WAITPID) ends (RAM).
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
D=lab/runs/close_probe/distill; LOG=$D/chain_sealed_ccw.log; OUT=lab/runs/sealed_ccw; H=hf_cache/cc_weights
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
log() { echo "[sealed_ccw] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { local m=$1; shift; nohup $S -m $m "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $D/sealed_ccw.server.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
mkdir -p $OUT
WAITPID=${WAITPID:-0}
log "waiting for PID $WAITPID"
while kill -0 $WAITPID 2>/dev/null; do sleep 60; done
"$P" -c "import sys; sys.path.insert(0,'.'); from engine import sealed_set as ss; ss.load('data/sealed_confirm2.json'); print('seal ok')" >> $LOG 2>&1 || { log "SEAL CHECK FAILED"; exit 1; }
for arm in "base $M" "vec $M --control-vector-scaled hf_cache/cc/cc.gguf:1.0" "merged $H/cc_merged-Q6_K.gguf" "placebo_merged $H/placebo_merged-Q6_K.gguf"; do
  set -- $arm; name=$1; shift
  [ -f $OUT/$name.json ] && continue
  log "arm $name"; serve "$@"
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b, sealed_set as ss
s=ss.load('data/sealed_confirm2.json')
b.run('$name', 8095, out_dir='$OUT', max_tokens=2048, gsm_idx=s['gsm8k'], math_idx=s['math500'], log=lambda m: print(m, flush=True))" >> $LOG 2>&1 || log "ARM FAILED $name"
  stop
done
log "CHAIN DONE"
