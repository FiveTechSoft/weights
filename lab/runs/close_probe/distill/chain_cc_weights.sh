#!/usr/bin/env bash
# cc compiled into down_proj (lab/probes/cc_weights/PLAN.md, fixed before running): fit -> G0 -> merge -> dev arms.
set -u
cd /c/weights
export PYTHONUTF8=1
export WSP_LORA_CONVERTER=/c/Users/Anto/llamacpp/convert-src/convert_lora_to_gguf.py
export WSP_CONVERT_PYTHON="$HOME/.venvs/llama-convert/Scripts/python.exe"
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
D=lab/runs/close_probe/distill; LOG=$D/chain_cc_weights.log; OUT=lab/runs/cc_weights; H=hf_cache/cc_weights
C=lab/probes/cc_weights/compile.py; A=lab/probes/cc_weights/agree.py
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
log() { echo "[ccw] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { local m=$1; shift; nohup $S -m $m "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $D/ccw.server.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
bench() { "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('$1', 8095, out_dir='$OUT', max_tokens=2048, log=lambda m: print(m, flush=True))" >> $LOG 2>&1 || log "ARM FAILED $1"; }
mkdir -p $OUT
log "waiting for the vector fit"
FITPID=${FITPID:-0}
until [ -f $H/vec/fit_summary.json ]; do kill -0 $FITPID 2>/dev/null || [ -f $H/vec/fit_summary.json ] || { log "FIT DIED"; exit 1; }; sleep 30; done
"$P" $C gate $H/vec >> $LOG 2>&1
grep -q '"G0": true' $H/vec/gate.json || { log "STOP: G0 fails -> negative (PLAN)"; log "CHAIN DONE"; exit 0; }
[ -f $H/vec/lora_r1.gguf ] || { log "lora"; "$P" $C lora $H/vec >> $LOG 2>&1 || log "LORA FAILED"; }
[ -f $H/cc_merged-Q6_K.gguf ] || { log "merge vec"; "$P" $C merge $H/vec $H/cc_merged-Q6_K.gguf >> $LOG 2>&1 || { log "MERGE FAILED"; exit 1; }; }
"$P" $C gate $H/vec >> $LOG 2>&1
[ -f $H/placebo/fit_summary.json ] || { log "fit placebo"; "$P" $C fit hf_cache/cc/cc.placebo2026.gguf $H/placebo >> $OUT/fit_placebo.log 2>&1 || { log "PLACEBO FIT FAILED"; exit 1; }; }
[ -f $H/placebo_merged-Q6_K.gguf ] || { log "merge placebo"; "$P" $C merge $H/placebo $H/placebo_merged-Q6_K.gguf >> $LOG 2>&1 || { log "MERGE FAILED"; exit 1; }; }
"$P" $C gate $H/placebo >> $LOG 2>&1
log "G0 agreement"
[ -f $OUT/agree_vector.json ] || { serve $M --control-vector-scaled hf_cache/cc/cc.gguf:1.0; "$P" $A gen vector 8095 >> $LOG 2>&1; stop; }
[ -f $OUT/agree_merged.json ] || { serve $H/cc_merged-Q6_K.gguf; "$P" $A gen merged 8095 >> $LOG 2>&1; stop; }
"$P" $A cmp vector merged >> $LOG 2>&1
for arm in "merged $H/cc_merged-Q6_K.gguf" "placebo_merged $H/placebo_merged-Q6_K.gguf" "lora $M --lora $H/vec/lora_r1.gguf"; do
  set -- $arm; name=$1; shift
  [ -f $OUT/dev_$name.json ] && continue
  log "dev $name"; serve "$@"; bench dev_$name; stop
done
log "CHAIN DONE"
