#!/usr/bin/env bash
# Name test of "conclude" outside maths (lab/probes/cc/PLAN.md, "Name test"): HumanEval-20 and Horn logic 120,
# base / cc_v2 x SC / placebo_v2 x SC. Waits for chain_no31.sh; runs only if cc_v2 passed dev.
set -u
cd /c/weights
export PYTHONUTF8=1
export CC_RUNS=lab/runs/cc_v2
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; LOG=$OUT/chain.log; NT=lab/probes/cc/name_test.py; HE=lab/probes/close_probe/he_eval.py
V=hf_cache/cc/cc_v2.gguf; PL=hf_cache/cc/cc_v2.placebo2026.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
log() { echo "[cc_name] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m $M "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $OUT/server_name.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
until grep -q "^\[cc_no31\] NO31 DONE" $LOG 2>/dev/null; do sleep 60; done
DV=$("$P" -c "import json;print(json.load(open('$OUT/verdict.json'))['verdict'])" 2>/dev/null)
[ "$DV" = PASS ] || { log "cc_v2 dev $DV: name test not run"; exit 0; }
SC=$("$P" -c "import json;print(json.load(open('$OUT/scale.json'))['scale'])")
log "name test at scale $SC"
mkdir -p $OUT/name
for arm in base vec placebo; do
  case $arm in base) A="";; vec) A="--control-vector-scaled $V:$SC";; placebo) A="--control-vector-scaled $PL:$SC";; esac
  if [ ! -f lab/runs/close_probe/fast/he_name_$arm.json ] || [ ! -f $OUT/name/logic_$arm.json ]; then
    log "arm $arm"; serve $A
    [ -f lab/runs/close_probe/fast/he_name_$arm.json ] || "$P" $HE name_$arm 8095 --n 20 --cap 2048 --parallel 2 >> $LOG 2>&1 || log "HE FAILED $arm"
    [ -f $OUT/name/logic_$arm.json ] || "$P" $NT logic 8095 $arm >> $LOG 2>&1 || log "LOGIC FAILED $arm"
    stop
  fi
done
"$P" $NT verdict >> $LOG 2>&1
log "NAME DONE"
