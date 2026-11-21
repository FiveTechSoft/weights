#!/usr/bin/env bash
# Stage-0 premise probe for lab/probes/cc_phi/PLAN.md (Phi-4-mini-reasoning): P1 reasoning parsed, P2 base over-thinks,
# P3 control vectors applied to phi3, (P4 offsets checked on CPU, 2026-11-15: ok). Waits for cc08b_chain.sh. Nothing decides
# here; the plan reads the numbers. set -e + ERR trap.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_phi; LOG=$R/probe.log; PR=lab/probes/cc_phi/probe.py
G=hf_cache/phi4mr-gguf/Phi-4-mini-reasoning-Q6_K.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8097
SP=""
mkdir -p $R
log() { echo "[phi-probe] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # [extra args]
  nohup $S -m $G "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/probe.server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $*"; stop; exit 1; }; }

log "start"
until grep -qE "G3CC-B DONE|FAILED" lab/runs/act_map/cc08b_chain.log 2>/dev/null; do sleep 120; done
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 60; done
log "gguf sha256 $(sha256sum $G | cut -c1-16)"

serve
log "P1/P2: 5 GSM8K + 5 MATH dev items at 2048"
"$P" $PR items $PORT 2048 >> $LOG 2>&1
CUT=$("$P" -c "
import json; d=json.load(open('$R/probe_2048.json')); print(sum(1 for x in d['items'] if x['finish']!='stop'))")
log "cut at 2048: $CUT of 10"
if [ "$CUT" -le 2 ]; then log "P2 at 2048 fails: repeating at 1024"; "$P" $PR items $PORT 1024 >> $LOG 2>&1; fi
log "P3 base outputs at 512 (3 GSM + 3 MATH via gsm_n/math_n = 5, kept 5+5)"
"$P" $PR items $PORT 512 _base >> $LOG 2>&1
stop

"$P" $PR randvec $R/randvec.gguf >> $LOG 2>&1
log "P3: same prompts with a random control vector, scale 8"
serve --control-vector-scaled $R/randvec.gguf:8
if grep -q "no valid control vector" $R/probe.server.log; then log "control vector rejected"; stop; exit 1; fi
"$P" $PR items $PORT 512 _vec >> $LOG 2>&1
stop
"$P" $PR compare $R/probe_512_base.json $R/probe_512_vec.json >> $LOG 2>&1
log "PROBE DONE"
