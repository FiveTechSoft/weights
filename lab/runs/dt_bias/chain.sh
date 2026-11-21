#!/usr/bin/env bash
# Time-step bias +-0.5 arms (lab/probes/dt_bias/PLAN.md, prediction row 49): builds the three GGUFs with wsp.py edit on
# the CPU if missing (add-scalar / add-scalar-signed on blk.*.ssm_dt.bias), serves each once with the cc_v2 flags and
# runs its bench_run program (dev-2048 suite, identity + provenance), then the comparisons and the table. Needs the
# base-twice runs lab/runs/bench/qwen9b_base_{a,b}.json (queue 2). set -eu + ERR trap, sha256 of every served GGUF in
# the log, one GPU job at a time (waits for llama-server and llama-perplexity). Resumable.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/dt_bias; LOG=$R/chain.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
BASE=hf_cache/h4/A0-Q6_K.gguf
PORT=8095
SP=""
declare -A GG PR
GG[plus]=servidos/qwen35-9b__dt_plus05.gguf;        PR[plus]=ediciones/dt_plus05.json
GG[minus]=servidos/qwen35-9b__dt_minus05.gguf;      PR[minus]=ediciones/dt_minus05.json
GG[pm]=servidos/qwen35-9b__dt_pm05_seed2026.gguf;   PR[pm]=ediciones/dt_pm05_seed2026.json
mkdir -p $R
log() { echo "[dt_bias] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() { log "serving $1 sha256 $(sha256sum $1 | cut -c1-16)"
  nohup $S -m $1 -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $1"; stop; exit 1; }; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do log "GPU job alive, waiting"; sleep 120; done; }

log "start (time-step bias +-0.5 arms)"
for f in lab/runs/bench/qwen9b_base_a.json lab/runs/bench/qwen9b_base_b.json; do [ -f $f ] || { log "missing $f (band run first)"; exit 1; }; done
for a in plus minus pm; do
  [ -f ${GG[$a]} ] || { log "building ${GG[$a]} from $BASE with ${PR[$a]} (CPU)"; python wsp.py edit $BASE ${PR[$a]} >> $LOG 2>&1; }
  log "${GG[$a]} sha256 $(sha256sum ${GG[$a]} | cut -c1-16)"
done
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
for a in plus minus pm; do
  case $a in plus) outj=$R/dt_plus05.json;; minus) outj=$R/dt_minus05.json;; pm) outj=$R/dt_pm05.json;; esac
  [ -f $outj ] && { log "$a exists, skipped"; continue; }
  log "arm $a"; serve ${GG[$a]}
  "$P" -m engine.wsplang programs/dt_bias_$a.wspl --run >> $LOG 2>&1
  stop
done
"$P" -m engine.wsplang programs/dt_bias_compare.wspl --run >> $LOG 2>&1
log "CHAIN DONE"
