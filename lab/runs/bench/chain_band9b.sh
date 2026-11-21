#!/usr/bin/env bash
# Noise band of the Qwen3.5-9B Q6_K base on the standard dev-2048 suite (lab/probes/wsplang_bench/PLAN.md): the same
# GGUF served twice with the cc_v2 flags, bench_run through WSP-Lang programs, then bench_compare a vs b and the table.
# set -eu + ERR trap, sha256 of the served GGUF in the log, one llama-server at a time. Resumable.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/bench; LOG=$R/chain_band9b.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
M=hf_cache/h4/A0-Q6_K.gguf
PORT=8095
SP=""
log() { echo "[band9b] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() { log "serving $M sha256 $(sha256sum $M | cut -c1-16)"
  nohup $S -m $M -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server_band9b.log 2>&1 & SP=$!
  up || { log "SERVER FAILED"; stop; exit 1; }; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done; }

log "start (9B base twice on dev-2048)"
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
for a in a b; do
  [ -f $R/qwen9b_base_$a.json ] && { log "base_$a exists, skipped"; continue; }
  log "run base_$a"; serve
  "$P" -m engine.wsplang programs/bench_band9b_$a.wspl --run >> $LOG 2>&1
  stop
done
"$P" -m engine.wsplang programs/bench_band9b_compare.wspl --run >> $LOG 2>&1
log "band: $("$P" -c "import json;d=json.load(open('$R/cmp_qwen9b_band.json'));print(json.dumps({k:d[k] for k in d if k in ('relation','verdict','benchmarks','band')})[:400])")"
log "CHAIN DONE"
