#!/usr/bin/env bash
# 9B base on dev-2048-full (IFEval 541 + TriviaQA 200 on top of dev-2048; lab/probes/preserve/IFEVAL_TRIVIA.md):
# about 5.5 GPU hours, run once on the base to open the knowledge / instruction-following columns. set -eu + ERR trap,
# sha256 of the served GGUF in the log, one llama-server at a time. Resumable through bench_runner's partial files.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/bench; LOG=$R/chain_full9b.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
M=hf_cache/h4/A0-Q6_K.gguf
PORT=8095
SP=""
log() { echo "[full9b] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do log "GPU job alive, waiting"; sleep 120; done; }

log "start (9B base on dev-2048-full)"
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
[ -f $R/qwen9b_base_full.json ] && { log "base_full exists, skipped"; log "CHAIN DONE"; exit 0; }
log "serving $M sha256 $(sha256sum $M | cut -c1-16)"
nohup $S -m $M -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server_full9b.log 2>&1 & SP=$!
up || { log "SERVER FAILED"; stop; exit 1; }
"$P" -m engine.wsplang programs/bench_full9b_base.wspl --run >> $LOG 2>&1
stop
log "CHAIN DONE"
