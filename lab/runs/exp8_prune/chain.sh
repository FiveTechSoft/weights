#!/usr/bin/env bash
# D4k (lab/probes/exp8_prune/PLAN.md, prediction row 41): alphaXiv's pruned Exp 8 edit in the cc08 harness
# (results 45.12: GSM8K 0-99 + MATH-500 select_indices(2026, 50), thinking on, greedy, 2,048 tokens, 4 slots).
# Arms: full (replica again: run-to-run noise), d4k (4,096 largest entries per layer), d4k_comp (the other 61,440).
# set -eu + ERR trap, chain_guard config, 2-item smoke per served GGUF, sha256 of every served GGUF in the log.
# One GPU job at a time. Resumable (finished arms are skipped).
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/exp8_prune; LOG=$R/chain.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8096
SP=""
MT=2048
declare -A GG
GG[full]=servidos/qwen35-08b__exp8-replica.gguf
GG[d4k]=servidos/qwen35-08b__exp8-d4k.gguf
GG[d4k_comp]=servidos/qwen35-08b__exp8-d4k-complement.gguf
mkdir -p $R/smoke
log() { echo "[exp8_prune] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # model.gguf
  log "serving $1 sha256 $(sha256sum $1 | cut -c1-16)"
  nohup $S -m $1 -ngl 99 -c 16384 -np 4 -fa on --jinja --port $PORT > $R/server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $1"; stop; exit 1; }; }
smoke() {  # 2 GSM8K items at 256 tokens on the served GGUF
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('smoke_$1', $PORT, out_dir='$R/smoke', max_tokens=256, gsm_idx=[0, 1], math_idx=[], log=lambda m: print(m, flush=True))
d=json.load(open('$R/smoke/smoke_$1.json')); assert len(d['items'])==2; print('smoke ok', [x['tokens'] for x in d['items']], 'served', (d.get('served') or {}).get('model_path'))" >> $LOG 2>&1; }
bench() {  # name (cc08 test items: bench_runner defaults)
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('$1', $PORT, out_dir='$R', max_tokens=$MT, parallel=4, log=lambda m: print(m, flush=True))" >> $LOG 2>&1; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done; }

log "start (D4k)"
for a in full d4k d4k_comp; do [ -f ${GG[$a]} ] || { log "missing ${GG[$a]}"; exit 1; }; done
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
"$P" lab/probes/cc/chain_guard.py $R full=$(sha256sum ${GG[full]} | cut -c1-16) d4k=$(sha256sum ${GG[d4k]} | cut -c1-16) \
  d4k_comp=$(sha256sum ${GG[d4k_comp]} | cut -c1-16) budget=$MT commit=$(git rev-parse --short HEAD) >> $LOG 2>&1

for a in full d4k d4k_comp; do
  [ -f $R/test_$a.json ] && { log "test_$a exists, skipped"; continue; }
  log "arm $a"; serve ${GG[$a]}; smoke $a; bench test_$a; stop
done
log "CHAIN DONE"
