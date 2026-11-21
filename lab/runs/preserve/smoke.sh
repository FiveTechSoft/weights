#!/usr/bin/env bash
# Preservation battery smoke (lab/probes/preserve/PLAN.md): the 9B base only. 2 HumanEval tasks at 256 tokens as the
# smoke, then the full battery (HumanEval 164 + 40 compliance prompts, 2,048 tokens). set -eu + ERR trap, sha256 of the
# served GGUF in the log, one llama-server at a time. Not launched by the agent; the coordinator decides.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/preserve; LOG=$R/smoke.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
M=hf_cache/h4/A0-Q6_K.gguf
PORT=8095
SP=""
mkdir -p $R/smoke
log() { echo "[preserve] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {
  log "serving $1 sha256 $(sha256sum $1 | cut -c1-16)"
  nohup $S -m $1 -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $1"; stop; exit 1; }; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done; }

log "start (preservation battery smoke, 9B base)"
[ -f $M ] || { log "missing $M"; exit 1; }
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
"$P" lab/probes/cc/chain_guard.py $R gguf=$(sha256sum $M | cut -c1-16) budget=2048 commit=$(git rev-parse --short HEAD) >> $LOG 2>&1
serve $M
"$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import preserve_bench as pb
d = pb.run('smoke_base', $PORT, out_dir='$R/smoke', max_tokens=256, parallel=2, he_n=2, comp_n=0, log=lambda m: print(m, flush=True))
assert len(d['items']) == 2; print('smoke ok', [x['tokens'] for x in d['items']], 'served', (d.get('served') or {}).get('model_path'))" >> $LOG 2>&1
[ -f $R/base.json ] || "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import preserve_bench as pb
pb.run('base', $PORT, out_dir='$R', max_tokens=2048, parallel=2, log=lambda m: print(m, flush=True))" >> $LOG 2>&1
stop
log "SMOKE DONE"
