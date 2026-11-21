#!/usr/bin/env bash
# Token-budget control on Phi (lab/probes/cc_phi/PLAN.md "Token-budget control on Phi", prediction row 36): base, vector x0.5
# and merged served at 2,048 tokens on the dev items. set -eu + ERR trap, chain_guard config, 2-item smoke per served
# configuration, control-vector check for the vector arm, sha256 of every served GGUF in the log. One GPU job at a time.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_phi; O=$R/cap2048; H=hf_cache/cc_phi_op; LOG=$R/chain_b2048.log
G=hf_cache/phi4mr-gguf/Phi-4-mini-reasoning-Q6_K.gguf
V=hf_cache/cc_phi/cc_phi.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8097; MT=2048; SC=0.5
SP=""
mkdir -p $O $R/smoke
log() { echo "[cc_phi_b2048] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # model.gguf [--control-vector-scaled file:scale]
  local m=$1; shift
  log "serving $m $* sha256 $(sha256sum $m | cut -c1-16)"
  nohup $S -m $m "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server_b2048.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $m $*"; stop; exit 1; }
  if [ $# -gt 0 ] && grep -q "no valid control vector" $R/server_b2048.log; then log "control vector rejected: $*"; stop; exit 1; fi; }
smoke() {  # 2 GSM8K items at 256 tokens on the served configuration
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('smoke_$1', $PORT, out_dir='$R/smoke', max_tokens=256, gsm_idx=[0, 1], math_idx=[], log=lambda m: print(m, flush=True))
d=json.load(open('$R/smoke/smoke_$1.json')); assert len(d['items'])==2; print('smoke ok', [x['tokens'] for x in d['items']], 'served', (d.get('served') or {}).get('model_path'))" >> $LOG 2>&1; }
bench() {  # name (dev items = bench_runner defaults, the 45.15 / 45.16 items)
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('$1', $PORT, out_dir='$O', max_tokens=$MT, log=lambda m: print(m, flush=True))" >> $LOG 2>&1; }

log "start"
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
[ -f $H/merged-Q6_K.gguf ] || { log "merged GGUF missing"; exit 1; }
"$P" lab/probes/cc/chain_guard.py $O gguf=$(sha256sum $G | cut -c1-16) vector=$(sha256sum $V | cut -c1-16) merged=$(sha256sum $H/merged-Q6_K.gguf | cut -c1-16) budget=$MT scale=$SC commit=$(git rev-parse --short HEAD) >> $LOG 2>&1

[ -f $O/base_2048.json ] || { log "arm base"; serve $G; smoke b2048_base; bench base_2048; stop; }
[ -f $O/vec_2048.json ] || { log "arm vec x$SC"; serve $G --control-vector-scaled $V:$SC; smoke b2048_vec; bench vec_2048; stop; }
[ -f $O/merged_2048.json ] || { log "arm merged"; serve $H/merged-Q6_K.gguf; smoke b2048_merged; bench merged_2048; stop; }

"$P" -c "
import json
for n in ('base_2048', 'vec_2048', 'merged_2048'):
    d = json.load(open('$O/%s.json' % n)); out = {}
    for b in ('gsm8k', 'math500'):
        rs = [r for r in d['items'] if r['bench'] == b]
        out[b] = dict(ok=sum(r['ok'] for r in rs), ok2=sum(r['ok_independent'] for r in rs), cut=sum(r['finish'] == 'length' for r in rs),
                      fin_wrong=sum(1 for r in rs if r['finish'] == 'stop' and not r['ok']), tok=round(sum(r['tokens'] for r in rs) / len(rs)))
    print(n, json.dumps(out), 'served', (d.get('served') or {}).get('model_path'))" >> $LOG 2>&1
log "CHAIN DONE"
