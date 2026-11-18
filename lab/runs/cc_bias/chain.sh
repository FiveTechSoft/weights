#!/usr/bin/env bash
# Constant `</think>` logit bias as the control for the conclude vector (lab/probes/cc_bias/PLAN.md, prediction row 43,
# roadmap gap 18). Four dev arms on Qwen3.5-9B Q6_K (base GGUF, no control vector; bias in the request) on the cc_v2
# dev items (bench_runner defaults, 2,048 tokens, -np 2), then the pre-registered pick (tokens only) and the comparison
# against lab/runs/cc_v2/dev_vec.json and dev_base.json. set -eu + ERR trap, chain_guard config, 2-item smoke per
# served configuration, sha256 of the served GGUF in the log. One GPU job at a time. Resumable.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_bias; LOG=$R/chain.log
G=hf_cache/h4/A0-Q6_K.gguf
HF=hf_cache/qwen35-9b-hf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8095
SP=""
MT=2048
BIASES="2 4 6 8"
mkdir -p $R/smoke
log() { echo "[cc_bias] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # base model, no control vector
  log "serving $G sha256 $(sha256sum $G | cut -c1-16)"
  nohup $S -m $G -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $G"; stop; exit 1; }; }
smoke() {  # 2 GSM8K items at 256 tokens with the bias of this arm
  "$P" lab/probes/cc_bias/run_arm.py smoke_$1 $PORT $R/smoke $TID $2 256 "[0, 1]" "[]" >> $LOG 2>&1
  "$P" -c "
import json; d=json.load(open('$R/smoke/smoke_$1.json')); assert len(d['items'])==2
print('smoke ok', [x['tokens'] for x in d['items']], 'extra', d['extra'], 'served', (d.get('served') or {}).get('model_path'))" >> $LOG 2>&1; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done; }

log "start (constant </think> bias control)"
for f in lab/runs/cc_v2/dev_base.json lab/runs/cc_v2/dev_vec.json; do [ -f $f ] || { log "missing $f"; exit 1; }; done
TID=$("$P" -c "import sys; sys.path.insert(0,'.'); from lab.probes.cc_bias.bias import close_think_id; print(close_think_id('$HF'))")
log "</think> token id $TID (from $HF/tokenizer.json); budget $MT; biases $BIASES"
[ "$TID" = 248069 ] || { log "unexpected token id $TID (plan says 248069)"; exit 1; }
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
"$P" lab/probes/cc/chain_guard.py $R gguf=$(sha256sum $G | cut -c1-16) token=$TID budget=$MT biases="$BIASES" commit=$(git rev-parse --short HEAD) >> $LOG 2>&1

for bb in $BIASES; do
  [ -f $R/dev_bias_$bb.json ] && { log "dev_bias_$bb exists, skipped"; continue; }
  log "arm bias +$bb"; serve; smoke bias_$bb $bb
  "$P" lab/probes/cc_bias/run_arm.py dev_bias_$bb $PORT $R $TID $bb $MT >> $LOG 2>&1
  stop
done
"$P" lab/probes/cc_bias/bias.py pick >> $LOG 2>&1
"$P" lab/probes/cc_bias/bias.py compare >> $LOG 2>&1
log "verdict: $("$P" -c "import json;d=json.load(open('$R/compare.json'));print('picked bias', d['picked'], d['verdict'])")"
log "CHAIN DONE"
