#!/usr/bin/env bash
# Merged arm at 8,192 tokens, served as the model (cc/PLAN.md "Token-budget control", post-data note 2026-11-14;
# prediction row 32). The first attempt (chain_cap8192.sh) passed the merged GGUF as a control vector and served the
# base; this chain serves hf_cache/cc_op_v2/merged-Q6_K.gguf with -m and no control vector, same server settings as the
# base and vector arms (-c 40960 -np 2 -fa on --jinja, port 8095), bench_runner defaults (GSM8K dev 100 + MATH-500 50,
# greedy, 2 slots, max_tokens 8192). One GPU job at a time. set -e + ERR trap: any failure writes a FAILED line.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; LOG=$OUT/chain_merged8192.log; D=$OUT/cap8192
HM=hf_cache/cc_op_v2/merged-Q6_K.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8095
SP=""
log() { echo "[merged8192] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() { nohup $S -m "$1" -ngl 99 -c 40960 -np 2 -fa on --jinja --port $PORT > $OUT/server_merged8192.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $1"; stop; exit 1; }; }
bench() {  # name max_tokens [gsm_idx_json]
  local name="$1" mt="$2" gi="${3:-}"
  local kw=""; [ -n "$gi" ] && kw=", gsm_idx=$gi, math_idx=[]"
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('$name', $PORT, out_dir='$D', max_tokens=$mt, log=lambda m: print(m, flush=True)$kw)" >> $LOG 2>&1; }

log "start"
[ -f "$HM" ] || { log "missing $HM"; exit 1; }
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
[ -f $D/dev_merged_served_8192.json ] && { log "merged_served done (marker)"; log "CHAIN DONE"; exit 0; }

log "serve merged as model"
serve $HM
grep -q "no valid control vector" $OUT/server_merged8192.log && { log "server loaded a control vector?!"; stop; exit 1; }
log "smoke: 2 GSM8K items at 256 tokens"
bench smoke_merged_served 256 "[0, 1]"
"$P" -c "
import json; d=json.load(open('$D/smoke_merged_served.json')); assert len(d['items'])==2, d['items']; print('smoke ok', [x['tokens'] for x in d['items']])" >> $LOG 2>&1
log "merged at 8192 (GSM8K 100 + MATH 50)"
bench dev_merged_served_8192 8192
stop
"$P" -c "
import json
d=json.load(open('$D/dev_merged_served_8192.json'))['items']
for bench in ('gsm8k','math500'):
    g=[x for x in d if x['bench']==bench]
    print(bench, {'ok': sum(x['ok'] for x in g), 'ok2': sum(x['ok_independent'] for x in g), 'n': len(g),
          'cuts': sum(x['finish']!='stop' for x in g), 'finished_wrong': sum(x['finish']=='stop' and not x['ok'] for x in g),
          'mean_tokens': round(sum(x['tokens'] for x in g)/max(1,len(g)))})" >> $LOG 2>&1
log "CHAIN DONE"
