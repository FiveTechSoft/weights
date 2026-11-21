#!/usr/bin/env bash
# Third family, Part C: dose (lab/probes/cc_llama/PLAN.md Part C, prediction row 40). Dev arms of the Llama "conclude"
# vector at scales 0.25 and 0.35 (same 150 dev items and reference dev_base.json as Parts A-B), then the pre-registered
# gate (lab/probes/cc_llama/dose_gate.py). Only if the gate passes: sealed set 3 (data/sealed_confirm3.json, single use,
# 200 GSM8K + 100 MATH-500): base, vector at the chosen scale, same-norm placebo at that scale, cc_verdict.py.
# set -eu + ERR trap, chain_guard config, 2-item smoke per served configuration, control-vector check, sha256 of every
# served GGUF in the log. One GPU job at a time. Resumable.
set -eu
cd /c/weights
export PYTHONUTF8=1
export CC_RUNS=lab/runs/cc_llama CC_HF=hf_cache/r1llama8b-hf
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_llama; S3=lab/runs/sealed3_cc_llama; LOG=$R/chain_c.log
G=hf_cache/r1llama8b-gguf/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf
V=hf_cache/cc_llama/cc_llama.gguf; PL=hf_cache/cc_llama/cc_llama.placebo2026.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8097
SP=""
SCALES="0.25 0.35"
mkdir -p $R/smoke $S3
log() { echo "[cc_llama C] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # model.gguf [--control-vector-scaled file:scale]
  local m=$1; shift
  log "serving $m $* sha256 $(sha256sum $m | cut -c1-16)"
  nohup $S -m $m "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server_c.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $m $*"; stop; exit 1; }
  if [ $# -gt 0 ] && grep -q "no valid control vector" $R/server_c.log; then log "control vector rejected: $*"; stop; exit 1; fi; }
smoke() {  # 2 GSM8K items at 256 tokens on the served configuration
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('smoke_$1', $PORT, out_dir='$R/smoke', max_tokens=256, gsm_idx=[0, 1], math_idx=[], log=lambda m: print(m, flush=True))
d=json.load(open('$R/smoke/smoke_$1.json')); assert len(d['items'])==2; print('smoke ok', [x['tokens'] for x in d['items']], 'served', (d.get('served') or {}).get('model_path'))" >> $LOG 2>&1; }
bench() {  # name out_dir setup kwargs
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b, sealed_set as ss
$3
b.run('$1', $PORT, out_dir='$2', max_tokens=$MT, log=lambda m: print(m, flush=True)$4)" >> $LOG 2>&1; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done; }

log "start (Part C: dose)"
[ -f $R/budget.txt ] || { log "no budget.txt"; exit 1; }
[ -f $R/dev_base.json ] || { log "no dev_base.json (Part A reference)"; exit 1; }
MT=$(cat $R/budget.txt); log "budget $MT; scales $SCALES"
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
mkdir -p $R/dose
"$P" lab/probes/cc/chain_guard.py $R/dose gguf=$(sha256sum $G | cut -c1-16) vector=$(sha256sum $V | cut -c1-16) budget=$MT scales="$SCALES" commit=$(git rev-parse --short HEAD) >> $LOG 2>&1
log "vector sha256 $(sha256sum $V | cut -c1-16); placebo sha256 $(sha256sum $PL | cut -c1-16)"

for sc in $SCALES; do
  [ -f $R/dev_vec_s$sc.json ] && continue
  log "dev vector scale $sc"; serve $G --control-vector-scaled $V:$sc; smoke dev_vec_s$sc
  bench dev_vec_s$sc $R "" ""; stop
done
"$P" lab/probes/cc_llama/dose_gate.py $SCALES >> $LOG 2>&1
SC=$("$P" -c "import json;print(json.load(open('$R/dose.json'))['chosen'])")
OK=$("$P" -c "import json;print(json.load(open('$R/dose.json'))['pass'])")
log "gate: chosen scale $SC, pass $OK"
if [ "$OK" != True ]; then log "gate FAIL: sealed set 3 not used"; log "CHAIN DONE"; exit 0; fi

log "sealed set 3 (data/sealed_confirm3.json: single use) at scale $SC"
"$P" -c "import sys; sys.path.insert(0,'.'); from engine import sealed_set as ss; s=ss.load('data/sealed_confirm3.json'); print('seal ok', len(s['gsm8k']), len(s['math500']))" >> $LOG 2>&1
"$P" lab/probes/cc/chain_guard.py $S3 gguf=$(sha256sum $G | cut -c1-16) vector=$(sha256sum $V | cut -c1-16) budget=$MT scale=$SC commit=$(git rev-parse --short HEAD) >> $LOG 2>&1
SS3="s=ss.load('data/sealed_confirm3.json')"; SK=", gsm_idx=s['gsm8k'], math_idx=s['math500']"
[ -f $S3/base.json ] || { serve $G; smoke s3_base; bench base $S3 "$SS3" "$SK"; stop; }
[ -f $S3/vec.json ] || { serve $G --control-vector-scaled $V:$SC; smoke s3_vec; bench vec $S3 "$SS3" "$SK"; stop; }
[ -f $S3/placebo.json ] || { serve $G --control-vector-scaled $PL:$SC; smoke s3_placebo; bench placebo $S3 "$SS3" "$SK"; stop; }
SEALED_CC_RUNS=$S3 "$P" lab/probes/sealed/cc_verdict.py >> $LOG 2>&1
log "sealed 3 verdict: $(tail -n 1 $S3/verdict.txt)"
log "CHAIN DONE"
