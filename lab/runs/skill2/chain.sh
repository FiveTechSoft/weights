#!/usr/bin/env bash
# Second behaviour: "reason in Spanish" on Qwen3.5-9B (lab/probes/skill2/PLAN.md, prediction row 44).
# Part A: discovery generations (English / Spanish answers) -> rows -> op contrast_vector (+ placebo) -> screen (scale)
#         -> dev (base_en, vec_en, placebo_en; base_es, vec_es secondary) -> verdict C1-C3 -> if PASS: new sealed set
#         (data/sealed_skill2.json, single use) base_en / vec_en / placebo_en -> sealed verdict.
# Part B (only if the sealed set confirms): scaled vector -> calib -> op compile_control_vector -> dev merged /
#         placebo_merged -> verdict C1-C4 (cdev).
# set -eu + ERR trap, chain_guard, 2-item smoke per served configuration, control-vector check, sha256 in the log.
# One GPU job at a time (no_server before anything). Resumable: finished arms are skipped.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
K=lab/probes/skill2/skill2.py
R=lab/runs/skill2; LOG=$R/chain.log
G=hf_cache/h4/A0-Q6_K.gguf; HF=hf_cache/qwen35-9b-hf
V=hf_cache/skill2/skill2.gguf; PL=hf_cache/skill2/skill2.placebo2026.gguf
VS=hf_cache/skill2/skill2_scaled.gguf; PLS=hf_cache/skill2/skill2.placebo2026_scaled.gguf
H=hf_cache/skill2_op
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8095
SP=""
MT=2048
SCALES="0.5 1.0 2.0 4.0"
mkdir -p $R/smoke hf_cache/skill2
log() { echo "[skill2] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # model.gguf [--control-vector-scaled file:scale]
  local m=$1; shift
  log "serving $m $* sha256 $(sha256sum $m | cut -c1-16)"
  nohup $S -m $m "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $m $*"; stop; exit 1; }
  if [ $# -gt 0 ] && grep -q "no valid control vector" $R/server.log; then log "control vector rejected: $*"; stop; exit 1; fi; }
smoke() {  # 2 GSM8K items at 256 tokens on the served configuration
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('smoke_$1', $PORT, out_dir='$R/smoke', max_tokens=256, gsm_idx=[0, 1], math_idx=[], log=lambda m: print(m, flush=True))
d=json.load(open('$R/smoke/smoke_$1.json')); assert len(d['items'])==2; print('smoke ok', [x['tokens'] for x in d['items']], 'served', (d.get('served') or {}).get('model_path'))" >> $LOG 2>&1; }
arm() {  # name set [es]
  "$P" $K arm $1 $PORT $2 ${3:-} >> $LOG 2>&1; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done; }

log "start (skill2: reason in Spanish)"
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
"$P" lab/probes/cc/chain_guard.py $R gguf=$(sha256sum $G | cut -c1-16) budget=$MT scales="$SCALES" commit=$(git rev-parse --short HEAD) >> $LOG 2>&1
"$P" $K split >> $LOG 2>&1

# ---------- Part A ----------
if [ ! -f $R/rows.json ]; then
  log "discovery generations (GSM8K train 64-127, plain and with the Spanish instruction)"
  serve $G; smoke gen; "$P" $K gen $PORT >> $LOG 2>&1; stop
  "$P" $K rows >> $LOG 2>&1 || { log "STOP: fewer than 30 Spanish/English pairs (rows.json)"; log "CHAIN DONE"; exit 0; }
fi
log "rows: $(wc -l < $R/pos.jsonl) pairs"

if [ ! -f $V ]; then
  no_server
  log "op contrast_vector (programs/skill2_vector.wspl)"
  "$P" -m engine.wsplang programs/skill2_vector.wspl --run >> $LOG 2>&1
fi
[ -f $V ] && [ -f $PL ] || { log "vector or placebo missing after the op"; exit 1; }
log "vector sha256 $(sha256sum $V | cut -c1-16); placebo sha256 $(sha256sum $PL | cut -c1-16)"

[ -f $R/screen_base.json ] || { log "screen base"; serve $G; smoke screen_base; arm screen_base screen; stop; }
for sc in $SCALES; do
  [ -f $R/screen_s$sc.json ] && continue
  log "screen scale $sc"; serve $G --control-vector-scaled $V:$sc; smoke screen_s$sc; arm screen_s$sc screen; stop
done
"$P" $K choose >> $LOG 2>&1 || { log "STOP: no scale reaches base + 30 pp Spanish thinking with accuracy >= base - 3 (scale.json)"; log "CHAIN DONE"; exit 0; }
SC=$("$P" -c "import json;print(json.load(open('$R/scale.json'))['scale'])")
log "scale $SC"
"$P" lab/probes/cc/chain_guard.py $R/dev gguf=$(sha256sum $G | cut -c1-16) vector=$(sha256sum $V | cut -c1-16) budget=$MT scale=$SC commit=$(git rev-parse --short HEAD) >> $LOG 2>&1

[ -f $R/dev_base_en.json ]    || { log "dev base_en";    serve $G; smoke dev_base_en; arm dev_base_en dev; stop; }
[ -f $R/dev_vec_en.json ]     || { log "dev vec_en";     serve $G --control-vector-scaled $V:$SC; smoke dev_vec_en; arm dev_vec_en dev; stop; }
[ -f $R/dev_placebo_en.json ] || { log "dev placebo_en"; serve $G --control-vector-scaled $PL:$SC; smoke dev_placebo_en; arm dev_placebo_en dev; stop; }
[ -f $R/dev_base_es.json ]    || { log "dev base_es (secondary: with the instruction)"; serve $G; smoke dev_base_es; arm dev_base_es dev es; stop; }
[ -f $R/dev_vec_es.json ]     || { log "dev vec_es (secondary)"; serve $G --control-vector-scaled $V:$SC; smoke dev_vec_es; arm dev_vec_es dev es; stop; }
if "$P" $K verdict dev >> $LOG 2>&1; then DV=PASS; else DV=FAIL; fi
log "dev verdict $DV ($R/dev_verdict.json)"
[ "$DV" = PASS ] || { log "dev FAIL: no sealed run"; log "CHAIN DONE"; exit 0; }

log "sealed set (data/sealed_skill2.json: made now if absent, single use)"
[ -f data/sealed_skill2.json ] || "$P" $K make_sealed >> $LOG 2>&1
"$P" -c "import sys; sys.path.insert(0,'.'); from engine import sealed_set as ss; s=ss.load('data/sealed_skill2.json'); print('seal ok', len(s['gsm8k']), len(s['math500']))" >> $LOG 2>&1
"$P" lab/probes/cc/chain_guard.py $R/sealed gguf=$(sha256sum $G | cut -c1-16) vector=$(sha256sum $V | cut -c1-16) budget=$MT scale=$SC commit=$(git rev-parse --short HEAD) >> $LOG 2>&1
[ -f $R/sealed_base_en.json ]    || { serve $G; smoke sealed_base_en; arm sealed_base_en sealed; stop; }
[ -f $R/sealed_vec_en.json ]     || { serve $G --control-vector-scaled $V:$SC; smoke sealed_vec_en; arm sealed_vec_en sealed; stop; }
[ -f $R/sealed_placebo_en.json ] || { serve $G --control-vector-scaled $PL:$SC; smoke sealed_placebo_en; arm sealed_placebo_en sealed; stop; }
if "$P" $K verdict sealed >> $LOG 2>&1; then SV=CONFIRMS; else SV="DOES NOT CONFIRM"; fi
log "sealed verdict: $SV ($R/sealed_verdict.json)"
log "PART A DONE"
[ "$SV" = CONFIRMS ] || { log "Part B (compile) not run"; log "CHAIN DONE"; exit 0; }

# ---------- Part B: compile into down_proj ----------
[ -f $VS ]  || { log "scale vector x$SC";  "$P" lab/probes/cc_phi/scale_vector.py $V $VS $SC >> $LOG 2>&1; }
[ -f $PLS ] || { log "scale placebo x$SC"; "$P" lab/probes/cc_phi/scale_vector.py $PL $PLS $SC >> $LOG 2>&1; }
log "vector x$SC sha256 $(sha256sum $VS | cut -c1-16); placebo x$SC sha256 $(sha256sum $PLS | cut -c1-16)"
[ -f $R/calib.jsonl ] || { log "calib rows (finished English traces)"; "$P" $K calib >> $LOG 2>&1; }
if [ ! -f $H/report.json ]; then
  no_server
  log "op compile_control_vector (programs/skill2_compile.wspl)"
  "$P" -m engine.wsplang programs/skill2_compile.wspl --run >> $LOG 2>&1
fi
grep -q '"pass": true' $H/report.json || { log "STOP: G0 fails"; log "CHAIN DONE"; exit 0; }
[ -f $H/merged-Q6_K.gguf ] || { log "no merged GGUF written"; exit 1; }
cp -n $R/dev_base_en.json $R/cdev_base_en.json
[ -f $R/cdev_vec_en.json ]     || { log "dev merged";         serve $H/merged-Q6_K.gguf;         smoke cdev_vec_en;     arm cdev_vec_en dev;     stop; }
[ -f $R/cdev_placebo_en.json ] || { log "dev placebo_merged"; serve $H/placebo_merged-Q6_K.gguf; smoke cdev_placebo_en; arm cdev_placebo_en dev; stop; }
if "$P" $K verdict cdev >> $LOG 2>&1; then CV=PASS; else CV=FAIL; fi
log "compile dev verdict $CV ($R/cdev_verdict.json)"
log "CHAIN DONE"
