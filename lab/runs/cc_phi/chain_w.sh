#!/usr/bin/env bash
# Compile the Phi conclude direction into down_proj (lab/probes/cc_phi/PLAN.md, "Compile the Phi direction", prediction
# row 35): scaled vector + placebo -> op compile_control_vector -> G0 -> dev (merged, placebo merged; C1-C4) -> sealed set 2
# if PASS -> HumanEval-20 (reported). set -eu + ERR trap, chain_guard config, 2-item smoke per served configuration,
# control-vector check for the vector arm, sha256 of every served GGUF in the log. One GPU job at a time.
set -eu
cd /c/weights
export PYTHONUTF8=1
export CCW_CC=lab/runs/cc_phi CCW_RUNS=lab/runs/cc_phi/weights CCW_SEALED=lab/runs/sealed_ccw_phi
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_phi; W=$R/weights; SO=lab/runs/sealed_ccw_phi; H=hf_cache/cc_phi_op; LOG=$R/chain_w.log
G=hf_cache/phi4mr-gguf/Phi-4-mini-reasoning-Q6_K.gguf
V=hf_cache/cc_phi/cc_phi.gguf; PL=hf_cache/cc_phi/cc_phi.placebo2026.gguf
VS=hf_cache/cc_phi/cc_phi_x0.5.gguf; PLS=hf_cache/cc_phi/cc_phi.placebo2026_x0.5.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; HE=lab/probes/close_probe/he_eval.py
PORT=8097; MT=1024; SC=0.5
SP=""
mkdir -p $W $SO $R/smoke
log() { echo "[cc_phi_w] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # model.gguf [--control-vector-scaled file:scale]
  local m=$1; shift
  log "serving $m $* sha256 $(sha256sum $m | cut -c1-16)"
  nohup $S -m $m "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server_w.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $m $*"; stop; exit 1; }
  if [ $# -gt 0 ] && grep -q "no valid control vector" $R/server_w.log; then log "control vector rejected: $*"; stop; exit 1; fi; }
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

log "start"
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
"$P" lab/probes/cc/chain_guard.py $W gguf=$(sha256sum $G | cut -c1-16) vector=$(sha256sum $V | cut -c1-16) budget=$MT scale=$SC commit=$(git rev-parse --short HEAD) >> $LOG 2>&1

# 0. scaled vector and placebo (the op writes the vector at x1.0; Phi's chosen scale is 0.5)
[ -f $VS ] || { log "scale vector x$SC"; "$P" lab/probes/cc_phi/scale_vector.py $V $VS $SC >> $LOG 2>&1; }
[ -f $PLS ] || { log "scale placebo x$SC"; "$P" lab/probes/cc_phi/scale_vector.py $PL $PLS $SC >> $LOG 2>&1; }
log "vector x$SC sha256 $(sha256sum $VS | cut -c1-16); placebo x$SC sha256 $(sha256sum $PLS | cut -c1-16)"

# 1. op (GPU; 3.8B bf16 streams layer by layer)
if [ ! -f $H/report.json ]; then
  log "op compile_control_vector (programs/cc_phi_compile.wspl)"
  "$P" -m engine.wsplang programs/cc_phi_compile.wspl --run >> $LOG 2>&1
fi
"$P" -c "
import json; r=json.load(open('$H/report.json')); print(json.dumps({'G0': r['G0'], 'placebo_G0': r.get('placebo', {}).get('G0'), 'calib': r['calib'], 'merge': {k: v for k, v in r['vector'].get('merge', {}).items() if k != 'layers'}, 'seconds': r['seconds']}))" >> $LOG 2>&1
grep -q '"pass": true' $H/report.json || { log "STOP: G0 fails -> negative (PLAN)"; log "CHAIN DONE"; exit 0; }
[ -f $H/merged-Q6_K.gguf ] || { log "no merged GGUF written"; exit 1; }

# 2. dev: merged and placebo merged as -m, no flags (references dev_base / dev_vec of results 45.15)
for arm in "merged $H/merged-Q6_K.gguf" "placebo_merged $H/placebo_merged-Q6_K.gguf"; do
  set -- $arm; name=$1; m=$2
  [ -f $W/dev_$name.json ] && continue
  log "dev $name"; serve $m; smoke dev_$name; bench dev_$name $W "" ""; stop
done
"$P" lab/probes/cc_weights/verdict.py >> $LOG 2>&1
DW=$("$P" -c "import json;print(json.load(open('$W/verdict.json'))['verdict'])")
log "dev verdict $DW"
[ "$DW" = PASS ] || { log "dev $DW: sealed not run"; log "CHAIN DONE"; exit 0; }

# 3. sealed set 2 (data/sealed_confirm2.json): first use for Phi; base, vector x0.5, merged, placebo merged
log "sealed set 2 (first use for this model)"
"$P" -c "import sys; sys.path.insert(0,'.'); from engine import sealed_set as ss; ss.load('data/sealed_confirm2.json'); print('seal ok')" >> $LOG 2>&1
SS="s=ss.load('data/sealed_confirm2.json')"; SK=", gsm_idx=s['gsm8k'], math_idx=s['math500']"
[ -f $SO/base.json ] || { serve $G; smoke sealed_base; bench base $SO "$SS" "$SK"; stop; }
[ -f $SO/vec.json ] || { serve $G --control-vector-scaled $V:$SC; smoke sealed_vec; bench vec $SO "$SS" "$SK"; stop; }
[ -f $SO/merged.json ] || { serve $H/merged-Q6_K.gguf; smoke sealed_merged; bench merged $SO "$SS" "$SK"; stop; }
[ -f $SO/placebo_merged.json ] || { serve $H/placebo_merged-Q6_K.gguf; smoke sealed_placebo_merged; bench placebo_merged $SO "$SS" "$SK"; stop; }
"$P" lab/probes/cc_weights/verdict.py sealed >> $LOG 2>&1
log "sealed verdict $("$P" -c "import json;print(json.load(open('$SO/verdict.json'))['verdict'])")"

# 4. retention (reported, not a criterion): HumanEval first 20 on base and merged
for arm in "phi_base $G" "phi_merged $H/merged-Q6_K.gguf"; do
  set -- $arm; name=$1; m=$2
  [ -f lab/runs/close_probe/fast/he_$name.json ] && continue
  log "HumanEval-20 $name"; serve $m; "$P" $HE $name $PORT --n 20 --cap 2048 --parallel 2 >> $LOG 2>&1 || log "HE FAILED $name"; stop
done
log "CHAIN DONE"
