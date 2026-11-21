#!/usr/bin/env bash
# "Conclude" on a third family: DeepSeek-R1-Distill-Llama-8B (lab/probes/cc_llama/PLAN.md, prediction rows 37-38).
# Runs only after probe.sh wrote lab/runs/cc_llama/budget.txt. Part A: traces -> rows -> op (vector + placebo) -> screen
# (base, 4 scales) -> choose -> dev (base, vec, placebo) -> verdict -> sealed if PASS. Part B (only if A's dev PASS):
# scaled vector + placebo -> calib -> op compile_control_vector -> G0 -> dev (merged, placebo merged; C1-C4) -> sealed
# set 2 if PASS -> HumanEval-20 (reported). set -eu + ERR trap, chain_guard config, 2-item smoke per served
# configuration, control-vector check, sha256 of every served GGUF in the log. One GPU job at a time. Resumable.
set -eu
cd /c/weights
export PYTHONUTF8=1
export CC_RUNS=lab/runs/cc_llama CC_HF=hf_cache/r1llama8b-hf
export CCW_CC=lab/runs/cc_llama CCW_RUNS=lab/runs/cc_llama/weights CCW_SEALED=lab/runs/sealed_ccw_llama
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_llama; W=$R/weights; SOUT=lab/runs/sealed_cc_llama; SO=lab/runs/sealed_ccw_llama; LOG=$R/chain.log
G=hf_cache/r1llama8b-gguf/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf
V=hf_cache/cc_llama/cc_llama.gguf; PL=hf_cache/cc_llama/cc_llama.placebo2026.gguf
VS=hf_cache/cc_llama/cc_llama_scaled.gguf; PLS=hf_cache/cc_llama/cc_llama.placebo2026_scaled.gguf
H=hf_cache/cc_llama_op
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; HE=lab/probes/close_probe/he_eval.py
PORT=8097
SP=""
mkdir -p $R $R/smoke hf_cache/cc_llama
log() { echo "[cc_llama] $* ($(date +%H:%M:%S))" >> $LOG; }
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
bench() {  # name out_dir setup kwargs
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b, sealed_set as ss
$3
b.run('$1', $PORT, out_dir='$2', max_tokens=$MT, log=lambda m: print(m, flush=True)$4)" >> $LOG 2>&1; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done; }

log "start"
[ -f $R/budget.txt ] || { log "no budget.txt: the stage-0 probe has not passed P2; not started"; exit 1; }
MT=$(cat $R/budget.txt); log "budget $MT"
no_server
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
"$P" lab/probes/cc/chain_guard.py $R gguf=$(sha256sum $G | cut -c1-16) budget=$MT commit=$(git rev-parse --short HEAD) >> $LOG 2>&1

# ---------- Part A: the direction as a control vector (row 37) ----------
[ -f $R/split.json ] || { log "split"; "$P" lab/probes/cc/cc.py split >> $LOG 2>&1; }

if [ ! -f $R/pos.jsonl ]; then
  serve $G; smoke traces
  log "traces (128 discovery problems, 4096 cap)"; "$P" lab/probes/cc/cc.py traces $PORT >> $LOG 2>&1
  log "rows (cc.py rule: the template writes <think>)"; "$P" lab/probes/cc/cc.py rows >> $LOG 2>&1
  N=$(wc -l < $R/pos.jsonl)
  if [ "$N" -lt 30 ]; then
    log "only $N rows: traces extra (GSM8K train 64-191)"; "$P" lab/probes/cc/cc.py traces $PORT extra >> $LOG 2>&1
    "$P" lab/probes/cc/cc.py rows >> $LOG 2>&1; N=$(wc -l < $R/pos.jsonl)
    [ "$N" -ge 30 ] || { log "STOP: $N rows < 30 after extra"; stop; log "CHAIN DONE"; exit 0; }
  fi
  stop
fi
log "rows: $(wc -l < $R/pos.jsonl)"

if [ ! -f $V ]; then
  no_server
  log "op contrast_vector (programs/cc_llama_vector.wspl)"; "$P" -m engine.wsplang programs/cc_llama_vector.wspl --run >> $LOG 2>&1
  "$P" -c "
import json; r=json.load(open('hf_cache/cc_llama/cc_llama.json')); n=r['increment_norms']; ks=sorted(n, key=int)
print(json.dumps({'pairs': r['pairs'], 'layers': r['layers'], 'last_row_norm': r['last_layer_norm'], 'inc_last': n[ks[-1]], 'inc_prev': n[ks[-2]], 'inc_median': sorted(n.values())[len(n)//2]}))" >> $LOG 2>&1
fi
log "vector sha256 $(sha256sum $V | cut -c1-16); placebo sha256 $(sha256sum $PL | cut -c1-16)"

SCR="s=json.load(open('$R/split.json'))"
if [ ! -f $R/screen_base.json ]; then log "screen base"; serve $G; smoke screen_base; bench screen_base $R "$SCR" ", gsm_idx=[], math_idx=s['screen_math']"; stop; fi
for sc in 0.25 0.5 1.0 2.0; do
  [ -f $R/screen_s$sc.json ] && continue
  log "screen scale $sc"; serve $G --control-vector-scaled $V:$sc; smoke screen_s$sc
  bench screen_s$sc $R "$SCR" ", gsm_idx=[], math_idx=s['screen_math']"; stop
done
"$P" lab/probes/cc/cc.py choose >> $LOG 2>&1
SC=$("$P" -c "import json;print(json.load(open('$R/scale.json'))['scale'])")
if [ "$SC" = "None" ] || [ -z "$SC" ]; then log "STOP: no scale beats base on the screen -> FAIL"; log "CHAIN DONE"; exit 0; fi

log "dev at scale $SC"
[ -f $R/dev_base.json ] || { serve $G; smoke dev_base; bench dev_base $R "" ""; stop; }
[ -f $R/dev_vec.json ] || { serve $G --control-vector-scaled $V:$SC; smoke dev_vec; bench dev_vec $R "" ""; stop; }
[ -f $R/dev_placebo.json ] || { serve $G --control-vector-scaled $PL:$SC; smoke dev_placebo; bench dev_placebo $R "" ""; stop; }
"$P" lab/probes/cc/cc.py verdict >> $LOG 2>&1
DV=$("$P" -c "import json;print(json.load(open('$R/verdict.json'))['verdict'])")
log "dev verdict $DV"
if [ "$DV" != PASS ]; then log "dev FAIL: no sealed run, no Part B"; log "CHAIN DONE"; exit 0; fi

log "sealed confirmation (data/sealed_confirm.json: first use for this model)"
mkdir -p $SOUT
"$P" -m engine.sealed_set check >> $LOG 2>&1
SS="s=ss.load()"; SK=", gsm_idx=s['gsm8k'], math_idx=s['math500']"
[ -f $SOUT/base.json ] || { serve $G; smoke sealed_base; bench base $SOUT "$SS" "$SK"; stop; }
[ -f $SOUT/vec.json ] || { serve $G --control-vector-scaled $V:$SC; smoke sealed_vec; bench vec $SOUT "$SS" "$SK"; stop; }
[ -f $SOUT/placebo.json ] || { serve $G --control-vector-scaled $PL:$SC; smoke sealed_placebo; bench placebo $SOUT "$SS" "$SK"; stop; }
SEALED_CC_RUNS=$SOUT "$P" lab/probes/sealed/cc_verdict.py >> $LOG 2>&1
log "PART A DONE (sealed verdict in $SOUT)"

# ---------- Part B: the direction compiled into down_proj (row 38) ----------
mkdir -p $W $SO
"$P" lab/probes/cc/chain_guard.py $W gguf=$(sha256sum $G | cut -c1-16) vector=$(sha256sum $V | cut -c1-16) budget=$MT scale=$SC commit=$(git rev-parse --short HEAD) >> $LOG 2>&1
[ -f $VS ] || { log "scale vector x$SC"; "$P" lab/probes/cc_phi/scale_vector.py $V $VS $SC >> $LOG 2>&1; }
[ -f $PLS ] || { log "scale placebo x$SC"; "$P" lab/probes/cc_phi/scale_vector.py $PL $PLS $SC >> $LOG 2>&1; }
log "vector x$SC sha256 $(sha256sum $VS | cut -c1-16); placebo x$SC sha256 $(sha256sum $PLS | cut -c1-16)"
[ -f $R/calib.jsonl ] || { log "calib rows (no prefix)"; "$P" lab/probes/cc_llama/rows.py calib >> $LOG 2>&1; }

if [ ! -f $H/report.json ]; then
  no_server
  log "op compile_control_vector (programs/cc_llama_compile.wspl)"
  "$P" -m engine.wsplang programs/cc_llama_compile.wspl --run >> $LOG 2>&1
fi
"$P" -c "
import json; r=json.load(open('$H/report.json')); print(json.dumps({'G0': r['G0'], 'placebo_G0': r.get('placebo', {}).get('G0'), 'calib': r['calib'], 'merge': {k: v for k, v in r['vector'].get('merge', {}).items() if k != 'layers'}, 'seconds': r['seconds']}))" >> $LOG 2>&1
grep -q '"pass": true' $H/report.json || { log "STOP: G0 fails -> negative (PLAN)"; log "CHAIN DONE"; exit 0; }
[ -f $H/merged-Q6_K.gguf ] || { log "no merged GGUF written"; exit 1; }

for arm in "merged $H/merged-Q6_K.gguf" "placebo_merged $H/placebo_merged-Q6_K.gguf"; do
  set -- $arm; name=$1; m=$2
  [ -f $W/dev_$name.json ] && continue
  log "dev $name"; serve $m; smoke dev_$name; bench dev_$name $W "" ""; stop
done
"$P" lab/probes/cc_weights/verdict.py >> $LOG 2>&1
DW=$("$P" -c "import json;print(json.load(open('$W/verdict.json'))['verdict'])")
log "dev verdict (compile) $DW"
[ "$DW" = PASS ] || { log "dev $DW: sealed 2 not run"; log "CHAIN DONE"; exit 0; }

log "sealed set 2 (first use for this model)"
"$P" -c "import sys; sys.path.insert(0,'.'); from engine import sealed_set as ss; ss.load('data/sealed_confirm2.json'); print('seal ok')" >> $LOG 2>&1
SS2="s=ss.load('data/sealed_confirm2.json')"
[ -f $SO/base.json ] || { serve $G; smoke s2_base; bench base $SO "$SS2" "$SK"; stop; }
[ -f $SO/vec.json ] || { serve $G --control-vector-scaled $V:$SC; smoke s2_vec; bench vec $SO "$SS2" "$SK"; stop; }
[ -f $SO/merged.json ] || { serve $H/merged-Q6_K.gguf; smoke s2_merged; bench merged $SO "$SS2" "$SK"; stop; }
[ -f $SO/placebo_merged.json ] || { serve $H/placebo_merged-Q6_K.gguf; smoke s2_placebo_merged; bench placebo_merged $SO "$SS2" "$SK"; stop; }
"$P" lab/probes/cc_weights/verdict.py sealed >> $LOG 2>&1
log "sealed verdict (compile) $("$P" -c "import json;print(json.load(open('$SO/verdict.json'))['verdict'])")"

for arm in "llama_base $G" "llama_merged $H/merged-Q6_K.gguf"; do
  set -- $arm; name=$1; m=$2
  [ -f lab/runs/close_probe/fast/he_$name.json ] && continue
  log "HumanEval-20 $name"; serve $m; "$P" $HE $name $PORT --n 20 --cap 2048 --parallel 2 >> $LOG 2>&1 || log "HE FAILED $name"; stop
done
log "CHAIN DONE"
