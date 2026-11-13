#!/usr/bin/env bash
# Recompile the corrected conclude vector into down_proj (lab/probes/cc_weights/PLAN.md, "Repeat with the corrected
# vector", 2026-11-12): op -> mechanical check -> dev (merged, placebo merged) -> verdict -> sealed set 2 if PASS.
# Waits for chain_comp.sh (one GPU job at a time); runs only if cc_v2 passed dev.
set -u
cd /c/weights
export PYTHONUTF8=1
export WSP_LORA_CONVERTER=/c/Users/Anto/llamacpp/convert-src/convert_lora_to_gguf.py
export WSP_CONVERT_PYTHON="$HOME/.venvs/llama-convert/Scripts/python.exe"
export CCW_CC=lab/runs/cc_v2 CCW_RUNS=lab/runs/cc_v2/weights CCW_SEALED=lab/runs/sealed_ccw_v2
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; LOG=$OUT/chain.log; W=$OUT/weights; SO=lab/runs/sealed_ccw_v2; H=hf_cache/cc_op_v2
V=hf_cache/cc/cc_v2.gguf; VD=lab/probes/cc_weights/verdict.py; HE=lab/probes/close_probe/he_eval.py
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
log() { echo "[cc_w2] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { local m=$1; shift; nohup $S -m $m "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $OUT/server_w2.log 2>&1 & SP=$!; up || { log "SERVER FAILED $m"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
bench() {  # name out_dir [sealed]
  local sealed="${3:-}"
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b, sealed_set as ss
kw = {}
if '$sealed' == 'sealed':
    s = ss.load('data/sealed_confirm2.json'); kw = dict(gsm_idx=s['gsm8k'], math_idx=s['math500'])
b.run('$1', 8095, out_dir='$2', max_tokens=2048, log=lambda m: print(m, flush=True), **kw)" >> $LOG 2>&1 || log "ARM FAILED $1"; }

until grep -qE "^\[cc_comp\] (COMP DONE|cc_v2 dev .* composition not run)" $LOG 2>/dev/null; do sleep 60; done
DV=$("$P" -c "import json;print(json.load(open('$OUT/verdict.json'))['verdict'])" 2>/dev/null)
[ "$DV" = PASS ] || { log "cc_v2 dev $DV: recompilation not run"; exit 0; }
SC=$("$P" -c "import json;print(json.load(open('$OUT/scale.json'))['scale'])")
[ "$SC" = 1.0 ] || { log "scale $SC != 1.0: the compiled write assumes the vector at x1.0 (PLAN); not run"; exit 0; }
mkdir -p $W
if [ ! -f $H/report.json ]; then
  log "op compile_control_vector from cc_v2 (programs/cc_compile_v2.wspl)"
  "$P" -m engine.wsplang programs/cc_compile_v2.wspl --run >> $LOG 2>&1 || { log "OP FAILED"; exit 1; }
fi
grep -q '"pass": true' $H/report.json || { log "STOP: G0 fails -> negative (PLAN)"; log "W2 DONE"; exit 0; }
[ -f $H/merged-Q6_K.gguf ] || { log "no merged GGUF written"; exit 1; }
log "mechanical check against the first compilation"
"$P" lab/probes/cc_weights/v2_check.py >> $LOG 2>&1 || log "CHECK FAILED (continuing: descriptive)"
for arm in "merged $H/merged-Q6_K.gguf" "placebo_merged $H/placebo_merged-Q6_K.gguf"; do
  set -- $arm; name=$1; shift
  [ -f $W/dev_$name.json ] && continue
  log "dev $name"; serve "$@"; bench dev_$name $W; stop
done
"$P" $VD >> $LOG 2>&1
DW=$("$P" -c "import json;print(json.load(open('$W/verdict.json'))['verdict'])" 2>/dev/null)
log "dev verdict $DW"
[ "$DW" = PASS ] || { log "W2 DONE (sealed not run)"; exit 0; }
log "sealed set 2 (third use): vec_v2, merged, placebo merged; base reused from lab/runs/sealed_ccw"
mkdir -p $SO
"$P" -c "import sys; sys.path.insert(0,'.'); from engine import sealed_set as ss; ss.load('data/sealed_confirm2.json'); print('seal ok')" >> $LOG 2>&1 || { log "SEAL CHECK FAILED"; exit 1; }
[ -f $SO/base.json ] || cp lab/runs/sealed_ccw/base.json $SO/base.json
for arm in "vec $M --control-vector-scaled $V:1.0" "merged $H/merged-Q6_K.gguf" "placebo_merged $H/placebo_merged-Q6_K.gguf"; do
  set -- $arm; name=$1; shift
  [ -f $SO/$name.json ] && continue
  log "sealed $name"; serve "$@"; bench $name $SO sealed; stop
done
"$P" $VD sealed >> $LOG 2>&1
log "sealed verdict $("$P" -c "import json;print(json.load(open('$SO/verdict.json'))['verdict'])" 2>/dev/null)"
log "retention: HumanEval-20 on the merged model (reported, not a criterion)"
[ -f lab/runs/close_probe/fast/he_ccw2_merged.json ] || { serve $H/merged-Q6_K.gguf; "$P" $HE ccw2_merged 8095 --n 20 --cap 2048 --parallel 2 >> $LOG 2>&1 || log "HE FAILED"; stop; }
log "W2 DONE"
