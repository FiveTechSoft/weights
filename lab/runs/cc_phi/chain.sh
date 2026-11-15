#!/usr/bin/env bash
# "Conclude" on a second family: Phi-4-mini-reasoning (lab/probes/cc_phi/PLAN.md, prediction row 34). Runs only after the
# stage-0 probe passed and lab/runs/cc_phi/budget.txt holds the budget (2048 or 1024). Steps: traces -> rows -> op (vector +
# placebo) -> screen (base, 4 scales) -> choose -> dev (base, vec, placebo) -> verdict -> sealed if PASS.
# set -e + ERR trap, 2-item smoke per served configuration, control-vector check, GGUF sha256 logged. One GPU job at a time.
set -eu
cd /c/weights
export PYTHONUTF8=1
export CC_RUNS=lab/runs/cc_phi
export CC_HF=hf_cache/phi4mr-hf
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_phi; SOUT=lab/runs/sealed_cc_phi; LOG=$R/chain.log
G=hf_cache/phi4mr-gguf/Phi-4-mini-reasoning-Q6_K.gguf
V=hf_cache/cc_phi/cc_phi.gguf; PL=hf_cache/cc_phi/cc_phi.placebo2026.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8097
SP=""
mkdir -p $R hf_cache/cc_phi
log() { echo "[cc_phi] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # [--control-vector-scaled file:scale]
  nohup $S -m $G "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $*"; stop; exit 1; }
  if [ $# -gt 0 ] && grep -q "no valid control vector" $R/server.log; then log "control vector rejected: $*"; stop; exit 1; fi; }
smoke() {  # 2 GSM8K items at 256 tokens on the served configuration
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('smoke_$1', $PORT, out_dir='$R/smoke', max_tokens=256, gsm_idx=[0, 1], math_idx=[], log=lambda m: print(m, flush=True))
d=json.load(open('$R/smoke/smoke_$1.json')); assert len(d['items'])==2; print('smoke ok', [x['tokens'] for x in d['items']])" >> $LOG 2>&1; }
bench() {  # name out_dir setup kwargs
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b, sealed_set as ss
$3
b.run('$1', $PORT, out_dir='$2', max_tokens=$MT, log=lambda m: print(m, flush=True)$4)" >> $LOG 2>&1; }

log "start"
[ -f $R/budget.txt ] || { log "no budget.txt: the stage-0 probe has not been read; not started"; exit 1; }
MT=$(cat $R/budget.txt); log "budget $MT"
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
log "gguf sha256 $(sha256sum $G | cut -c1-16)"
mkdir -p $R/smoke

# 0. split (same seed as the Qwen study)
[ -f $R/split.json ] || { log "split"; "$P" lab/probes/cc/cc.py split >> $LOG 2>&1; }

# 1. traces (base, 4,096 cap), 2. rows
if [ ! -f $R/pos.jsonl ]; then
  serve; smoke traces
  log "traces (128 discovery problems, 4096 cap)"; "$P" lab/probes/cc/cc.py traces $PORT >> $LOG 2>&1
  log "rows"; "$P" lab/probes/cc_phi/rows.py >> $LOG 2>&1
  N=$(wc -l < $R/pos.jsonl)
  if [ "$N" -lt 30 ]; then
    log "only $N rows: traces extra (GSM8K train 64-191)"; "$P" lab/probes/cc/cc.py traces $PORT extra >> $LOG 2>&1
    "$P" lab/probes/cc_phi/rows.py >> $LOG 2>&1; N=$(wc -l < $R/pos.jsonl)
    [ "$N" -ge 30 ] || { log "STOP: $N rows < 30 after extra"; stop; log "CHAIN DONE"; exit 0; }
  fi
  stop
fi
log "rows: $(wc -l < $R/pos.jsonl)"

# 3. vector + placebo (op on the GPU; 3.8B bf16 fits)
if [ ! -f $V ]; then
  until ! tasklist 2>/dev/null | grep -qi "llama-server"; do sleep 30; done
  log "op contrast_vector"; "$P" -m engine.wsplang programs/cc_phi_vector.wspl --run >> $LOG 2>&1
  "$P" -c "
import json; r=json.load(open('hf_cache/cc_phi/cc_phi.json')); n=r['increment_norms']; ks=sorted(n, key=int)
print(json.dumps({'pairs': r['pairs'], 'layers': r['layers'], 'last_row_norm': r['last_layer_norm'], 'inc_last': n[ks[-1]], 'inc_prev': n[ks[-2]], 'inc_median': sorted(n.values())[len(n)//2]}))" >> $LOG 2>&1
fi

# 4. screen (MATH screen items)
SCR="s=json.load(open('$R/split.json'))"
if [ ! -f $R/screen_base.json ]; then log "screen base"; serve; smoke screen_base; bench screen_base $R "$SCR" ", gsm_idx=[], math_idx=s['screen_math']"; stop; fi
for sc in 0.25 0.5 1.0 2.0; do
  [ -f $R/screen_s$sc.json ] && continue
  log "screen scale $sc"; serve --control-vector-scaled $V:$sc; smoke screen_s$sc
  bench screen_s$sc $R "$SCR" ", gsm_idx=[], math_idx=s['screen_math']"; stop
done
"$P" lab/probes/cc/cc.py choose >> $LOG 2>&1
SC=$("$P" -c "import json;print(json.load(open('$R/scale.json'))['scale'])")
if [ "$SC" = "None" ] || [ -z "$SC" ]; then log "STOP: no scale beats base on the screen -> FAIL"; log "CHAIN DONE"; exit 0; fi

# 5. dev
log "dev at scale $SC"
[ -f $R/dev_base.json ] || { serve; smoke dev_base; bench dev_base $R "" ""; stop; }
[ -f $R/dev_vec.json ] || { serve --control-vector-scaled $V:$SC; smoke dev_vec; bench dev_vec $R "" ""; stop; }
[ -f $R/dev_placebo.json ] || { serve --control-vector-scaled $PL:$SC; smoke dev_placebo; bench dev_placebo $R "" ""; stop; }
"$P" lab/probes/cc/cc.py verdict >> $LOG 2>&1
DV=$("$P" -c "import json;print(json.load(open('$R/verdict.json'))['verdict'])")
log "dev verdict $DV"
if [ "$DV" != PASS ]; then log "dev FAIL: no sealed run"; log "CHAIN DONE"; exit 0; fi

# 6. sealed (data/sealed_confirm.json: first use for this model)
log "sealed confirmation"
mkdir -p $SOUT
"$P" -m engine.sealed_set check >> $LOG 2>&1
SS="s=ss.load()"; SK=", gsm_idx=s['gsm8k'], math_idx=s['math500']"
[ -f $SOUT/base.json ] || { serve; smoke sealed_base; bench base $SOUT "$SS" "$SK"; stop; }
[ -f $SOUT/vec.json ] || { serve --control-vector-scaled $V:$SC; smoke sealed_vec; bench vec $SOUT "$SS" "$SK"; stop; }
[ -f $SOUT/placebo.json ] || { serve --control-vector-scaled $PL:$SC; smoke sealed_placebo; bench placebo $SOUT "$SS" "$SK"; stop; }
SEALED_CC_RUNS=$SOUT "$P" lab/probes/sealed/cc_verdict.py >> $LOG 2>&1
log "CHAIN DONE"
