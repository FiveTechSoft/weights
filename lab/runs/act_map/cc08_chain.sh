#!/usr/bin/env bash
# Gate 3b (lab/probes/act_map/PLAN.md, addendum "Gate 3b", prediction row 30): the 9B's "conclude" direction written
# into the 0.8B through the gate 1 maps. Vectors (own, port, embport, placebo), dev scale choice, test arms, verdict.
# Waits for the placebo-seeds chain (one GPU job at a time). Idempotent: finished arms are skipped.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"; PC=lab/probes/act_map/port_cc.py
O=lab/runs/act_map; OUT=$O/cc08; LOG=$O/cc08_chain.log; C=hf_cache/act_map/cvec08
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
M08=hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf
EXP8=servidos/qwen35-08b__exp8-replica.gguf; EXP8P=servidos/qwen35-08b__exp8-placebo.gguf
PORT=8096
SCALES="0.25 0.5 1.0 2.0 4.0"
mkdir -p $OUT
log() { echo "[g3cc] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m "$@" -ngl 99 -c 16384 -np 4 -fa on --jinja --port $PORT > $O/cc08.server.log 2>&1 & SP=$!; up || { log "SERVER FAILED $*"; stop; exit 1; }; }
stop() {  # own server only: the Windows PID listening on $PORT
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
bench() {  # name items   (items: dev | test)
  local name="$1" items="${2:-test}"
  if [ "$items" = dev ]; then IDX="d=json.load(open('$OUT/dev_items.json')); kw=dict(gsm_idx=d['gsm8k'], math_idx=d['math500'])"; else IDX="kw={}"; fi
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
$IDX
b.run('$name', $PORT, out_dir='$OUT', max_tokens=2048, parallel=4, log=lambda m: print(m, flush=True), **kw)" >> $LOG 2>&1 || log "ARM FAILED $name"; }

until grep -qE "^\[cc_pl\] PLACEBO DONE" lab/runs/cc_v2/chain.log 2>/dev/null; do sleep 120; done
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
if netstat -ano | grep ':8095 ' | grep -q LISTENING; then log "port 8095 busy (another chain): not started"; exit 1; fi

log "dev items (never used anywhere)"
[ -f $OUT/dev_items.json ] || "$P" $PC items >> $LOG 2>&1 || { log "ITEMS FAILED"; exit 1; }
log "edit check (Exp 8 replica vs base)"
"$P" $PC editcheck >> $LOG 2>&1; EDITOK=$?
[ $EDITOK = 0 ] || log "edit check failed: exp8 arms not run, no Exp 8 reference (reported)"
log "vectors + step 0 (9B and 0.8B forwards over the cc rows)"
[ -f $OUT/step0.json ] || "$P" $PC vectors >> $LOG 2>&1 || { log "VECTORS FAILED"; exit 1; }

log "dev base"
if [ ! -f $OUT/dev_base.json ]; then serve $M08; bench dev_base dev; stop; fi
for arm in own port embport; do
  for sc in $SCALES; do
    [ -f $OUT/dev_${arm}_s$sc.json ] && continue
    log "dev $arm scale $sc"
    serve $M08 --control-vector-scaled $C/cc_$arm.gguf:$sc
    bench dev_${arm}_s$sc dev; stop
  done
done
declare -A SC
for arm in own port embport; do SC[$arm]=$("$P" $PC choose $arm 2>>$LOG | tail -1); log "scale $arm = ${SC[$arm]}"; done
echo "{\"own\": \"${SC[own]}\", \"port\": \"${SC[port]}\", \"embport\": \"${SC[embport]}\", \"placebo\": \"${SC[port]}\"}" > $OUT/scales.json

log "test base"
if [ ! -f $OUT/test_base.json ]; then serve $M08; bench test_base; stop; fi
for arm in own port embport placebo; do
  [ -f $OUT/test_$arm.json ] && continue
  if [ $arm = placebo ]; then sc=${SC[port]}; f=$C/cc_port.placebo2026.gguf; else sc=${SC[$arm]}; f=$C/cc_$arm.gguf; fi
  if [ "$sc" = none ] || [ -z "$sc" ]; then log "$arm: no dev scale beats the base, not run"; continue; fi
  log "test $arm scale $sc"
  serve $M08 --control-vector-scaled $f:$sc
  bench test_$arm; stop
done
if [ $EDITOK = 0 ]; then
  if [ ! -f $OUT/test_exp8.json ]; then log "test exp8 (reference edit)"; serve $EXP8; bench test_exp8; stop; fi
  if [ ! -f $OUT/test_exp8_placebo.json ]; then log "test exp8_placebo"; serve $EXP8P; bench test_exp8_placebo; stop; fi
fi

log "verdict"
"$P" $PC verdict >> $LOG 2>&1 || log "VERDICT FAILED"
log "G3CC DONE"
