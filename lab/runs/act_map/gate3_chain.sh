#!/usr/bin/env bash
# act_map gate 3 (PLAN.md addendum "Gate 3"): port the 9B "happy" direction into the 0.8B. One GPU job at a time.
set -u
cd /c/weights
export PYTHONUTF8=1
export HAPPY_RUNS=/c/weights/lab/runs/act_map/happy08
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"; H=lab/probes/happy/happy.py; PT=lab/probes/act_map/port.py
O=lab/runs/act_map; LOG=$O/gate3_chain.log; C=hf_cache/act_map/cvec08
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M08=hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf
JUDGE=hf_cache/own_q6k/mimo-Q6_K.gguf; PORT=8096
SCALES="0.125 0.25 0.5 1.0 2.0 4.0"
mkdir -p $HAPPY_RUNS
log() { echo "[gate3] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
serve() { nohup $S -m "$@" -ngl 99 -c 16384 -np 4 -fa on --jinja --port $PORT > $O/gate3.server.log 2>&1 & SP=$!; up || { log "SERVER FAILED $*"; kill $SP; exit 1; }; }
stop() { kill $SP; sleep 5; }

log "vectors + step 0"
"$P" $PT vectors >> $LOG 2>&1 || { log "VECTORS FAILED"; exit 1; }

log "dev generations"
serve $M08
"$P" $H gen $PORT dev dev_base >> $LOG 2>&1; stop
DEVNAMES="dev_base"
for arm in own port embport; do
  for sc in $SCALES; do
    serve $M08 --control-vector-scaled $C/happy_$arm.gguf:$sc
    "$P" $H gen $PORT dev dev_${arm}_s$sc >> $LOG 2>&1; stop
    DEVNAMES="$DEVNAMES dev_${arm}_s$sc"
  done
done

log "judge dev"
serve $JUDGE
"$P" $H judge $PORT $DEVNAMES >> $LOG 2>&1; stop

# relative path: the Windows python does not read MSYS paths inside a -c string (first run: empty count, the guard passed)
BOK=$("$P" -c "import json;print(sum(bool(r['coherent']) for r in json.load(open('lab/runs/act_map/happy08/dev_base.json',encoding='utf-8'))['rows']))")
log "base dev coherent $BOK/10"
if ! [ "${BOK:-0}" -ge 9 ] 2>/dev/null; then log "judge not valid on the 0.8B: stop"; "$P" $PT verdict >> $LOG 2>&1; log "CHAIN DONE"; exit 0; fi

declare -A SC
for arm in own port embport; do SC[$arm]=$("$P" $PT choose $arm 2>>$LOG | tail -1); log "scale $arm = ${SC[$arm]}"; done
echo "{\"own\": \"${SC[own]}\", \"port\": \"${SC[port]}\", \"embport\": \"${SC[embport]}\", \"placebo\": \"${SC[port]}\"}" > $HAPPY_RUNS/scales.json

log "test generations"
serve $M08
"$P" $H gen $PORT test test_base >> $LOG 2>&1
"$P" lab/probes/close_probe_run/fast_eval.py gsm08_happy_base $PORT --bench gsm8k --guard >> $LOG 2>&1; stop
TESTNAMES="test_base"
for arm in own port embport placebo; do
  if [ $arm = placebo ]; then sc=${SC[port]}; f=$C/happy_port.placebo2026.gguf; else sc=${SC[$arm]}; f=$C/happy_$arm.gguf; fi
  if [ "$sc" = none ]; then log "$arm: no coherent dev scale, not run"; continue; fi
  serve $M08 --control-vector-scaled $f:$sc
  "$P" $H gen $PORT test test_$arm >> $LOG 2>&1
  "$P" lab/probes/close_probe_run/fast_eval.py gsm08_happy_$arm $PORT --bench gsm8k --guard >> $LOG 2>&1; stop
  TESTNAMES="$TESTNAMES test_$arm"
done

log "judge test"
serve $JUDGE
"$P" $H judge $PORT $TESTNAMES >> $LOG 2>&1; stop
"$P" $PT verdict >> $LOG 2>&1
log "CHAIN DONE"
