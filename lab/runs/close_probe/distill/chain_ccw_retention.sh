#!/usr/bin/env bash
# Retention of the compiled 'conclude' edit (reported, not a criterion; cc_weights PLAN "reported"): perplexity on
# lab/data/corpus.txt (Alice, Gutenberg; tracked) and HumanEval first 20, for base / control vector / merged / placebo merged.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
D=lab/runs/close_probe/distill; LOG=$D/chain_ccw_retention.log; OUT=lab/runs/cc_weights/retention; H=hf_cache/cc_weights
B=/c/Users/Anto/llamacpp/bin; M=hf_cache/h4/A0-Q6_K.gguf; V=hf_cache/cc/cc.gguf
log() { echo "[ret] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { local m=$1; shift; nohup $B/llama-server.exe -m $m "$@" -ngl 99 -c 16384 -np 4 -fa on --jinja --port 8095 > $D/ret.server.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
log "waiting for chain_cc08"
until grep -q "CHAIN DONE" $D/chain_cc08.log 2>/dev/null; do sleep 60; done
mkdir -p $OUT
for arm in "base $M" "vec $M --control-vector-scaled $V:1.0" "merged $H/cc_merged-Q6_K.gguf" "placebo_merged $H/placebo_merged-Q6_K.gguf"; do
  set -- $arm; name=$1; shift
  if [ ! -f $OUT/ppl_$name.txt ]; then
    log "ppl $name"
    $B/llama-perplexity.exe -m "$@" -f lab/data/corpus.txt -c 512 --chunks 60 -ngl 99 > $OUT/ppl_$name.txt 2>&1 || log "PPL FAILED $name"
    grep -o "Final estimate: PPL = [0-9.]* +/- [0-9.]*" $OUT/ppl_$name.txt >> $LOG
  fi
  if [ ! -f lab/runs/close_probe/fast/he_ccw_$name.json ]; then
    log "humaneval $name"; serve "$@"
    "$P" lab/probes/close_probe/he_eval.py ccw_$name 8095 --n 20 --cap 2048 >> $LOG 2>&1 || log "HE FAILED $name"
    stop
  fi
done
log "CHAIN DONE"
