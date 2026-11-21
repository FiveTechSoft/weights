#!/usr/bin/env bash
# lab/probes/ternary/PLAN.md step 3 (informational): the 0.8B base and the Exp 8 replica quantised to TQ2_0 with
# llama-quantize (no recovery training), served on the CPU, 10 GSM8K items at 512 tokens each. Outputs in
# lab/runs/ternary/ (reports) and servidos/ (GGUF, ignored by git). One llama-server at a time (checked).
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
Q=/c/Users/Anto/llamacpp/bin/llama-quantize.exe
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
R=lab/runs/ternary; LOG=$R/tq_smoke.log; PORT=8098; SP=""
log() { echo "[tq_smoke] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 2
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 2; SP=""; }
up() { ok=0; for k in $(seq 1 60); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 2; done; [ $ok = 1 ]; }
if tasklist 2>/dev/null | grep -qi "llama-server"; then log "another llama-server is alive: not started"; exit 1; fi
declare -A SRC
SRC[base]=hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf
SRC[exp8]=servidos/qwen35-08b__exp8-replica.gguf
log "start"
for a in base exp8; do
  out=servidos/qwen35-08b__${a}-TQ2_0.gguf
  # both sources are Q6_K GGUF (the replica was edited in Q6_K), so this is a requantisation Q6_K -> TQ2_0
  # write to .tmp and rename on success, so a failed quantize never leaves a file the next run would trust
  [ -f $out ] || { log "quantize $a -> TQ2_0 (from Q6_K, --allow-requantize)"; rm -f $out.tmp
    $Q --allow-requantize ${SRC[$a]} $out.tmp TQ2_0 > $R/quantize_$a.log 2>&1; mv $out.tmp $out; }
  log "$a TQ2_0 sha256 $(sha256sum $out | cut -c1-16) size $(stat -c %s $out)"
  [ -f $R/tq_$a.json ] && { log "tq_$a exists, skipped"; continue; }
  log "serving $out on the CPU"
  nohup $S -m $out -ngl 0 -c 4096 -np 2 --jinja --port $PORT > $R/tq_server_$a.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $a"; stop; exit 1; }
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('tq_$a', $PORT, out_dir='$R', max_tokens=512, parallel=2, gsm_idx=list(range(10)), math_idx=[], log=lambda m: print(m, flush=True))" >> $LOG 2>&1
  stop
done
log "SMOKE DONE"
