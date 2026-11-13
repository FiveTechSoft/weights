#!/usr/bin/env bash
# Prompt control (lab/probes/cc/PLAN.md, "Prompt control", prediction row 28): base A0 at 2,048 with a closing sentence
# (P_close) or a neutral sentence (P_neutral) appended after the usual suffix; dev items; no vector, no adapter.
# Waits for the recompilation chain (one GPU job at a time).
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; LOG=$OUT/chain.log; PR=$OUT/prompt
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
CLOSE=" Once you have the answer, stop reasoning and give it: no re-checking, no restating."
NEUTRAL=" Use clear notation and define every variable you introduce before using it."
log() { echo "[cc_prompt] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m $M -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $OUT/server_prompt.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
bench() {  # name extra
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('$1', 8095, out_dir='$PR', max_tokens=2048, extra=sys.argv[1], log=lambda m: print(m, flush=True))" "$2" >> $LOG 2>&1 || log "ARM FAILED $1"; }

until grep -qE "^\[cc_w2\] W2 DONE" $LOG 2>/dev/null; do sleep 120; done
mkdir -p $PR
"$P" -c "import sys; sys.path.insert(0,'.'); import inspect; from engine import bench_runner as b; assert 'extra' in inspect.signature(b.run).parameters" || { log "bench_runner has no 'extra': not run"; exit 1; }
serve
[ -f $PR/P_close.json ] || { log "arm P_close"; bench P_close "$CLOSE"; }
[ -f $PR/P_neutral.json ] || { log "arm P_neutral"; bench P_neutral "$NEUTRAL"; }
stop
log "PROMPT DONE"
