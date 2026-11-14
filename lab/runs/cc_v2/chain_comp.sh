#!/usr/bin/env bash
# Composition of conclude (cc_v2) with the Jackrong adapter (lab/probes/cc/PLAN.md, "Composition"): arms J, J+C, J+P on
# the dev set. Waits for chain_name.sh; runs only if cc_v2 passed dev. Smoke test of the two flags first.
set -u
cd /c/weights
export PYTHONUTF8=1
export CC_RUNS=lab/runs/cc_v2
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; LOG=$OUT/chain.log; COMP=$OUT/comp; L=hf_cache/close_lora
V=hf_cache/cc/cc_v2.gguf; PL=hf_cache/cc/cc_v2.placebo2026.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
# this llama-server build takes several adapters comma-separated (repeating --lora keeps only the last one); id 0 = adapter, 1 = placebo
LORAS="--lora $L/demo_jack_r16.gguf,$L/placebo_demo_jack_r16.gguf --lora-init-without-apply"
log() { echo "[cc_comp] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m $M $LORAS "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $OUT/server_comp.log 2>&1 & SP=$!; up || { log "SERVER FAILED $*"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
bench() {  # name lora_list extra_kwargs
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('$1', 8095, out_dir='$COMP', max_tokens=2048, lora=$2, log=lambda m: print(m, flush=True)$3)" >> $LOG 2>&1 || log "ARM FAILED $1"; }

until grep -qE "^\[cc_name\] (NAME DONE|cc_v2 dev .* name test not run)" $LOG 2>/dev/null; do sleep 60; done
DV=$("$P" -c "import json;print(json.load(open('$OUT/verdict.json'))['verdict'])" 2>/dev/null)
[ "$DV" = PASS ] || { log "cc_v2 dev $DV: composition not run"; exit 0; }
SC=$("$P" -c "import json;print(json.load(open('$OUT/scale.json'))['scale'])")
mkdir -p $COMP
log "smoke: adapter + control vector together (2 GSM8K items)"
serve --control-vector-scaled $V:$SC
bench smoke "[(0, 0.5)]" ", gsm_n=2, math_n=0"
grep -q '"model": "smoke"' $COMP/smoke.json 2>/dev/null || { log "SMOKE FAILED: server did not serve LoRA + control vector together"; stop; exit 1; }
log "arm JC (adapter x0.5 + cc_v2 x$SC)"
[ -f $COMP/JC.json ] || bench JC "[(0, 0.5)]" ""
stop
log "arm J (adapter x0.5)"
serve
[ -f $COMP/J.json ] || bench J "[(0, 0.5)]" ""
stop
log "arm JP (adapter x0.5 + placebo x$SC)"
serve --control-vector-scaled $PL:$SC
[ -f $COMP/JP.json ] || bench JP "[(0, 0.5)]" ""
stop
"$P" lab/probes/cc/comp.py verdict >> $LOG 2>&1
log "COMP DONE"
