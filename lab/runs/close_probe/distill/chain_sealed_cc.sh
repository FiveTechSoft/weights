#!/usr/bin/env bash
# Sealed confirmation 2 (lab/probes/sealed/cc_confirm.md): base / cc x1.0 / placebo x1.0 on the sealed set, 2048 tokens.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
D=lab/runs/close_probe/distill; LOG=$D/chain_sealed_cc.log; OUT=lab/runs/sealed_cc
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
log() { echo "[sealed_cc] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m $M "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $D/sealed_cc.server.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
mkdir -p $OUT
"$P" -m engine.sealed_set check >> $LOG 2>&1 || { log "SEAL CHECK FAILED"; exit 1; }
for arm in "vec hf_cache/cc/cc.gguf" "placebo hf_cache/cc/cc.placebo2026.gguf" "base -"; do
  set -- $arm
  [ -f $OUT/$1.json ] && continue
  log "arm $1"
  if [ "$2" = "-" ]; then serve; else serve --control-vector-scaled $2:1.0; fi
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b, sealed_set as ss
s=ss.load()
b.run('$1', 8095, out_dir='$OUT', max_tokens=2048, gsm_idx=s['gsm8k'], math_idx=s['math500'], log=lambda m: print(m, flush=True))" >> $LOG 2>&1 || log "ARM FAILED $1"
  stop
done
log "CHAIN DONE"
