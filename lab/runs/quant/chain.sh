#!/usr/bin/env bash
# 2-bit quantisation arms (lab/probes/quantize/PLAN.md, row 53): build (CPU + imatrix on GPU), dist-shift, rank-2048.
# One GPU job at a time, sha logs, smoke per server, resumable. About 8 GPU h. Queue 9 after the HumanEval chain.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/quant; LOG=$R/chain.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8095
SP=""
log() { echo "[rank] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do log "GPU job alive, waiting"; sleep 120; done; }
smoke() {  # one short request must answer 200 with a choice
  local code
  code=$(curl -s -o $R/smoke.json -w "%{http_code}" -H "Content-Type: application/json" \
         -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":16,"temperature":0}' \
         http://127.0.0.1:$PORT/v1/chat/completions)
  if [ "$code" = "200" ] && grep -q '"choices"' $R/smoke.json; then log "smoke ok"; return 0; fi
  log "SMOKE FAILED (http $code)"; return 1; }
arm() {  # name out.json program model.gguf [server extras...]
  local name=$1 out=$2 prog=$3 m=$4; shift 4
  if [ -f "$out" ]; then log "$name exists, skipped"; return 0; fi
  no_server
  if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
  log "arm $name: serving $m $* sha256 $(sha256sum "$m" | cut -c1-16)"
  nohup $S -m "$m" "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server_$name.log 2>&1 & SP=$!
  up || { log "SERVER FAILED ($name)"; stop; exit 1; }
  smoke || { stop; exit 1; }
  "$P" -m engine.wsplang "$prog" --run >> $LOG 2>&1
  stop
  log "arm $name done"
}
log "start (2-bit quantisation; HEAD $(git rev-parse --short HEAD))"
Q=hf_cache/quant
no_server
if [ -f $Q/qwen9b-Q2_K.gguf ] && [ -f $Q/qwen9b-IQ2_XS.gguf ] && [ -f $Q/compiled-IQ2_XS.gguf ]; then log "quantised files exist, build skipped"
else log "build: quantize (imatrix on GPU)"; "$P" -m engine.wsplang programs/quant_build.wspl --run >> $LOG 2>&1; fi
for f in $Q/qwen9b-Q2_K.gguf $Q/qwen9b-IQ2_XS.gguf $Q/compiled-IQ2_XS.gguf; do [ -f $f ] || { log "FAILED: $f missing after build"; exit 1; }; log "built $f $(stat -c %s $f) bytes sha256 $(sha256sum $f | cut -c1-16)"; done
if [ -f $R/dist_compiled_iq2xs.json ]; then log "dist-shift exists, skipped"; else log "dist-shift"; "$P" -m engine.wsplang programs/quant_dist.wspl --run >> $LOG 2>&1; fi
arm base_q2k       $R/rank_base_q2k.json       programs/quant_rank_base_q2k.wspl       $Q/qwen9b-Q2_K.gguf
arm base_iq2xs     $R/rank_base_iq2xs.json     programs/quant_rank_base_iq2xs.wspl     $Q/qwen9b-IQ2_XS.gguf
arm compiled_iq2xs $R/rank_compiled_iq2xs.json programs/quant_rank_compiled_iq2xs.wspl $Q/compiled-IQ2_XS.gguf
"$P" -m engine.wsplang programs/quant_compare.wspl --run >> $LOG 2>&1
log "CHAIN DONE"
