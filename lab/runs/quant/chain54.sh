#!/usr/bin/env bash
# Row 54 (lab/probes/quantize/PLAN54.md): loop control (DRY), edit re-injected as fp16 LoRA over 2 bits, Q2_K with
# imatrix. One GPU job at a time, sha logs, smoke per server, resumable (bench_run resumes from .partial.jsonl).
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/quant; LOG=$R/chain54.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8095
SP=""
Q=hf_cache/quant
A0=hf_cache/h4/A0-Q6_K.gguf
LORA=hf_cache/cc_op_v2/vector/lora_r1.gguf
log() { echo "[54] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "${SP:-}" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do log "GPU job alive, waiting"; sleep 120; done; }
smoke() {
  local code
  code=$(curl -s -o $R/smoke54.json -w "%{http_code}" -H "Content-Type: application/json" \
         -d '{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":16,"temperature":0}' \
         http://127.0.0.1:$PORT/v1/chat/completions)
  if [ "$code" = "200" ] && grep -q '"choices"' $R/smoke54.json; then log "smoke ok"; return 0; fi
  log "SMOKE FAILED (http $code)"; return 1; }
arm() {  # name model.gguf [server extras...]
  local name=$1 m=$2; shift 2
  local out=$R/rank_$name.json prog=programs/quant54_rank_$name.wspl
  if [ -f "$out" ]; then log "$name exists, skipped"; return 0; fi
  no_server
  if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
  log "arm $name: serving $m ${*:-} sha256 $(sha256sum "$m" | cut -c1-16)"
  nohup $S -m "$m" "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/server54_$name.log 2>&1 & SP=$!
  up || { log "SERVER FAILED ($name)"; stop; exit 1; }
  smoke || { stop; exit 1; }
  "$P" -m engine.wsplang "$prog" --run >> $LOG 2>&1
  stop
  [ -f "$out" ] || { log "FAILED: $out missing after $name"; exit 1; }
  log "arm $name done"
}
log "start (row 54; HEAD $(git rev-parse --short HEAD))"
no_server
[ -f $Q/qwen9b-Q2_K-imx.gguf ] || { log "FAILED: q2k_imx not built"; exit 1; }
if [ -f $R/dist_q2k_imx.json ]; then log "dist q2k_imx exists, skipped"; else log "dist q2k_imx"; "$P" -m engine.wsplang programs/quant54_dist.wspl --run >> $LOG 2>&1; fi
arm q6_lora          $A0                     --lora $LORA --lora-init-without-apply
arm iq2_lora         $Q/qwen9b-IQ2_XS.gguf   --lora $LORA --lora-init-without-apply
arm iq2_dry          $Q/qwen9b-IQ2_XS.gguf
arm compiled_iq2_dry $Q/compiled-IQ2_XS.gguf
arm q6_dry           $A0
arm q2k_imx          $Q/qwen9b-Q2_K-imx.gguf
"$P" -m engine.wsplang programs/quant54_compare.wspl --run >> $LOG 2>&1
log "CHAIN DONE"
