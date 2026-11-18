#!/usr/bin/env bash
# Stage 1 of lab/probes/ranking/PLAN.md: six configurations on rank-2048 (dev-2048 + TriviaQA 200), one llama-server at
# a time, served sha and a one-request smoke per configuration, resumable (an arm whose JSON exists is skipped).
# About 14 GPU h. Launched by lab/runs/queue8_20261119.sh after QUEUE7 DONE.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/rank; LOG=$R/chain.log
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
log "start (stage 1, rank-2048; HEAD $(git rev-parse --short HEAD))"
arm base     $R/qwen9b_base.json     programs/rank_base.wspl     hf_cache/h4/A0-Q6_K.gguf
arm mimo     $R/mimo.json            programs/rank_mimo.wspl     /c/Users/Anto/llamacpp/mimo-v2.6-distill-qwen-9b-Q6_K.gguf
arm jack05   $R/qwen9b_jack05.json   programs/rank_jack05.wspl   hf_cache/h4/A0-Q6_K.gguf --lora hf_cache/close_lora/demo_jack_r16.gguf --lora-init-without-apply
arm compiled $R/qwen9b_compiled.json programs/rank_compiled.wspl hf_cache/cc_op_v2/merged-Q6_K.gguf
arm llama    $R/r1llama8b.json       programs/rank_llama.wspl    hf_cache/r1llama8b-gguf/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf
arm phi      $R/phi4mr.json          programs/rank_phi.wspl      hf_cache/phi4mr-gguf/Phi-4-mini-reasoning-Q6_K.gguf
log "comparisons and table"
"$P" -m engine.wsplang programs/rank_compare.wspl --run >> $LOG 2>&1
log "CHAIN DONE"
