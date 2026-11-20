#!/usr/bin/env bash
# HumanEval recovery arms (lab/probes/ranking/HUMANEVAL_PLAN.md, row 52): vector 0.5, 0.7, Jackrong x0.5 + vector 1.0.
# one server at a time, sha and smoke per configuration, resumable. About 7.5 GPU h.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/rank; LOG=$R/chain_he.log
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
log "start (HumanEval recovery, rank-2048; HEAD $(git rev-parse --short HEAD))"
CV=hf_cache/cc/cc_v2.gguf
log "vector sha256 $(sha256sum $CV | cut -c1-16)"
arm vec05        $R/qwen9b_vec05.json        programs/rank_vec05.wspl        hf_cache/h4/A0-Q6_K.gguf --control-vector-scaled $CV:0.5
arm vec07        $R/qwen9b_vec07.json        programs/rank_vec07.wspl        hf_cache/h4/A0-Q6_K.gguf --control-vector-scaled $CV:0.7
arm jack05_vec10 $R/qwen9b_jack05_vec10.json programs/rank_jack05_vec10.wspl hf_cache/h4/A0-Q6_K.gguf --lora hf_cache/close_lora/demo_jack_r16.gguf --lora-init-without-apply --control-vector-scaled $CV:1.0
for n in vec05 vec07 jack05_vec10; do grep -qi "no valid control vector" $R/server_$n.log && { log "FAILED: control vector rejected in $n"; exit 1; }; done
"$P" -m engine.wsplang programs/rank_compare_he.wspl --run >> $LOG 2>&1
log "CHAIN DONE"
