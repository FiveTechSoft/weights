#!/usr/bin/env bash
# Sealed confirmation 1 (lab/probes/sealed/jackrong_confirm.md): base / Jackrong r16 x0.5 / placebo x0.5 on the sealed set, 2048 tokens.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"; C="$HOME/.venvs/llama-convert/Scripts/python.exe"
D=lab/runs/close_probe/distill; LOG=$D/chain_sealed_jack.log; L=hf_cache/close_lora; OUT=lab/runs/sealed
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
log() { echo "[sealed] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
mkdir -p $OUT
"$P" -m engine.sealed_set check >> $LOG 2>&1 || { log "SEAL CHECK FAILED"; exit 1; }
if [ ! -f $L/placebo_demo_jack_r16.gguf ]; then
  log "placebo"
  "$P" lab/probes/close_probe/placebo_lora.py $L/demo_jack_r16 $L/placebo_demo_jack_r16 2026 >> $LOG 2>&1 || { log "PLACEBO FAILED"; exit 1; }
  "$C" /c/Users/Anto/llamacpp/convert-src/convert_lora_to_gguf.py $L/placebo_demo_jack_r16 --base hf_cache/qwen35-9b-hf --outtype f16 --outfile $L/placebo_demo_jack_r16.gguf > $D/convert_placebo_jack.log 2>&1 || { log "CONVERT FAILED"; exit 1; }
fi
nohup $S -m hf_cache/h4/A0-Q6_K.gguf --lora $L/demo_jack_r16.gguf --lora $L/placebo_demo_jack_r16.gguf --lora-init-without-apply -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $D/sealed.server.log 2>&1 &
SP=$!; up || { log "SERVER FAILED"; kill $SP; exit 1; }
for arm in "jack_a050 0 0.5" "placebo_a050 1 0.5" "base 0 0.0"; do
  set -- $arm
  [ -f $OUT/$1.json ] && continue
  log "arm $1"
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b, sealed_set as ss
s=ss.load()
b.run('$1', 8095, out_dir='$OUT', lora=[($2, $3)], max_tokens=2048, gsm_idx=s['gsm8k'], math_idx=s['math500'], log=lambda m: print(m, flush=True))" >> $LOG 2>&1 || log "ARM FAILED $1"
done
kill $SP
log "CHAIN DONE"
