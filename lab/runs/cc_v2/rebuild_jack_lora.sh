#!/usr/bin/env bash
# hf_cache/close_lora/ was deleted after 43.15; rebuild the Jackrong rank-16 adapter with the same op and program
# (programs/transplant_jackrong_math.wspl: exact SVD, deterministic) and its norm-matched placebo (placebo_lora.py,
# seed 2026, as in lab/probes/sealed/jackrong_confirm.md), then start the composition chain again.
set -u
cd /c/weights
export PYTHONUTF8=1
export WSP_LORA_CONVERTER=/c/Users/Anto/llamacpp/convert-src/convert_lora_to_gguf.py
export WSP_CONVERT_PYTHON="$HOME/.venvs/llama-convert/Scripts/python.exe"
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
L=hf_cache/close_lora; LOG=lab/runs/cc_v2/chain.log
log() { echo "[jack_lora] $* ($(date +%H:%M:%S))" >> $LOG; }
log "rebuilding the Jackrong r16 adapter (op delta_to_lora, programs/transplant_jackrong_math.wspl)"
[ -f $L/demo_jack_r16.gguf ] || "$P" -m engine.wsplang programs/transplant_jackrong_math.wspl --run >> lab/runs/cc_v2/rebuild_jack_lora.log 2>&1 || { log "OP FAILED"; exit 1; }
[ -f $L/demo_jack_r16.gguf ] || { log "no adapter GGUF written"; exit 1; }
if [ ! -f $L/placebo_demo_jack_r16.gguf ]; then
  log "placebo (same rank and norm, seed 2026) and its GGUF"
  [ -d $L/placebo_demo_jack_r16 ] || "$P" lab/probes/close_probe/placebo_lora.py $L/demo_jack_r16 $L/placebo_demo_jack_r16 2026 >> lab/runs/cc_v2/rebuild_jack_lora.log 2>&1 || { log "PLACEBO FAILED"; exit 1; }
  "$WSP_CONVERT_PYTHON" "$WSP_LORA_CONVERTER" $L/placebo_demo_jack_r16 --base hf_cache/qwen35-9b-hf --outtype f16 --outfile $L/placebo_demo_jack_r16.gguf >> lab/runs/cc_v2/rebuild_jack_lora.log 2>&1 || { log "PLACEBO CONVERT FAILED"; exit 1; }
fi
sha256sum $L/demo_jack_r16.gguf $L/placebo_demo_jack_r16.gguf >> $LOG
log "adapter and placebo ready; starting the composition chain"
nohup bash lab/runs/cc_v2/chain_comp.sh > /dev/null 2>&1 &
log "JACK LORA DONE"
