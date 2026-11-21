#!/usr/bin/env bash
# energy verdict (delta_to_lora rank 16, read from the hub) for 4 new Qwen3.5-9B donors: code x3, medical x1
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"; C="$HOME/.venvs/llama-convert/Scripts/python.exe"
export WSP_LORA_CONVERTER=/c/Users/Anto/llamacpp/convert-src/convert_lora_to_gguf.py WSP_CONVERT_PYTHON="$C"
LOG=lab/runs/close_probe/distill/chain_donors4.log
log() { echo "[donors4] $* ($(date +%H:%M:%S))" >> $LOG; }
for d in omnicoder jackod oxcoder huatuo; do
  log "op $d"
  "$P" -m engine.wsplang programs/extract_$d.wspl --run >> $LOG 2>&1 || log "OP FAILED $d"
  f=hf_cache/close_lora/${d}_r16/energy.json; [ -f $f ] && log "energy $d $(python -c "import json;print(json.load(open('$f')).get('verdict'), json.load(open('$f')).get('weighted_kept_energy', ''))")"
done
log "CHAIN DONE"
