#!/usr/bin/env bash
# taskvec_mult — unique experiment (driver decision 2026-11-09): task vector + selective SVD over Qwen3.5-0.8B.
# Prereg: lab/probes/taskvec_mult/PREREG.md. SHORT protocol (driver 2026-11-09 ~11:35):
#   Phase A cheap: donor mult + adapters r8/16/64/256 mult + placebo mult  -> screen (analyze.py --screen)
#   Phase B on the WINNER only: gsm8k s10 + math500 s10 (cap 1536) -> full analyze + verdict.
# Waits until the 12GB GPU is free (no llama-server running). Resumable via file markers.
# usage: bash lab/probes/taskvec_mult/chain_taskvec.sh
set -u
export PYTHONUTF8=1
cd /c/weights
export WSP_LORA_CONVERTER=/c/Users/Anto/llamacpp/convert-src/convert_lora_to_gguf.py
export WSP_CONVERT_PYTHON="$HOME/.venvs/llama-convert/Scripts/python.exe"
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
Q="$P"
D=lab/runs/taskvec_mult; LOG=$D/chain.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
M08=hf_cache/qwen35-08b-hf
B08=hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf
CS=/c/Users/Anto/llamacpp/convert-src/convert_hf_to_gguf.py
LC=/c/Users/Anto/llamacpp/convert-src/convert_lora_to_gguf.py
PORT=8096
FE=lab/probes/close_probe_run/fast_eval.py
EM=lab/probes/taskvec_mult/eval_mult.py

log() { echo "[taskvec] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 60); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
serve() { nohup $S -m "$1" ${2:+--lora "$2" --lora-init-without-apply} -ngl 99 -c 3072 -np 12 -fa on --jinja --port $PORT >> $D/server.log 2>&1 & SP=$!; up || { log "SERVER FAILED ($1)"; kill $SP; exit 1; }; }
down() { kill $SP 2>/dev/null; sleep 3; }
mult() { $P $EM "$1" $PORT >> $LOG 2>&1 || { log "EVALMULT FAILED $1"; exit 1; }; }
battery() { log "battery $1"; $P $FE "gsm_$2" $PORT --bench gsm8k --cap 1536 --guard --parallel 12 ${3:+--lora $3} >> $LOG 2>&1 && mv lab/runs/close_probe/fast/gsm_$2.json $D/fast/ || { log "GSM FAILED $2"; exit 1; }
  $P $FE "math_$2" $PORT --bench math500 --cap 1536 --guard --parallel 12 ${3:+--lora $3} >> $LOG 2>&1 && mv lab/runs/close_probe/fast/math_$2.json $D/fast/ || { log "MATH FAILED $2"; exit 1; } }
lora_gguf() { "$Q" $LC "$1" --base $M08 --outtype f16 --outfile "$2" >> $LOG 2>&1; }
gpu_free() { pgrep -x llama-server >/dev/null 2>&1; }

mkdir -p $D $D/fast
log "start"
while gpu_free; do sleep 60; done
log "GPU free"

# 1) data (already generated)
[ -f $D/mult_test.jsonl ] || { $P lab/probes/taskvec_mult/gen_mult.py --root $D >> $LOG 2>&1 || { log "GEN FAILED"; exit 1; }; }
log "data ready"

# 2) base mult eval (done: 0/200)
if [ ! -f $D/fast/base_mult.json ]; then
  serve $B08; mult base_mult; down
fi
log "base ready"

# 3) train donor (full fine-tune; ~30-40 min)
if [ ! -f $D/train_donor.json ]; then
  $P lab/probes/taskvec_mult/train_donor.py --base $M08 --train $D/mult_train.jsonl --val $D/mult_val.jsonl --out $D/donor_b >> $LOG 2>&1 || { log "TRAIN FAILED"; exit 1; }
fi
log "donor trained"

# 4) donor full -> Q6_K -> mult eval only (criteria D)
if [ ! -f $D/fast/donor_mult.json ]; then
  [ -f $D/donor_b-Q6_K.gguf ] || {
    "$P" $CS $D/donor_b --outtype bf16 --outfile $D/donor_b-bf16.gguf >> $LOG 2>&1 || { log "CONVERT DONOR FAILED"; exit 1; }
    /c/Users/Anto/llamacpp/bin/llama-quantize.exe $D/donor_b-bf16.gguf $D/donor_b-Q6_K.gguf Q6_K > $D/quantize_donor.log 2>&1 || { log "QUANTIZE DONOR FAILED"; exit 1; }
    rm -f $D/donor_b-bf16.gguf
  }
  serve $D/donor_b-Q6_K.gguf; mult donor_mult; down
fi
log "donor eval done"

# 5) adapters via delta_to_lora + Phase A mult screen
[ -f $D/adapter_r256/adapter_model.safetensors ] || { $P -m engine.wsplang programs/taskvec_mult.wspl --run >> $LOG 2>&1 || { log "EXTRACT FAILED"; exit 1; }; }
log "adapters extracted"
for r in 8 16 64 256; do
  if [ ! -f $D/fast/adapter_r${r}_mult.json ]; then
    [ -f $D/adapter_r${r}.gguf ] || lora_gguf $D/adapter_r${r} $D/adapter_r${r}.gguf
    serve $B08 $D/adapter_r${r}.gguf; mult adapter_r${r}_mult; down
  fi
done
log "adapter screen done"

# 6) placebos (mult only)
for r in 16 256; do
  if [ ! -f $D/fast/placebo_r${r}_mult.json ]; then
    [ -f $D/placebo_r${r}/adapter_model.safetensors ] || $P lab/probes/close_probe/placebo_lora.py $D/adapter_r${r} $D/placebo_r${r} 0 >> $LOG 2>&1
    [ -f $D/placebo_r${r}.gguf ] || lora_gguf $D/placebo_r${r} $D/placebo_r${r}.gguf
    serve $B08 $D/placebo_r${r}.gguf; mult placebo_r${r}_mult; down
  fi
done
$P lab/probes/taskvec_mult/analyze.py --screen >> $LOG 2>&1 || { log "SCREEN FAILED"; exit 1; }
log "screen done"
W=$(python - <<'EOF'
import json
try:
    print(json.load(open(r"lab/runs/taskvec_mult/fast/screen.json"))["best"] or "")
except Exception:
    print("")
EOF
)
log "winner rank: ${W:-none}"

# 7) Phase B: winner retention battery only
if [ -n "${W:-}" ]; then
  if [ ! -f $D/fast/gsm_adapter_r${W}_08.json ]; then
    while gpu_free; do sleep 60; done
    serve $B08 $D/adapter_r${W}.gguf
    battery adapter_r${W}_mult adapter_r${W}_08 0:1
    down
  fi
fi

# 8) verdict
$P lab/probes/taskvec_mult/analyze.py >> $LOG 2>&1 && log "CHAIN DONE" || log "ANALYZE FAILED"
