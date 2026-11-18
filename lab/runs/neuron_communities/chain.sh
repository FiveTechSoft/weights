#!/bin/bash
# Phase A of lab/probes/neuron_communities/PLAN.md on the 9B (roadmap gap 28, prediction row 50). Draft for the
# coordinator: no queue wait inside (the queue script waits for its marker); one GPU job; nothing else on the card.
# Steps and estimates (RTX 3060, 9B bf16 streamed 4 layers at a time; 44 discovery traces, ~48k tokens, 18 positions):
#   1. unit tests (CPU, ~20 s)
#   2. attribution: forward with group caches (~5 min) + backward 8 groups x 44 traces x 18 vJPs (~60-100 min);
#      disk cache ~10 GB under lab/runs/neuron_communities/cache, deleted at the end
#   3. communities (CPU, ~1 min)
#   4. ablation on the finished held-out traces (9): base + A1 + 20 placebos + A2 + A2-random + A3 per layer,
#      one-row route, batches of 8 (~10-20 min)
set -eu
cd /c/weights
OUT=lab/runs/neuron_communities
LOG=$OUT/chain.log
PY="$HOME/.venvs/weights-cuda/Scripts/python.exe"
GROUP="${GROUP:-4}"
mkdir -p "$OUT"
log() { echo "[nc] $* ($(date +%T))" | tee -a "$LOG"; }
trap 'log "FAILED at line $LINENO"' ERR
log "start; HEAD $(git rev-parse --short HEAD); group $GROUP"
if tasklist 2>/dev/null | grep -qi "llama-server"; then
  log "FAILED: llama-server is running; the GPU is not free"
  exit 2
fi
if [ -f "$OUT/attributions.npz" ]; then
  log "attributions.npz exists; skipping attribution"
else
  log "step 1: unit tests"
  PYTHONUTF8=1 python -m unittest tests.test_neuron_communities 2>&1 | tail -3 | tee -a "$LOG"
  log "step 2: attribution (9B, cuda, bf16)"
  PYTHONUTF8=1 "$PY" lab/probes/neuron_communities/attribute.py run --out "$OUT" --device cuda --dtype bfloat16 \
      --group "$GROUP" --keep 64 2>> "$OUT/attribute.err" | tee -a "$LOG"
fi
log "step 3: communities"
PYTHONUTF8=1 "$PY" lab/probes/neuron_communities/attribute.py communities --out "$OUT" 2>> "$OUT/attribute.err" | tee -a "$LOG"
log "step 4: ablation on held-out (split)"
PYTHONUTF8=1 "$PY" lab/probes/neuron_communities/attribute.py ablate --out "$OUT" --device cuda --heldout split \
    --placebos 20 --cond-batch 8 2>> "$OUT/attribute.err" | tee -a "$LOG"
log "summary: $(PYTHONUTF8=1 python -c "import json;s=json.load(open('$OUT/summary.json'));print({k:s[k] for k in ('n_heldout','A1_pass','A2_pass','verdict')})")"
log "CHAIN DONE"
