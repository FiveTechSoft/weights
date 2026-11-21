#!/usr/bin/env bash
# implant_detect (lab/probes/implant_detect/PLAN.md): CPU only, ~5 GB RAM; after cc_neurons (which needs ~18 GB RAM).
set -u
cd /c/weights
export PYTHONUTF8=1
OUT=lab/runs/implant_detect; LOG=$OUT/run.log
mkdir -p $OUT
echo "[imp] waiting for cc_neurons ($(date +%H:%M:%S))" >> $LOG
until grep -q "CHAIN DONE" lab/runs/cc_neurons/run.log 2>/dev/null; do sleep 60; done
echo "[imp] start ($(date +%H:%M:%S))" >> $LOG
"$HOME/.venvs/weights-cuda/Scripts/python.exe" lab/probes/implant_detect/detect.py run $OUT >> $LOG 2>&1 || echo "[imp] FAILED" >> $LOG
echo "[imp] CHAIN DONE ($(date +%H:%M:%S))" >> $LOG
