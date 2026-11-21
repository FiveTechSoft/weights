#!/usr/bin/env bash
# cc in the neuron basis, phase A (lab/probes/cc_neurons/PLAN.md): one GPU pass after the sealed cc_weights chain ends.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_neurons; LOG=$OUT/run.log
mkdir -p $OUT
WAITPID=${WAITPID:-0}
echo "[ccn] waiting for PID $WAITPID ($(date +%H:%M:%S))" >> $LOG
while kill -0 $WAITPID 2>/dev/null; do sleep 60; done
echo "[ccn] start ($(date +%H:%M:%S))" >> $LOG
"$P" lab/probes/cc_neurons/decompose.py run $OUT >> $LOG 2>&1 || echo "[ccn] FAILED" >> $LOG
echo "[ccn] CHAIN DONE ($(date +%H:%M:%S))" >> $LOG
