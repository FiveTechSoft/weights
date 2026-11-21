#!/bin/bash
cd /c/weights
PY=~/.venvs/weights-cuda/Scripts/python.exe
mkdir -p lab/runs/h7
LOG=lab/runs/h7/chain.log
for t in mul3x3 mul4x3 sq4 mul2x2x1 muladd1 div6_2 mod3_1 add3x3 rev4; do
  $PY lab/probes/h7_run/run_task.py --task $t > lab/runs/h7/$t.log 2>&1; echo "$t rc=$? $(date +%H:%M:%S)" >> $LOG
done
echo "ALL DONE $(date +%H:%M:%S)" >> $LOG
