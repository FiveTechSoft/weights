#!/bin/bash
cd /c/weights
PY=~/.venvs/weights-cuda/Scripts/python.exe
LOG=lab/runs/h7/post.log
$PY lab/probes/h7_run/geometry.py > lab/runs/h7/geometry.log 2>&1; echo "geometry rc=$? $(date +%H:%M:%S)" >> $LOG
$PY lab/probes/h7_run/predict.py > lab/runs/h7/predict.log 2>&1; echo "predict rc=$? $(date +%H:%M:%S)" >> $LOG
echo "ALL DONE $(date +%H:%M:%S)" >> $LOG
