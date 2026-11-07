#!/usr/bin/env bash
# H6 stage 0: seeds 1, 2, 3 one after another (frozen copy; do not edit while running)
cd /c/weights
export PYTHONUTF8=1
for s in 1 2 3; do
  echo "[h6] seed $s start $(date +%H:%M:%S)"
  ~/.venvs/weights-cuda/Scripts/python.exe lab/probes/h6/train.py --seed $s > lab/runs/h6/seed$s.log 2>&1
  echo "[h6] seed $s rc=$? $(date +%H:%M:%S)"
done
