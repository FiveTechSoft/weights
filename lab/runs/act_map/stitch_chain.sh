#!/bin/bash
cd /c/weights
PY=$HOME/.venvs/weights-cuda/Scripts/python.exe
echo "[st] start $(date +%T)"
$PY lab/probes/act_map/stitch.py run --src q08 --dst q9 || echo "[st] q08->q9 FAILED"
echo "[st] q08->q9 done $(date +%T)"
$PY lab/probes/act_map/stitch.py run --src q9 --dst q08 || echo "[st] q9->q08 FAILED"
echo "[st] CHAIN DONE $(date +%T)"
