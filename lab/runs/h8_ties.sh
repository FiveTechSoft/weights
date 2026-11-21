#!/bin/bash
cd /c/weights
~/.venvs/weights-cuda/Scripts/python.exe lab/probes/h8_run/h8.py --part c > lab/runs/h8/ties.log 2>&1
echo "ties rc=$? $(date +%H:%M:%S)" >> lab/runs/h8/chain.log
