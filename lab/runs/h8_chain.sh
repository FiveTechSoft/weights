#!/bin/bash
cd /c/weights
mkdir -p lab/runs/h8
~/.venvs/weights-cuda/Scripts/python.exe lab/probes/h8_run/h8.py --part all > lab/runs/h8/h8.log 2>&1
echo "h8 rc=$? $(date +%H:%M:%S)" >> lab/runs/h8/chain.log
