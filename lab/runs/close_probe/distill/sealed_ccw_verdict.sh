#!/usr/bin/env bash
cd /c/weights
until grep -q "CHAIN DONE" lab/runs/close_probe/distill/chain_sealed_ccw.log; do sleep 60; done
"$HOME/.venvs/weights-cuda/Scripts/python.exe" lab/probes/cc_weights/verdict.py sealed > lab/runs/sealed_ccw/verdict.log 2>&1
