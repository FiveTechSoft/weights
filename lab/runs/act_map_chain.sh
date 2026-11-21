#!/bin/bash
# act_map gate 1: captures (GPU, after the H6g/H6h chain) then fit (CPU). One GPU job at a time.
cd /c/weights
P=$HOME/.venvs/weights-cuda/Scripts/python.exe
A=lab/probes/act_map/act_map.py
mkdir -p lab/runs/act_map
LOG=lab/runs/act_map/chain.log
log() { echo "[am] $* ($(date +%H:%M:%S))" >> $LOG; }
log "waiting for h6gh CHAIN DONE"
until grep -q "CHAIN DONE" lab/runs/h6/h6gh_chain.log 2>/dev/null; do sleep 60; done
log "start; free disk $(df -h /c | tail -1 | awk '{print $4}')"
"$P" $A capture --model hf_cache/qwen35-08b-hf --name q08 >> $LOG 2>&1 || { log "CAPTURE q08 FAILED"; log "CHAIN DONE"; exit 1; }
"$P" $A capture --model hf_cache/qwen35-9b-hf --name q9 >> $LOG 2>&1 || { log "CAPTURE q9 FAILED"; log "CHAIN DONE"; exit 1; }
log "captures done; free disk $(df -h /c | tail -1 | awk '{print $4}')"
"$P" $A fit --a q9 --b q08 --save-maps >> $LOG 2>&1 || log "FIT FAILED"
log "CHAIN DONE"
