#!/usr/bin/env bash
# Ninth GPU queue: waits for the HumanEval-recovery chain (lab/runs/rank/chain_he.log, CHAIN DONE), then runs the 2-bit
# quantisation chain (lab/runs/quant/chain.sh, about 8 GPU h). Only waiter on that marker.
set -eu
cd /c/weights
LOG=lab/runs/queue9_20261120.log
log() { echo "[queue9] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for CHAIN DONE in lab/runs/rank/chain_he.log"
until grep -q "CHAIN DONE\|FAILED" lab/runs/rank/chain_he.log 2>/dev/null; do sleep 300; done
until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do sleep 60; done
log "HumanEval chain finished; running the quantisation chain"
bash lab/runs/quant/chain.sh > lab/runs/quant/chain.stdout 2> lab/runs/quant/chain.stderr || log "quant chain exited non-zero"
log "QUEUE9 DONE"
