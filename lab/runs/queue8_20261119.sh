#!/usr/bin/env bash
# Eighth GPU queue: waits for queue 7 (neuron communities) to write QUEUE7 DONE, then runs stage 1 of the ranking
# (lab/runs/rank/chain.sh, about 14 GPU h). Sequencing by end markers only; this is the only waiter on QUEUE7 DONE.
set -eu
cd /c/weights
LOG=lab/runs/queue8_20261119.log
log() { echo "[queue8] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for QUEUE7 DONE of queue7_20261118"
until grep -q "QUEUE7 DONE" lab/runs/queue7_20261118.log 2>/dev/null; do sleep 300; done
until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do sleep 60; done
log "queue 7 finished; running the ranking stage 1"
bash lab/runs/rank/chain.sh > lab/runs/rank/chain.stdout 2> lab/runs/rank/chain.stderr || log "rank chain exited non-zero"
log "QUEUE8 DONE"
