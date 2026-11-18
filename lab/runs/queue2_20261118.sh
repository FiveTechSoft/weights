#!/usr/bin/env bash
# Second GPU queue of 2026-11-18: waits for the first queue (cc_bias -> preserve base -> skill2) to write QUEUE DONE,
# then runs the 9B noise-band chain for the standard benchmark suite (lab/runs/bench/chain_band9b.sh). Sequencing by
# end markers only, so two chains never serve at the same time.
set -eu
cd /c/weights
LOG=lab/runs/queue2_20261118.log
log() { echo "[queue2] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for QUEUE DONE of queue_20261118"
until grep -q "QUEUE DONE" lab/runs/queue_20261118.log 2>/dev/null; do sleep 300; done
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do sleep 60; done
log "queue 1 finished; running the 9B band chain"
bash lab/runs/bench/chain_band9b.sh > lab/runs/bench/chain_band9b.stdout 2> lab/runs/bench/chain_band9b.stderr || log "band chain exited non-zero"
log "QUEUE2 DONE"
