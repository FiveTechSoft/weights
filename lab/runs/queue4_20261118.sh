#!/usr/bin/env bash
# Fourth GPU queue of 2026-11-18: waits for queue 3 (dist-shift) to write QUEUE3 DONE, then runs the 9B base on
# dev-2048-full (lab/runs/bench/chain_full9b.sh, about 5.5 GPU hours). Sequencing by end markers only.
set -eu
cd /c/weights
LOG=lab/runs/queue4_20261118.log
log() { echo "[queue4] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for QUEUE3 DONE of queue3_20261118"
until grep -q "QUEUE3 DONE" lab/runs/queue3_20261118.log 2>/dev/null; do sleep 300; done
until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do sleep 60; done
log "queue 3 finished; running the dev-2048-full base chain"
bash lab/runs/bench/chain_full9b.sh > lab/runs/bench/chain_full9b.stdout 2> lab/runs/bench/chain_full9b.stderr || log "full chain exited non-zero"
log "QUEUE4 DONE"
