#!/usr/bin/env bash
# Fifth GPU queue of 2026-11-18: waits for queue 4 (9B base on dev-2048-full) to write QUEUE4 DONE, then runs the
# dt_bias chain (lab/runs/dt_bias/chain.sh: +0.5, -0.5 and random-sign placebo on the 9B's linear-attention time-step
# biases, dev-2048 with the battery, about 5.5 GPU hours). Sequencing by end markers only.
set -eu
cd /c/weights
LOG=lab/runs/queue5_20261118.log
log() { echo "[queue5] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for QUEUE4 DONE of queue4_20261118"
until grep -q "QUEUE4 DONE" lab/runs/queue4_20261118.log 2>/dev/null; do sleep 300; done
until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do sleep 60; done
log "queue 4 finished; running the dt_bias chain"
bash lab/runs/dt_bias/chain.sh > lab/runs/dt_bias/chain.stdout 2> lab/runs/dt_bias/chain.stderr || log "dt_bias chain exited non-zero"
log "QUEUE5 DONE"
