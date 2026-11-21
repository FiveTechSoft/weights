#!/usr/bin/env bash
# Third GPU queue of 2026-11-18: waits for queue 2 (9B noise band) to write QUEUE2 DONE, then runs the dist-shift chain
# (lab/runs/bench/chain_dist9b.sh: llama-perplexity on the 9B base, compiled, placebo compiled, vector and placebo
# vector; saves the base logits once). Sequencing by end markers so two GPU jobs never overlap.
set -eu
cd /c/weights
LOG=lab/runs/queue3_20261118.log
log() { echo "[queue3] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for QUEUE2 DONE of queue2_20261118"
until grep -q "QUEUE2 DONE" lab/runs/queue2_20261118.log 2>/dev/null; do sleep 300; done
until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do sleep 60; done
log "queue 2 finished; running the dist-shift chain"
bash lab/runs/bench/chain_dist9b.sh > lab/runs/bench/chain_dist9b.stdout 2> lab/runs/bench/chain_dist9b.stderr || log "dist-shift chain exited non-zero"
log "QUEUE3 DONE"
