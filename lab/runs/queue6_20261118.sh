#!/usr/bin/env bash
# Sixth GPU queue of 2026-11-18: waits for queue 5 (dt_bias chain) to write QUEUE5 DONE, then runs the missing
# dist-shift arm (lab/runs/bench/chain_dist9b_merged.sh: compiled conclude merged Q6_K through llama-perplexity, about
# 10 GPU minutes, then the comparisons and the table). Sequencing by end markers only; queue 5 is the only waiter on
# QUEUE4 DONE, this one the only waiter on QUEUE5 DONE.
set -eu
cd /c/weights
LOG=lab/runs/queue6_20261118.log
log() { echo "[queue6] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for QUEUE5 DONE of queue5_20261118"
until grep -q "QUEUE5 DONE" lab/runs/queue5_20261118.log 2>/dev/null; do sleep 300; done
until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do sleep 60; done
log "queue 5 finished; running the dist-shift merged arm"
bash lab/runs/bench/chain_dist9b_merged.sh > lab/runs/bench/chain_dist9b_merged.stdout 2> lab/runs/bench/chain_dist9b_merged.stderr || log "dist9b_merged chain exited non-zero"
log "QUEUE6 DONE"
