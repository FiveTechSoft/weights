#!/usr/bin/env bash
# Seventh GPU queue of 2026-11-18: waits for queue 6 (dist-shift merged arm) to write QUEUE6 DONE, then runs phase A of
# lab/probes/neuron_communities/PLAN.md (lab/runs/neuron_communities/chain.sh: exact-gradient attribution of the
# </think> logit on the 9B, one layer on the GPU at a time, graph + Louvain, necessity / sufficiency tests; about 2 GPU
# h; HF torch job, so it must never overlap llama-server). Sequencing by end markers only: queue 6 is the only waiter
# on QUEUE5 DONE, this one the only waiter on QUEUE6 DONE.
set -eu
cd /c/weights
LOG=lab/runs/queue7_20261118.log
log() { echo "[queue7] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for QUEUE6 DONE of queue6_20261118"
until grep -q "QUEUE6 DONE" lab/runs/queue6_20261118.log 2>/dev/null; do sleep 300; done
until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do sleep 60; done
log "queue 6 finished; running neuron_communities phase A"
bash lab/runs/neuron_communities/chain.sh > lab/runs/neuron_communities/chain.stdout 2> lab/runs/neuron_communities/chain.stderr || log "neuron_communities chain exited non-zero"
log "QUEUE7 DONE"
