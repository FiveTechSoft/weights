#!/usr/bin/env bash
# GPU queue of 2026-11-18 (roadmap: 18 -> 19 base -> 17). Waits for the running cc_bias chain to write CHAIN DONE,
# then runs the preservation battery on the 9B base (lab/runs/preserve/smoke.sh), then the skill2 chain
# (lab/runs/skill2/chain.sh). Each child has its own set -eu, trap, guards and smoke; this script only sequences them
# by their end markers so two chains never serve at the same time (their own no_server waits would race in the gaps
# between arms). One GPU job at a time.
set -eu
cd /c/weights
LOG=lab/runs/queue_20261118.log
log() { echo "[queue] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start; waiting for cc_bias CHAIN DONE"
until grep -q "CHAIN DONE" lab/runs/cc_bias/chain.log 2>/dev/null; do
  if grep -q "FAILED" lab/runs/cc_bias/chain.log 2>/dev/null; then log "cc_bias FAILED: queue continues after it stops"; break; fi
  sleep 120
done
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do sleep 60; done
log "cc_bias finished; running preserve smoke (9B base battery)"
bash lab/runs/preserve/smoke.sh > lab/runs/preserve/smoke.stdout 2> lab/runs/preserve/smoke.stderr || log "preserve smoke exited non-zero"
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do sleep 60; done
log "preserve done; running skill2 chain"
bash lab/runs/skill2/chain.sh > lab/runs/skill2/chain.stdout 2> lab/runs/skill2/chain.stderr || log "skill2 chain exited non-zero"
log "QUEUE DONE"
