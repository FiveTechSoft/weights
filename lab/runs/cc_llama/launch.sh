#!/usr/bin/env bash
# Waits for the download, runs the stage-0 probe, and starts chain.sh only if the probe's own readings allow it
# (lab/probes/cc_llama/PLAN.md): budget.txt written (P2), control vector applied (P3), every ffn_down Q6_K (P6).
# Anything else stops here and is read by hand. set -eu; one GPU job at a time (probe.sh and chain.sh wait themselves).
set -eu
cd /c/weights
R=lab/runs/cc_llama; LOG=$R/launch.log
log() { echo "[launch] $* ($(date +%H:%M:%S))" >> $LOG; }
log "waiting for DOWNLOAD DONE"
until grep -q "DOWNLOAD DONE" $R/download.log 2>/dev/null; do sleep 60; done
grep -q "GGUF_EXIT 0" $R/download.log || { log "GGUF download failed"; exit 1; }
log "download done; probe"
bash $R/probe.sh 2>> $R/probe.stderr || { log "probe exited $?"; exit 1; }
grep -q "PROBE DONE" $R/probe.log || { log "no PROBE DONE"; exit 1; }
[ -f $R/budget.txt ] || { log "P2 failed at both budgets: no chain"; exit 0; }
grep -q '"P3_applied": true' $R/probe.log || { log "P3 failed (outputs identical with the random vector): no chain"; exit 0; }
grep -q '"P6_all_q6k": true' $R/probe.log || { log "P6 failed (ffn_down not all Q6_K or blocks != 32): no chain"; exit 0; }
log "probe passed (budget $(cat $R/budget.txt)); chain"
bash $R/chain.sh 2>> $R/chain.stderr || log "chain exited $?"
log "LAUNCH DONE"
