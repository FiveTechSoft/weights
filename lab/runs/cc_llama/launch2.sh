#!/usr/bin/env bash
# Second launcher (the first was stopped by hand after P4 showed the slow-tokenizer bug; engine/tok.py fixes it, see the
# [rev] note in lab/probes/cc_llama/PLAN.md). Waits for the probe already running to finish, applies the same gates as
# launch.sh (budget.txt = P2, P3 applied, P6 all Q6_K) plus P4 re-run with the fixed loader, then starts chain.sh.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_llama; LOG=$R/launch.log
log() { echo "[launch2] $* ($(date +%H:%M:%S))" >> $LOG; }
log "waiting for PROBE DONE"
until grep -q "PROBE DONE\|FAILED" $R/probe.log 2>/dev/null; do sleep 60; done
grep -q "PROBE DONE" $R/probe.log || { log "probe FAILED; no chain"; exit 1; }
[ -f $R/budget.txt ] || { log "P2 failed at both budgets: no chain"; exit 0; }
grep -q '"P3_applied": true' $R/probe.log || { log "P3 failed: no chain"; exit 0; }
grep -q '"P6_all_q6k": true' $R/probe.log || { log "P6 failed: no chain"; exit 0; }
log "P4 again with engine/tok.py"; "$P" lab/probes/cc_llama/probe.py offsets >> $R/probe.log 2>&1
grep -q '"P4_offsets_ok": true' $R/probe.log || { log "P4 still failing: no chain"; exit 0; }
log "probe passed (budget $(cat $R/budget.txt)); chain"
bash $R/chain.sh 2>> $R/chain.stderr || log "chain exited $?"
log "LAUNCH DONE"
