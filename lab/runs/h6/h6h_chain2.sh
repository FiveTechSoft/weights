#!/bin/bash
# After the act_map chain: H6h smoke (device fix), compile_control_vector acceptance run, then H6h seed-1 arms.
cd /c/weights
P=$HOME/.venvs/weights-cuda/Scripts/python.exe
LOG=lab/runs/h6/h6h_chain2.log
H=lab/probes/h6/h6h.py
log() { echo "[h6h2] $* ($(date +%H:%M:%S))" >> $LOG; }
log "waiting for act_map CHAIN DONE"
until grep -q "^\[am\] CHAIN DONE" lab/runs/act_map/chain.log 2>/dev/null; do sleep 60; done
log "start"
"$P" $H run --arm W_all --seed 1 --rounds 1 --k 20 --smoke >> $LOG 2>&1 || { log "H6h SMOKE FAILED: stop"; log "CHAIN DONE"; exit 1; }
log "h6h smoke ok"
FREE=$(df -BG --output=avail /c | tail -1 | tr -dc 0-9)
if [ "$FREE" -ge 17 ]; then
  log "op repro start (free ${FREE}G)"
  "$P" -m engine.wsplang programs/cc_compile_repro.wspl --run >> $LOG 2>&1 || log "OP REPRO FAILED"
  "$P" lab/probes/cc_weights/op_accept.py >> $LOG 2>&1 || log "OP ACCEPT FAILED"
  log "op repro done"
else
  log "op repro SKIPPED: only ${FREE}G free (needs ~15G)"
fi
for arm in W_all G_all W_down; do
  "$P" $H run --arm $arm --seed 1 >> $LOG 2>&1 || log "H6h $arm FAILED"
  log "h6h $arm done"
done
"$P" $H verdict >> $LOG 2>&1
log "CHAIN DONE"
