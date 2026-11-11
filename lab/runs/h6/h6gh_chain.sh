#!/bin/bash
# H6g BP ceilings + gates + smoke, then H6h smoke + seed-1 arms. Waits for the cc_neurons rerun (one GPU job at a time).
cd /c/weights
P=$HOME/.venvs/weights-cuda/Scripts/python.exe
LOG=lab/runs/h6/h6gh_chain.log
G=lab/probes/h6/h6g.py; H=lab/probes/h6/h6h.py
log() { echo "[h6gh] $* ($(date +%H:%M:%S))" >> $LOG; }
log "waiting for cc_neurons rerun"
until grep -q "^EXIT" lab/runs/cc_neurons/rerun.log 2>/dev/null; do sleep 60; done
log "start"
"$P" $G run --seed 1 --basis random --r 1 --only mlp.down_proj --method bp --lr 1e-5 --budget 1 >> $LOG 2>&1 || log "BP1 FAILED"
"$P" $G run --seed 1 --basis random --r 1 --method bp --lr 1e-5 --budget 1 >> $LOG 2>&1 || log "BP2 FAILED"
for arm in "1 mlp.down_proj" "1 all" "20 all"; do
  set -- $arm; r=$1; only=$2; ONLY=""; [ "$only" != "all" ] && ONLY="--only $only"
  "$P" $G gate --seed 1 --basis random --r $r $ONLY >> $LOG 2>&1 || log "GATE FAILED r$r $only bf16"
  f=lab/runs/h6/h6g/gate_random_r${r}_${only}_bf16_seed1.json
  e=$("$P" -c "import json;print(json.load(open('$f'))['eps_chosen'])" 2>/dev/null)
  if [ "$e" = "None" ] || [ -z "$e" ]; then
    log "gate r$r $only: bf16 none -> fp32"
    "$P" $G gate --seed 1 --basis random --r $r $ONLY --dtype fp32 >> $LOG 2>&1 || log "GATE FAILED r$r $only fp32"
  fi
  log "gate r$r $only done"
done
# H6g smoke on Z1 with its gate eps (bf16 if it passed, else fp32)
for dt in bf16 fp32; do
  f=lab/runs/h6/h6g/gate_random_r1_mlp.down_proj_${dt}_seed1.json
  e=$("$P" -c "import json;print(json.load(open('$f'))['eps_chosen'])" 2>/dev/null)
  if [ -n "$e" ] && [ "$e" != "None" ]; then
    "$P" $G run --seed 1 --basis random --r 1 --only mlp.down_proj --dtype $dt --eps $e --lr 1e-5 --budget 0.05 --probe >> $LOG 2>&1 || log "H6g SMOKE FAILED"
    log "h6g smoke done ($dt eps $e)"; break
  fi
done
"$P" $H run --arm W_all --seed 1 --rounds 1 --k 20 --smoke >> $LOG 2>&1 || { log "H6h SMOKE FAILED: stop"; log "CHAIN DONE"; exit 1; }
log "h6h smoke ok"
for arm in W_all G_all W_down; do
  "$P" $H run --arm $arm --seed 1 >> $LOG 2>&1 || log "H6h $arm FAILED"
  log "h6h $arm done"
done
"$P" $H verdict >> $LOG 2>&1
log "CHAIN DONE"
