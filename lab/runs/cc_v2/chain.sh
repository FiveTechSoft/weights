#!/usr/bin/env bash
# Repeat of results 43.17-43.18 with the corrected contrast_vector op (lab/probes/cc/PLAN.md, addendum 2026-11-12).
set -u
cd /c/weights
export PYTHONUTF8=1
export CC_RUNS=lab/runs/cc_v2
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; OLD=lab/runs/cc; SOUT=lab/runs/sealed_cc_v2; LOG=$OUT/chain.log
V=hf_cache/cc/cc_v2.gguf; PL=hf_cache/cc/cc_v2.placebo2026.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; M=hf_cache/h4/A0-Q6_K.gguf
log() { echo "[cc_v2] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
SP=""
serve() { nohup $S -m $M "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port 8095 > $OUT/server.log 2>&1 & SP=$!; up || { log "SERVER FAILED"; stop; exit 1; }; }
stop() {  # own server only: the Windows PID listening on 8095
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3
  for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done
  sleep 3; SP=""; }
bench() {  # name out_dir setup kwargs
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b, sealed_set as ss
$3
b.run('$1', 8095, out_dir='$2', max_tokens=2048, log=lambda m: print(m, flush=True)$4)" >> $LOG 2>&1 || log "ARM FAILED $1"; }

if [ ! -f $V ]; then
  log "op contrast_vector (corrected)"
  "$P" -m engine.wsplang programs/cc_vector_v2.wspl --run >> $LOG 2>&1 || { log "OP FAILED"; exit 1; }
fi
log "mechanical check: direction.1..30 unchanged"
"$P" - >> $LOG 2>&1 <<'PY' || { log "CHECK FAILED"; exit 1; }
import json, numpy as np, gguf
def rd(p):
    r = gguf.GGUFReader(p)
    return {int(t.name.split(".")[1]): np.array(t.data, dtype=np.float64) for t in r.tensors}
a, b = rd("hf_cache/cc/cc.gguf"), rd("hf_cache/cc/cc_v2.gguf")
assert sorted(a) == sorted(b), (sorted(a), sorted(b))
last = max(a)
rel = {k: float(np.abs(a[k] - b[k]).max() / max(np.linalg.norm(a[k]), 1e-30)) for k in a if k != last}
out = {"max_rel_diff_1_to_%d" % (last - 1): max(rel.values()), "norm_last_old": float(np.linalg.norm(a[last])),
       "norm_last_new": float(np.linalg.norm(b[last])), "norm_prev": float(np.linalg.norm(a[last - 1])),
       "cos_last_old_new": float(a[last] @ b[last] / (np.linalg.norm(a[last]) * np.linalg.norm(b[last])))}
json.dump(out, open("lab/runs/cc_v2/op_check.json", "w"), indent=1)
print(json.dumps(out))
assert out["max_rel_diff_1_to_%d" % (last - 1)] < 1e-3, "layers other than the last changed"
PY

SCR="s=json.load(open('$OUT/split.json'))"
if [ ! -f $OUT/screen_base.json ]; then log "screen base (identity check)"; serve; bench screen_base $OUT "$SCR" ", gsm_idx=[], math_idx=s['screen_math']"; stop; fi
SAME=$("$P" -c "
import json
L=lambda p:{(r['bench'],r['i']):(r['ok'],r['pred'],r['tokens'],r['tail']) for r in json.load(open(p,encoding='utf-8'))['items']}
a,b=L('$OLD/screen_base.json'),L('$OUT/screen_base.json'); print(int(a==b))")
log "screen_base identical to 43.17: $SAME"
for sc in 0.25 0.5 1.0 2.0; do
  [ -f $OUT/screen_s$sc.json ] && continue
  log "screen scale $sc"; serve --control-vector-scaled $V:$sc
  bench screen_s$sc $OUT "$SCR" ", gsm_idx=[], math_idx=s['screen_math']"; stop
done
"$P" lab/probes/cc/cc.py choose >> $LOG 2>&1
SC=$("$P" -c "import json;print(json.load(open('$OUT/scale.json'))['scale'])")
if [ "$SC" = "None" ] || [ -z "$SC" ]; then log "STOP: no scale beats base on the screen -> FAIL"; log "CHAIN DONE"; exit 0; fi
log "dev at scale $SC"
if [ "$SAME" = 1 ]; then cp $OLD/dev_base.json $OUT/dev_base.json; log "dev_base reused";
else [ -f $OUT/dev_base.json ] || { serve; bench dev_base $OUT "" ""; stop; }; fi
[ -f $OUT/dev_vec.json ] || { serve --control-vector-scaled $V:$SC; bench dev_vec $OUT "" ""; stop; }
[ -f $OUT/dev_placebo.json ] || { serve --control-vector-scaled $PL:$SC; bench dev_placebo $OUT "" ""; stop; }
"$P" lab/probes/cc/cc.py verdict >> $LOG 2>&1
DV=$("$P" -c "import json;print(json.load(open('$OUT/verdict.json'))['verdict'])")
log "dev verdict $DV"
if [ "$DV" != PASS ]; then log "dev FAIL: no sealed run"; log "CHAIN DONE"; exit 0; fi

log "sealed confirmation"
mkdir -p $SOUT
"$P" -m engine.sealed_set check >> $LOG 2>&1 || { log "SEAL CHECK FAILED"; exit 1; }
SS="s=ss.load()"; SK=", gsm_idx=s['gsm8k'], math_idx=s['math500']"
if [ "$SAME" = 1 ]; then cp lab/runs/sealed_cc/base.json $SOUT/base.json; log "sealed base reused";
else [ -f $SOUT/base.json ] || { serve; bench base $SOUT "$SS" "$SK"; stop; }; fi
[ -f $SOUT/vec.json ] || { serve --control-vector-scaled $V:$SC; bench vec $SOUT "$SS" "$SK"; stop; }
[ -f $SOUT/placebo.json ] || { serve --control-vector-scaled $PL:$SC; bench placebo $SOUT "$SS" "$SK"; stop; }
SEALED_CC_RUNS=$SOUT "$P" lab/probes/sealed/cc_verdict.py >> $LOG 2>&1
log "CHAIN DONE"
