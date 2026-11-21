#!/usr/bin/env bash
# Repeat of results 43.13 with the corrected contrast_vector op (lab/probes/happy/PLAN.md, "Repeat with the corrected op").
set -u
cd /c/weights
export PYTHONUTF8=1
export HAPPY_RUNS=/c/weights/lab/runs/happy_v2
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"; H=lab/probes/happy/happy.py; R=lab/runs/happy_v2; LOG=$R/chain.log
S=/c/Users/Anto/llamacpp/bin/llama-server.exe; BASE=hf_cache/h4/A0-Q6_K.gguf
V=hf_cache/cvec/happy_v2.gguf; PL=hf_cache/cvec/happy_v2.placebo2026.gguf; PORT=8095
log() { echo "[happy_v2] $* ($(date +%H:%M:%S))" >> $LOG; }
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
serve() { nohup $S -m "$@" -ngl 99 -c 16384 -np 4 -fa on --jinja --port $PORT > $R/server.log 2>&1 & SP=$!; up || { log "SERVER FAILED $*"; kill $SP; exit 1; }; }
stop() { kill $SP; sleep 5; }

log "op contrast_vector (corrected)"
[ -f $V ] || "$P" -m engine.wsplang programs/happy_vector_v2.wspl --run >> $LOG 2>&1 || { log "OP FAILED"; exit 1; }
log "mechanical check: direction.1..30 unchanged"
"$P" - >> $LOG 2>&1 <<'PY' || { log "CHECK FAILED"; exit 1; }
import json, numpy as np, gguf
def rd(p):
    r = gguf.GGUFReader(p)
    return {int(t.name.split(".")[1]): np.array(t.data, dtype=np.float64) for t in r.tensors}
a, b = rd("hf_cache/cvec/happy.gguf"), rd("hf_cache/cvec/happy_v2.gguf")
assert sorted(a) == sorted(b), (sorted(a), sorted(b))
last = max(a)
rel = {k: float(np.abs(a[k] - b[k]).max() / max(np.linalg.norm(a[k]), 1e-30)) for k in a if k != last}
out = {"max_rel_diff_1_to_%d" % (last - 1): max(rel.values()), "norm_last_old": float(np.linalg.norm(a[last])),
       "norm_last_new": float(np.linalg.norm(b[last])), "cos_last_old_new": float(a[last] @ b[last] /
       (np.linalg.norm(a[last]) * np.linalg.norm(b[last])))}
json.dump(out, open("lab/runs/happy_v2/op_check.json", "w"), indent=1)
print(json.dumps(out))
assert out["max_rel_diff_1_to_%d" % (last - 1)] < 1e-3, "layers other than the last changed"
PY

log "base dev + test"
serve $BASE
"$P" $H gen $PORT dev dev_base >> $LOG 2>&1
"$P" $H gen $PORT test test_base >> $LOG 2>&1; stop
"$P" -c "
import json
a=json.load(open('lab/runs/happy/test_base.json',encoding='utf-8'))['rows']; b=json.load(open('lab/runs/happy_v2/test_base.json',encoding='utf-8'))['rows']
same=sum(x['response']==y['response'] for x,y in zip(a,b)); print('test_base identical to 43.13:', same, '/', len(a))" >> $LOG 2>&1
NAMES="dev_base"
for sc in 0.125 0.25 0.5 1.0 2.0 4.0; do
  serve $BASE --control-vector-scaled $V:$sc
  "$P" $H gen $PORT dev dev_s$sc >> $LOG 2>&1; stop
  NAMES="$NAMES dev_s$sc"
done
log "judge dev"
serve hf_cache/own_q6k/mimo-Q6_K.gguf
"$P" $H judge $PORT $NAMES >> $LOG 2>&1; stop
BOK=$("$P" -c "import json;print(sum(bool(r['coherent']) for r in json.load(open('lab/runs/happy_v2/dev_base.json',encoding='utf-8'))['rows']))")
log "base dev coherent $BOK/10"
if ! [ "${BOK:-0}" -ge 9 ] 2>/dev/null; then log "judge not valid: stop"; log "CHAIN DONE"; exit 0; fi
"$P" $H choose >> $LOG 2>&1
SC=$("$P" -c "import json;print(json.load(open('lab/runs/happy_v2/scale.json'))['scale'])")
log "scale $SC"
if [ "$SC" = "None" ] || [ -z "$SC" ]; then log "NO COHERENT SCALE -> FAIL"; log "CHAIN DONE"; exit 0; fi
log "test at scale $SC"
serve $BASE --control-vector-scaled $V:$SC
"$P" $H gen $PORT test test_vec >> $LOG 2>&1
"$P" lab/probes/close_probe_run/fast_eval.py gsm_happy_v2_s$SC $PORT --bench gsm8k --guard >> $LOG 2>&1; stop
serve $BASE --control-vector-scaled $PL:$SC
"$P" $H gen $PORT test test_placebo >> $LOG 2>&1; stop
"$P" -c "
import json
d=json.load(open('lab/runs/close_probe/fast/gsm_happy_v2_s$SC.json',encoding='utf-8'))['items']
json.dump({'base':9,'vec':sum(1 for r in d if r['ok']),'n':len(d),'file':'gsm_happy_v2_s$SC.json'},open('lab/runs/happy_v2/gsm.json','w'))" >> $LOG 2>&1
log "judge test"
serve hf_cache/own_q6k/mimo-Q6_K.gguf
"$P" $H judge $PORT test_base test_vec test_placebo >> $LOG 2>&1; stop
"$P" $H verdict >> $LOG 2>&1
log "CHAIN DONE"
