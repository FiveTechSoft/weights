#!/usr/bin/env bash
# Stage-0 premise probe for lab/probes/cc_llama/PLAN.md (DeepSeek-R1-Distill-Llama-8B): P1 reasoning parsed, P2 base
# over-thinks (writes budget.txt by the plan's rule: >= 3/10 cut at 2048 -> 2048, else >= 3/10 at 1024 -> 1024, else no
# file = STOP), P3 control vectors applied to `llama`, P4 offsets, P5 template writes <think>, P6 GGUF checks.
# Nothing else decides here. set -eu + ERR trap. One GPU job at a time.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/cc_llama; LOG=$R/probe.log; PR=lab/probes/cc_llama/probe.py
G=hf_cache/r1llama8b-gguf/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
PORT=8097
SP=""
mkdir -p $R
log() { echo "[llama-probe] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 100); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # [extra args]
  nohup $S -m $G "$@" -ngl 99 -c 16384 -np 2 -fa on --jinja --port $PORT > $R/probe.server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED $*"; stop; exit 1; }; }

log "start"
[ -f $G ] || { log "no GGUF at $G"; exit 1; }
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 60; done
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
log "gguf sha256 $(sha256sum $G | cut -c1-16)"

# P4, P5, P6 on the CPU first (no server needed)
log "P4 offsets"; "$P" $PR offsets >> $LOG 2>&1
log "P5 template"; "$P" lab/probes/cc_llama/rows.py template >> $LOG 2>&1
log "P6 gguf"; "$P" -c "
import gguf, json, collections
r = gguf.GGUFReader('$G')
def g(k):
    f = r.fields.get(k)
    if not f: return None
    try: return bytes(f.parts[f.data[0]]).decode()
    except Exception: return int(f.parts[f.data[0]][0])
types = collections.Counter(t.tensor_type.name for t in r.tensors if t.name.endswith('ffn_down.weight'))
blk = len({t.name.split('.')[1] for t in r.tensors if t.name.startswith('blk.')})
print(json.dumps({'name': g('general.name'), 'basename': g('general.basename'), 'arch': g('general.architecture'), 'blocks': blk,
                  'ffn_down_types': dict(types), 'P6_all_q6k': set(types) == {'Q6_K'} and blk == 32}))" >> $LOG 2>&1

serve
log "P1/P2: 5 GSM8K + 5 MATH dev items at 2048"
"$P" $PR items $PORT 2048 >> $LOG 2>&1
CUT=$("$P" -c "
import json; d=json.load(open('$R/probe_2048.json')); print(sum(1 for x in d['items'] if x['finish']!='stop'))")
log "cut at 2048: $CUT of 10"
if [ "$CUT" -ge 3 ]; then echo 2048 > $R/budget.txt; log "budget 2048 (plan rule)"
else
  log "P2 at 2048 fails: repeating at 1024"; "$P" $PR items $PORT 1024 >> $LOG 2>&1
  CUT1=$("$P" -c "
import json; d=json.load(open('$R/probe_1024.json')); print(sum(1 for x in d['items'] if x['finish']!='stop'))")
  log "cut at 1024: $CUT1 of 10"
  if [ "$CUT1" -ge 3 ]; then echo 1024 > $R/budget.txt; log "budget 1024 (plan rule)"; else log "P2 fails at both budgets: STOP (no budget.txt)"; fi
fi
log "P3 base outputs at 512"
"$P" $PR items $PORT 512 _base >> $LOG 2>&1
stop

"$P" $PR randvec $R/randvec.gguf >> $LOG 2>&1
log "P3: same prompts with a random control vector, scale 8"
serve --control-vector-scaled $R/randvec.gguf:8
if grep -q "no valid control vector" $R/probe.server.log; then log "control vector rejected"; stop; exit 1; fi
"$P" $PR items $PORT 512 _vec >> $LOG 2>&1
stop
"$P" $PR compare $R/probe_512_base.json $R/probe_512_vec.json >> $LOG 2>&1
log "PROBE DONE"
