#!/usr/bin/env bash
# Gate 3b loose end (lab/probes/act_map/PLAN.md addendum 2026-11-15, prediction row 33): port at 1.25 / 1.5 / 1.75 on the
# gate 3b dev items; reading rule in the plan; a test arm only if a "finish component" is present. Same settings as
# cc08_chain.sh dev arms. set -e + ERR trap, 2-item smoke, control-vector check, vector sha256 logged. One GPU job at a time.
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
O=lab/runs/act_map; IN=$O/cc08; OUT=$O/cc08b; LOG=$O/cc08b_chain.log; C=hf_cache/act_map/cvec08
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
M08=hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf
PORT=8096
SCALES="1.25 1.5 1.75"
SP=""
mkdir -p $OUT
log() { echo "[g3cc-b] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"; stop' ERR
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:$PORT/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null || true; sleep 3
  for w in $(netstat -ano | grep ":$PORT " | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1 || true; done
  sleep 3; SP=""; }
serve() {  # scale
  nohup $S -m $M08 --control-vector-scaled $C/cc_port.gguf:$1 -ngl 99 -c 16384 -np 4 -fa on --jinja --port $PORT > $O/cc08b.server.log 2>&1 & SP=$!
  up || { log "SERVER FAILED scale $1"; stop; exit 1; }
  if grep -q "no valid control vector" $O/cc08b.server.log; then log "control vector rejected at scale $1"; stop; exit 1; fi; }
bench() {  # name items max_tokens [gsm_idx_json]
  local name="$1" items="$2" mt="$3" gi="${4:-}"
  if [ -n "$gi" ]; then IDX="kw=dict(gsm_idx=$gi, math_idx=[])"
  elif [ "$items" = dev ]; then IDX="d=json.load(open('$IN/dev_items.json')); kw=dict(gsm_idx=d['gsm8k'], math_idx=d['math500'])"
  else IDX="kw={}"; fi
  "$P" -c "
import sys, json; sys.path.insert(0,'.'); from engine import bench_runner as b
$IDX
b.run('$name', $PORT, out_dir='$OUT', max_tokens=$mt, parallel=4, log=lambda m: print(m, flush=True), **kw)" >> $LOG 2>&1; }

log "start"
until ! tasklist 2>/dev/null | grep -qi "llama-server"; do log "llama-server alive, waiting"; sleep 120; done
if netstat -ano | grep ":$PORT " | grep -q LISTENING; then log "port $PORT busy: not started"; exit 1; fi
log "vector sha256 $(sha256sum $C/cc_port.gguf | cut -c1-16) dev_items sha256 $(sha256sum $IN/dev_items.json | cut -c1-16)"

first=1
for sc in $SCALES; do
  [ -f $OUT/dev_port_s$sc.json ] && { log "dev port $sc done (marker)"; continue; }
  serve $sc
  if [ $first = 1 ]; then log "smoke: 2 GSM8K items at 256 tokens (scale $sc)"; bench smoke_port 0 256 "[0, 1]"
    "$P" -c "
import json; d=json.load(open('$OUT/smoke_port.json')); assert len(d['items'])==2, d['items']; print('smoke ok', [x['tokens'] for x in d['items']])" >> $LOG 2>&1; first=0; fi
  log "dev port scale $sc"
  bench dev_port_s$sc dev 2048
  stop
done

log "readings"
"$P" -c "
import json, statistics
IN, OUT = '$IN', '$OUT'
rows = {}
for sc, p in (('1.0', IN + '/dev_port_s1.0.json'), ('1.25', OUT + '/dev_port_s1.25.json'), ('1.5', OUT + '/dev_port_s1.5.json'),
              ('1.75', OUT + '/dev_port_s1.75.json'), ('2.0', IN + '/dev_port_s2.0.json'), ('own 1.0', IN + '/dev_own_s1.0.json')):
    items = json.load(open(p, encoding='utf-8'))['items']
    r = {}
    for bench in ('gsm8k', 'math500'):
        g = [x for x in items if x['bench'] == bench]
        fin = [x for x in g if x['finish'] == 'stop']
        r[bench] = {'n': len(g), 'finished': len(fin), 'correct': sum(x['ok'] for x in g), 'correct2': sum(x['ok_independent'] for x in g),
                    'caf': round(sum(x['ok'] for x in fin) / len(fin), 3) if fin else None,
                    'median_think_chars_finished': statistics.median([x['reasoning_chars'] for x in fin]) if fin else None,
                    'mean_tokens': round(sum(x['tokens'] for x in g) / len(g))}
    rows[sc] = r
present = [sc for sc in ('1.25', '1.5', '1.75') if rows[sc]['gsm8k']['finished'] >= 10 and (rows[sc]['gsm8k']['caf'] or 0) >= 0.32]
best = max(present, key=lambda s: rows[s]['gsm8k']['caf']) if present else None
rows['rule'] = {'finish_component_present': bool(present), 'scales_passing': present, 'test_scale': best}
json.dump(rows, open(OUT + '/readings.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(rows))
open(OUT + '/test_scale.txt', 'w').write(best or 'none')" >> $LOG 2>&1
TS=$(cat $OUT/test_scale.txt)
if [ "$TS" != none ] && [ ! -f $OUT/test_port_s$TS.json ]; then
  log "finish component present: test port scale $TS"
  serve $TS; bench test_port_s$TS test 2048; stop
else
  log "no finish component (rule): no test arm"
fi
log "G3CC-B DONE"
