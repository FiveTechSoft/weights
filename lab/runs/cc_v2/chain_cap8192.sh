#!/usr/bin/env bash
# Token-budget control (PLAN.md "the base at 8,192", prediction row 31): base / vec / merged at 8,192 tokens
# on GSM8K dev 100. Waits for the placebo chain (row 29) and the gate-3b chain (row 30). One GPU job at a time.
# Idempotent per arm. trap EXIT writes a marker so a silent death is visible in the log.
set -u
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
OUT=lab/runs/cc_v2; LOG=$OUT/chain_cap8192.log; D=$OUT/cap8192
V=hf_cache/cc/cc_v2.gguf; HM=hf_cache/cc_op_v2/merged-Q6_K.gguf
S=/c/Users/Anto/llamacpp/bin/llama-server.exe
log() { echo "[cap8192] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "CHAIN EXITED code=$? (if not DONE, the chain died)"' EXIT
up() { ok=0; for k in $(seq 1 80); do curl -s http://127.0.0.1:8095/health | grep -q ok && { ok=1; break; }; sleep 3; done; [ $ok = 1 ]; }
serve() { local m=$1 cv=$2; extra=(); [ -n "$cv" ] && extra=(--control-vector-scaled "$cv:1.0")
  nohup $S -m "$m" "${extra[@]}" -ngl 99 -c 40960 -np 2 -fa on --jinja --port 8095 > $OUT/server_cap.log 2>&1 & SP=$!; up || { log "SERVER FAILED $m"; stop; exit 1; }; }
stop() { [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 3; for w in $(netstat -ano | grep ':8095 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //F //PID $w > /dev/null 2>&1; done; sleep 3; SP=""; }
bench() { local name="$1" sealed="${2:-}"
  "$P" -c "
import sys; sys.path.insert(0,'.'); from engine import bench_runner as b
b.run('$name', 8095, out_dir='$D', max_tokens=8192, log=lambda m: print(m, flush=True))" >> $LOG 2>&1 || log "ARM FAILED $name"; }

log "start"
until grep -q "PLACEBO DONE" $OUT/chain.log 2>/dev/null; do log "waiting placebo chain (row 29)"; sleep 600; done
# gate-3b chain is not launched yet; the GPU-free re-check before every arm keeps us from colliding with it

mkdir -p $D
for arm in "base|" "vec|$V" "merged|$HM"; do
  IFS='|' read -r name cv <<< "$arm"
  [ -f $D/dev_${name}_8192.json ] && { log "$name done (marker)"; continue; }
  while pgrep -x llama-server >/dev/null 2>&1; do log "gpu busy, waiting"; sleep 300; done
  log "$name at 8192"; serve hf_cache/h4/A0-Q6_K.gguf "$cv"
  bench dev_${name}_8192
  stop
done
log "readings"
"$P" -c "
import json
from pathlib import Path
D = Path('$D')
rows = {}
for a in ('base','vec','merged'):
    p = D / f'dev_{a}_8192.json'
    if not p.exists():
        continue
    items = json.loads(p.read_text(encoding='utf-8'))['items']
    g = [x for x in items if x.get('bench')=='gsm8k']
    rows[a] = {'ok': sum(x['ok'] for x in g), 'n': len(g),
               'cuts': sum(x.get('finish')!='stop' for x in g),
               'finished_wrong': sum(x.get('finish')=='stop' and not x['ok'] for x in g),
               'mean_tokens': round(sum(x.get('tokens',0) for x in g)/max(1,len(g)))}
th = json.loads((D/'thirteen.json').read_text(encoding='utf-8')) if (D/'thirteen.json').exists() else {}
v = json.loads((D/'dev_vec_8192.json').read_text(encoding='utf-8'))['items'] if (D/'dev_vec_8192.json').exists() else []
keyed = {(x.get('bench'),x.get('i')):x for x in v}
early = [r for r in th.get('early_wrong',[])]
rows['early_close_recheck'] = {str(r['i']): {'vec8192_ok': keyed.get(('gsm8k',r['i']),{}).get('ok')} for r in early}
(D/'readings.json').write_text(json.dumps(rows,indent=1),encoding='utf-8')
print(json.dumps(rows))" >> $LOG 2>&1
log "CHAIN DONE"
