#!/usr/bin/env bash
# dist-shift suite on the 9B (lab/probes/dist_shift/PLAN.md, prediction row 48): base (saves its logits once), the
# compiled conclude edit and its same-norm placebo (merged Q6_K), the conclude vector and its placebo at 1.0, all
# through llama-perplexity on the GPU (suite gpu_args -ngl 99), then bench_compare and the table. No llama-server:
# llama-perplexity loads the GGUF itself, so this chain must not overlap with any server (one GPU job at a time).
# set -eu + ERR trap, sha256 of every GGUF in the log. Resumable (bench_run outputs are skipped when present).
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/bench; LOG=$R/chain_dist9b.log
PP=/c/Users/Anto/llamacpp/bin/llama-perplexity.exe
TEXT=data/bench/wikitext2_test.txt
log() { echo "[dist9b] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"' ERR
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do log "llama process alive, waiting"; sleep 120; done; }

log "start (dist-shift on the 9B)"
[ -f $TEXT ] || { log "missing $TEXT: produce it first (data/bench/README.md), then python -m engine.wsplang_bench write-suites"; exit 1; }
"$P" -c "
import json,sys; s=json.load(open('engine/bench_suites/dist-shift.json',encoding='utf-8'))
missing=[k for k,c in s['components'].items() if not c.get('sha256')]
sys.exit(('suite has components without sha (rewrite the suite): %s' % missing) if missing else 0)" >> $LOG 2>&1 || { log "suite not pinned"; exit 1; }
for f in hf_cache/h4/A0-Q6_K.gguf hf_cache/cc_op_v2/merged-Q6_K.gguf hf_cache/cc_op_v2/placebo_merged-Q6_K.gguf hf_cache/cc/cc_v2.gguf hf_cache/cc/cc_v2.placebo2026.gguf; do
  [ -f $f ] || { log "missing $f"; exit 1; }; log "$f sha256 $(sha256sum $f | cut -c1-16)"; done
log "llama-perplexity sha256 $(sha256sum $PP | cut -c1-16)"
no_server
[ -f $R/dist9b_base.json ] || { log "base (saves logits once)"; "$P" -m engine.wsplang programs/dist9b_base.wspl --run >> $LOG 2>&1; }
for a in merged placebo_merged vec placebo_vec; do [ -f $R/dist9b_$a.json ] && log "dist9b_$a exists"; done
"$P" -m engine.wsplang programs/dist9b_arms.wspl --run >> $LOG 2>&1
"$P" -m engine.wsplang programs/dist9b_compare.wspl --run >> $LOG 2>&1
"$P" -c "
import json
for a in ('merged','placebo_merged','vec','placebo_vec'):
    c=json.load(open('lab/runs/bench/cmp_dist9b_%s.json' % a)); L=c['logits']
    print(a, 'ppl x', round(L['ppl_ratio'],4), 'KL', L['kl_mean'], 'top1', L['same_top_p'], 'delta', L['delta'], c['verdict'])" >> $LOG 2>&1
log "CHAIN DONE"
