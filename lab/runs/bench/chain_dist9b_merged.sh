#!/usr/bin/env bash
# The dist-shift arm that failed on 2026-11-18 (prediction row 48): the compiled conclude edit (merged Q6_K) with its
# compile report as provenance. The first chain died in read_provenance ("'list' object has no attribute 'get'": the
# report lists its outputs as a plain list of paths); fixed with a test in tests/test_wsplang_bench.py. Runs the one
# bench_run through llama-perplexity (GPU, no server may be up), then the full dist9b_compare program (idempotent
# registry: the three comparisons already registered are replaced by identical ones) and the summary line.
# set -eu + ERR trap, sha256 of the GGUFs in the log. Resumable (the bench_run output is skipped when present).
set -eu
cd /c/weights
export PYTHONUTF8=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
R=lab/runs/bench; LOG=$R/chain_dist9b_merged.log
PP=/c/Users/Anto/llamacpp/bin/llama-perplexity.exe
log() { echo "[dist9b_merged] $* ($(date +%H:%M:%S))" >> $LOG; }
trap 'log "FAILED at line $LINENO (exit $?)"' ERR
no_server() { until ! tasklist 2>/dev/null | grep -qi "llama-server\|llama-perplexity"; do log "llama process alive, waiting"; sleep 120; done; }

log "start (dist-shift: merged arm)"
for f in hf_cache/h4/A0-Q6_K.gguf hf_cache/cc_op_v2/merged-Q6_K.gguf hf_cache/cc_op_v2/report.json $R/dist9b_base.json; do
  [ -f $f ] || { log "missing $f"; exit 1; }; done
log "hf_cache/cc_op_v2/merged-Q6_K.gguf sha256 $(sha256sum hf_cache/cc_op_v2/merged-Q6_K.gguf | cut -c1-16)"
log "llama-perplexity sha256 $(sha256sum $PP | cut -c1-16)"
"$P" -m unittest tests.test_wsplang_bench -k compile_report >> $LOG 2>&1 || { log "provenance test fails; not running"; exit 1; }
no_server
[ -f $R/dist9b_merged.json ] && log "dist9b_merged exists" || "$P" -m engine.wsplang programs/dist9b_merged.wspl --run >> $LOG 2>&1
[ -f $R/dist9b_merged.json ] || { log "merged arm produced no output"; exit 1; }
"$P" -m engine.wsplang programs/dist9b_compare.wspl --run >> $LOG 2>&1
"$P" -c "
import json
for a in ('merged','placebo_merged','vec','placebo_vec'):
    c=json.load(open('lab/runs/bench/cmp_dist9b_%s.json' % a)); L=c['logits']
    print(a, 'ppl x', round(L['ppl_ratio'],4), 'KL', L['kl_mean'], 'top1', L['same_top_p'], 'delta', L['delta'], c['verdict'])" >> $LOG 2>&1
log "CHAIN DONE"
