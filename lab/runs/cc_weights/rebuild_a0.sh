#!/bin/bash
# Rebuild hf_cache/h4/A0-Q6_K.gguf (deleted with h4 on 2026-11-11) with the recipe of lab/probes/model_compare/own_q6k.sh,
# check it byte for byte against every tensor of cc_merged-Q6_K.gguf that the cc edit did not touch, then (after the
# H6h chain) run the compile_control_vector acceptance.
cd /c/weights
export PYTHONUTF8=1 GIT_TERMINAL_PROMPT=0
P=$HOME/.venvs/weights-cuda/Scripts/python.exe
CONV_PY=$HOME/.venvs/llama-convert/Scripts/python.exe
LLAMA=/c/Users/Anto/llamacpp
LOG=lab/runs/cc_weights/rebuild_a0.log
log() { echo "[a0] $* ($(date +%H:%M:%S))" >> $LOG; }
mkdir -p hf_cache/h4
log "convert"
"$CONV_PY" $LLAMA/convert-src/convert_hf_to_gguf.py hf_cache/qwen35-9b-hf --outtype bf16 --outfile hf_cache/h4/A0-bf16.gguf >> $LOG 2>&1 || { log "CONVERT FAILED"; exit 1; }
log "quantize"
$LLAMA/bin/llama-quantize.exe hf_cache/h4/A0-bf16.gguf hf_cache/h4/A0-Q6_K.gguf Q6_K >> $LOG 2>&1 || { log "QUANTIZE FAILED"; exit 1; }
rm -f hf_cache/h4/A0-bf16.gguf
log "compare"
"$P" - >> $LOG 2>&1 <<'PY'
import gc, json, re, numpy as np, gguf
a = gguf.GGUFReader("hf_cache/h4/A0-Q6_K.gguf"); b = gguf.GGUFReader("hf_cache/cc_weights/cc_merged-Q6_K.gguf")
ta = {t.name: t for t in a.tensors}; tb = {t.name: t for t in b.tensors}
edited = {f"blk.{l}.ffn_down.weight" for l in range(1, 32)}
same = diff = 0; bad = []
for n, t in ta.items():
    if n in edited: continue
    u = tb.get(n)
    if u is not None and u.tensor_type == t.tensor_type and np.array_equal(np.asarray(t.data), np.asarray(u.data)): same += 1
    else: diff += 1; bad.append(n)
res = {"tensors_compared": same + diff, "identical": same, "different": bad[:10], "names_equal": set(ta) == set(tb)}
print("A0_CHECK", json.dumps(res))
open("lab/runs/cc_weights/a0_rebuild_check.json", "w").write(json.dumps(res, indent=1))
PY
grep -q '"different": \[\]' lab/runs/cc_weights/a0_rebuild_check.json && log "A0 REBUILD IDENTICAL" || { log "A0 REBUILD DIFFERS: stop"; exit 1; }
log "waiting for h6h2 CHAIN DONE"
until grep -q "^\[h6h2\] CHAIN DONE" lab/runs/h6/h6h_chain2.log 2>/dev/null; do sleep 60; done
log "op repro start"
"$P" -m engine.wsplang programs/cc_compile_repro.wspl --run >> $LOG 2>&1 || log "OP REPRO FAILED"
"$P" lab/probes/cc_weights/op_accept.py >> $LOG 2>&1 || log "OP ACCEPT FAILED"
log "CHAIN DONE"
