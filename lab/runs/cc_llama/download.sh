#!/usr/bin/env bash
# Third family: DeepSeek-R1-Distill-Llama-8B. HF bf16 (deepseek-ai, 2 shards, ~16 GB) + GGUF Q6_K (unsloth, ~6.6 GB).
# Approved by the user 2026-11-16 ("adelante" to the sizes stated). Nothing else is written. Resumable (hf_hub downloads).
set -eu
cd /c/weights
export PYTHONUTF8=1 HF_HUB_DISABLE_TELEMETRY=1
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"
LOG=lab/runs/cc_llama/download.log
mkdir -p hf_cache/r1llama8b-hf hf_cache/r1llama8b-gguf
log() { echo "[dl] $* ($(date +%H:%M:%S))" >> $LOG; }
log "start"
"$P" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', local_dir='hf_cache/r1llama8b-hf', revision='6a6f4aa4197940add57724a7707d069478df56b1',
                      allow_patterns=['*.json', '*.safetensors', 'LICENSE', 'README.md'])
print('HF_DIR', p)" >> $LOG 2>&1
log "HF_EXIT $?"
"$P" -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF', 'DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf', local_dir='hf_cache/r1llama8b-gguf',
                    revision='615f8936e16dfde29dcc00be71145d4d5ce8ed53')
print('GGUF', p)" >> $LOG 2>&1
log "GGUF_EXIT $?"
sha256sum hf_cache/r1llama8b-gguf/DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf >> $LOG
du -sh hf_cache/r1llama8b-hf hf_cache/r1llama8b-gguf >> $LOG
log "DOWNLOAD DONE"
