#!/bin/bash
cd /c/weights
L=lab/runs/model_compare/jackrong_chain.log
bash lab/probes/model_compare/own_q6k.sh jackrong hf_cache/dp_keep_dsv4_distill 100 50 > lab/runs/model_compare/own_jackrong.log 2>&1; echo "jackrong rc=$? $(date +%H:%M:%S)" >> $L
bash lab/probes/model_compare/own_q6k.sh mimo hf_cache/mimo-distill-hf 100 50 > lab/runs/model_compare/own_mimo.log 2>&1; echo "mimo rc=$? $(date +%H:%M:%S)" >> $L
bash lab/probes/model_compare/own_q6k.sh base hf_cache/qwen35-9b-hf 0 50 > lab/runs/model_compare/own_base.log 2>&1; echo "base rc=$? $(date +%H:%M:%S)" >> $L
echo "ALL DONE $(date +%H:%M:%S)" >> $L
