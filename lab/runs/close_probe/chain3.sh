#!/usr/bin/env bash
set -u
cd /c/weights
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"; F=lab/probes/close_probe_run/fast_eval.py; LOG=lab/runs/close_probe/fast/chain3.log
serve() { M=$1; shift; taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 3
  nohup /c/Users/Anto/llamacpp/bin/llama-server.exe -m $M "$@" --lora-init-without-apply -ngl 99 -c 32768 -np 4 -fa on --jinja --port 8095 > lab/runs/close_probe/fast_chain3.server.log 2>&1 &
  for k in $(seq 1 60); do curl -s http://127.0.0.1:8095/health | grep -q ok && return 0; sleep 3; done; echo "server failed" >> $LOG; exit 1; }
L=hf_cache/close_lora
serve hf_cache/h4/A0-Q6_K.gguf --lora $L/jack_r16.gguf --lora $L/heretic_r4.gguf
"$P" $F math_base_jack05_her1 8095 --guard --lora 0:0.5,1:1 >> $LOG 2>&1
"$P" $F gsm_base_jack05_her1 8095 --bench gsm8k --guard --lora 0:0.5,1:1 >> $LOG 2>&1
serve hf_cache/own_q6k/mimo-Q6_K.gguf --lora $L/jack_r16.gguf
"$P" $F math_mimo 8095 --guard --lora 0:0 >> $LOG 2>&1
"$P" $F math_mimo_jack05 8095 --guard --lora 0:0.5 >> $LOG 2>&1
echo CHAIN3 DONE >> $LOG
