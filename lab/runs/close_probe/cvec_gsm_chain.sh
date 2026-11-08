#!/usr/bin/env bash
set -u
cd /c/weights
P="$HOME/.venvs/weights-cuda/Scripts/python.exe"; F=lab/probes/close_probe_run/fast_eval.py; LOG=lab/runs/close_probe/fast/chain2.log
serve() { taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 3
  nohup /c/Users/Anto/llamacpp/bin/llama-server.exe -m hf_cache/h4/A0-Q6_K.gguf "$@" --lora hf_cache/close_lora/jack_r16.gguf --lora-init-without-apply -ngl 99 -c 32768 -np 4 -fa on --jinja --port 8095 > lab/runs/close_probe/fast_chain2.server.log 2>&1 &
  for k in $(seq 1 60); do curl -s http://127.0.0.1:8095/health | grep -q ok && return 0; sleep 3; done; echo "server failed" >> $LOG; exit 1; }
for s in 1.0 0.5; do
  serve --control-vector-scaled hf_cache/close_lora/cvec_mimo_inc.gguf:$s
  "$P" $F cvec_inc_s${s/./} 8095 --guard --lora 0:0 >> $LOG 2>&1
done
serve
"$P" $F gsm_base 8095 --bench gsm8k --guard --lora 0:0 >> $LOG 2>&1
"$P" $F gsm_jack_r16_s05 8095 --bench gsm8k --guard --lora 0:0.5 >> $LOG 2>&1
echo CHAIN2 DONE >> $LOG
