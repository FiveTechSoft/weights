"""capture_activations.py — E5: score de neuronas intermediate por |mean(rep) - mean(pro)|
con forward hooks en down_proj de las capas objetivo (mismos 4 mini contextos de E2)."""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
P = r"C:\weights\hf_cache\mimo-distill-hf"
OUT = r"C:\weights\data\act_scores_e5.json"
TARGET_LAYERS = [17, 19]

sys.path.insert(0, r"C:\weights\blog_prompts")
import capture_residual as CR

tok = AutoTokenizer.from_pretrained(P, local_files_only=True)
tok.chat_template = open(r"C:\weights\data\chat_template.jinja", encoding="utf-8").read()
model = AutoModelForCausalLM.from_pretrained(P, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
model.eval()


def fwd_capture(msgs, layers):
    acts = {}
    hooks = []
    for idx in layers:
        mod = model.model.layers[idx].mlp.down_proj

        def make_hk(idx=idx):
            def hk(module, inp, out):
                acts[idx] = inp[0].detach().float().mean(axis=(0, 1)).numpy()
            return hk

        hooks.append(mod.register_forward_hook(make_hk(idx)))
    bt = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True, return_dict=True)
    ids_i = bt["input_ids"]
    if isinstance(ids_i, (list, tuple)) and len(ids_i) and not isinstance(ids_i[0], (list, tuple, int, float)):
        ids_i = ids_i[0]
    input_ids = torch.tensor([list(ids_i)], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    for h in hooks:
        h.remove()
    print(f"fwd {time.time() - t0:.0f}s acts={ {k: v.shape for k, v in acts.items()} }")
    return {k: acts[k].copy() for k in acts}


reps = {l: [] for l in TARGET_LAYERS}
pros = {l: [] for l in TARGET_LAYERS}
for kind in ("rep", "pro"):
    for i in (1, 2):
        pid = CR.PROMPT_REP if kind == "rep" else CR.PROMPT_PRO
        acts = fwd_capture(CR.build_msgs(kind, pid), TARGET_LAYERS)
        for l in TARGET_LAYERS:
            (reps if kind == "rep" else pros)[l].append(acts[l])

scores = {}
for l in TARGET_LAYERS:
    mr = np.stack(reps[l]).mean(axis=0)
    mp = np.stack(pros[l]).mean(axis=0)
    diff = np.abs(mr - mp)
    norm = diff / (diff.max() + 1e-30)
    top128 = np.argsort(norm)[::-1][:128].tolist()
    scores[str(l)] = {"top128": top128, "score_top8": norm[top128[:8]].round(4).tolist()}
    print(f"layer {l}: vecinos top8 score={norm[top128[:8]].round(3).tolist()}")
json.dump(scores, open(OUT, "w"), indent=1)
print("guardado:", OUT)
