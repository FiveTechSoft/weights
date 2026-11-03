"""capture_heads.py — E6: disparidad rep-vs-pro por SAMPLE-headed (self_attn) 
en capas de atencion de MiMo (HF), usados para el edit WSP sobre attn_output.

Scoraje: para cada head h (16 por capa), mean_c de (rep − pro) / (max|.|) por posicion. 
Salva: data/head_scores_e6.json (top heads por capa) y data/head_vecs_e6.npz (u_resid 4096).
"""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
P = r"C:\weights\hf_cache\mimo-distill-hf"
TARGET_LAYERS = [17, 19]  # se filtra a las que tengan self_attn
sys.path.insert(0, r"C:\weights\blog_prompts")
import capture_residual as CR

tok = AutoTokenizer.from_pretrained(P, local_files_only=True)
tok.chat_template = open(r"C:\weights\data\chat_template.jinja", encoding="utf-8").read()
model = AutoModelForCausalLM.from_pretrained(P, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
model.eval()

LAYERS_ATT = [l for l in range(32) if hasattr(model.model.layers[l], "self_attn") and model.model.layers[l].self_attn is not None]
print("capas de self_attn:", LAYERS_ATT)
TARGET_LAYERS = [l for l in (17, 19) if l in LAYERS_ATT]
print("objetivo:", TARGET_LAYERS)


def fwd_capture_heads(msgs, layers):
    outs = {}
    for idx in layers:

        def make_hk(idx=idx):
            def hk(module, inp, out):
                o = out[0].detach() if isinstance(out, tuple) else out.detach()  # (1,T,4096)
                if o.dim() == 3:
                    o = o[0]  # (T, 4096)
                headsz = o.shape[-1] // 16
                per_head = o.view(o.shape[0], 16, headsz).float().mean(axis=0).numpy()  # (16, headsz)
                outs[idx] = per_head.copy()
            return hk

        mod = model.model.layers[idx].self_attn
        mod.register_forward_hook(make_hk(idx))
    bt = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True, return_dict=True)
    ids_i = bt["input_ids"]
    if isinstance(ids_i, (list, tuple)) and len(ids_i) and not isinstance(ids_i[0], (list, tuple, int, float)):
        ids_i = ids_i[0]
    ii = torch.tensor([list(ids_i)], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        model(input_ids=ii, attention_mask=torch.ones_like(ii), use_cache=False)
    print(f"fwd {time.time() - t0:.0f}s out={[outs.get(l, None) is not None for l in layers]}")
    return outs


reps, pros = {}, {}
for kind in ("rep", "pro"):
    for i in (1, 2):
        pid = CR.PROMPT_REP if kind == "rep" else CR.PROMPT_PRO
        outs = fwd_capture_heads(CR.build_msgs(kind, pid), TARGET_LAYERS)
        for l in TARGET_LAYERS:
            (reps if kind == "rep" else pros).setdefault(l, []).append(outs[l])

scores, vecs = {}, {}
for l in TARGET_LAYERS:
    mr = np.stack(reps[l]).mean(axis=0)  # (16, headsz)
    mp = np.stack(pros[l]).mean(axis=0)
    diff = np.abs(mr - mp).mean(axis=1)  # (16,)
    top = np.argsort(diff)[::-1]
    scores[str(l)] = {"rank": top.tolist(), "scores": [round(float(diff[i]), 3) for i in top]}
    print(f"layer {l}: heads top4={top[:4].tolist()} scores={[round(float(diff[i]),3) for i in top[:4]]}")
json.dump(scores, open(r"C:\weights\data\head_scores_e6.json", "w"), indent=1)
u_dir = {}
for l in TARGET_LAYERS:
    mr = np.stack(reps[l]).mean(axis=0)
    mp = np.stack(pros[l]).mean(axis=0)
    for h in scores[str(l)]["rank"][:4]:
        d = (mr[h] - mp[h]).astype(np.float64)
        vecs[f"u_l{l}_h{h}"] = d
np.savez_compressed(r"C:\weights\data\head_vecs_e6.npz", **vecs)
print("guardado npz:", len(vecs), "vectores")
