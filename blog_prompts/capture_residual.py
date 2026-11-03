"""capture_residual.py — E2: dirección unitaria u de repetición desde residuos (método blog exp 8).

Contextos: 2 productivos vs 2 repetitivos (re-llamada idéntica de tools), construidos con el
formato real del rig (antml). Forward passes (budget 4). Salva u por capa en npz.
Uso: python blog_prompts/capture_residual.py
"""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
P = r"C:\weights\hf_cache\mimo-distill-hf"
OUT = r"C:\weights\data\residuos_e2.npz"

L_PROMPTS = json.load(open(r"C:\temp\opencode\rl_oss_prompts.json", encoding="utf-8"))
PROMPT_REP = "Investiga el repositorio /app: lista los fuentes relevantes y responde."
PROMPT_PRO = "Investiga el repositorio /app: busca como se carga el modelo y responde."
RECALL_CALL = "<function=bash><parameter=command>ls -la /app</parameter></function>"
PRO_CALLS = ["grep -rn 'load_model' /app/src | head", "find /app/config -name '*.yaml' | head", "wc -l /app/src/*.py"]

MOCK = "[mock harness] executed: ok. stdout captured: total 8\ndrwxr-xr-x .  .. .git src config README.md"


def build_msgs(kind, prompt):
    msgs = [{"role": "user", "content": prompt}]
    for t in range(3):
        call = RECALL_CALL if kind == "rep" else f"<function=bash><parameter=command>{PRO_CALLS[t]}</parameter></function>"
        msgs.append({"role": "assistant", "content": "Investigando.\n" + call})
        msgs.append({"role": "user", "content": f"Tool results:\nResult for `bash`: {MOCK}\nContinue."})
    return msgs


def marker_positions(input_ids, tok, marker_tokens):
    ids = input_ids.tolist()
    pos = {}
    for name, mids in marker_tokens.items():
        pos[name] = [i for i, tx in enumerate(ids) if tx in mids]
    return pos


def residuals_for(msgs, model, tok, marker_tokens):
    bt = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True, return_dict=True)
    ids_i = bt["input_ids"]
    if isinstance(ids_i, (list, tuple)) and len(ids_i) and not isinstance(ids_i[0], (list, tuple, int, float)):
        ids_i = ids_i[0]
    input_ids = torch.tensor([list(ids_i)], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hid = out.hidden_states  # tuple: (L+1) x (1, T, 4096)
    pos = marker_positions(input_ids[0], tok, marker_tokens)
    print(f"  fwd {time.time() - t0:.0f}s T={input_ids.shape[1]} pos={ {k: len(v) for k, v in pos.items()} }")
    resid = {}
    for k, xs in pos.items():
        vectors = []
        for l in range(len(hid)):
            mm = hid[l][0, xs, :].float().mean(axis=0).numpy()
            assert np.asarray(mm).shape == (4096,), f"shape invalido layer {l}: {np.asarray(mm).shape}"
            vectors.append(mm)
        resid[k] = np.stack(vectors)
        assert resid[k].shape == (len(hid), 4096), f"resid {k}: {resid[k].shape}"
    return resid


def main():
    tok = AutoTokenizer.from_pretrained(P, local_files_only=True)
    tu_ct = open(r"C:\weights\data\chat_template.jinja", encoding="utf-8").read()
    tok.chat_template = tu_ct
    ids_markers = {}
    for st in ("function", "|function"):
        try:
            tid = tok.convert_tokens_to_ids(st)
            if tid and tid != tok.unk_token_id:
                ids_markers.setdefault(st, [tid])
        except Exception:
            pass
    if "function" in tok.get_vocab():
        ids_markers["function"] = [tok.get_vocab()["function"]]
    print("markers:", ids_markers)
    toks_txt = tok.convert_ids_to_tokens(list(range(len(tok))))
    fnm = [i for i, tg in enumerate(toks_txt) if "function" in tg.lower()][:6]
    if fnm:
        ids_markers.setdefault("any_function", fnm)
    print("candidatos:", {k: v for k, v in ids_markers.items()})
    del toks_txt

    print("cargando modelo bf16 CPU (18GB)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(P, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True, device_map="cpu" if hasattr(torch, "device_map") else None)
    model.eval()
    print(f"cargo {time.time() - t0:.0f}s")

    all_res = []
    for kind, pid in (("rep", PROMPT_REP), ("pro", PROMPT_PRO)):
        for ctx_i in (1, 2):
            msgs = build_msgs(kind, pid)
            print(f"fwd {kind}#{ctx_i} ...")
            all_res.append((kind, ctx_i, residuals_for(msgs, model, tok, ids_markers)))
    L = len(all_res[0][2].get("function", all_res[0][2][list(all_res[0][2].keys())[0]])) if False else None
    key0 = list(all_res[0][2].keys())[0]
    L = all_res[0][2][key0].shape[0]
    saved = {}
    for l in range(L):
        rep = np.stack([r[2][key0][l] for r in all_res if r[0] == "rep"], axis=0)
        pro = np.stack([r[2][key0][l] for r in all_res if r[0] == "pro"], axis=0)
        u = (rep.mean(axis=0) - pro.mean(axis=0))
        assert u.shape == (4096,), f"u_l{l} escalar: rep {rep.shape} pro {pro.shape}"
        u = u / (np.linalg.norm(u) + 1e-30)
        saved[f"u_l{l}"] = u.astype(np.float32)
        saved[f"rep_l{l}"] = rep.mean(axis=0).astype(np.float32)
        saved[f"pro_l{l}"] = pro.mean(axis=0).astype(np.float32)
    np.savez_compressed(OUT, **saved)
    print("guardado:", OUT, "layers:", L)


if __name__ == "__main__":
    main()
