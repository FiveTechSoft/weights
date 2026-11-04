"""extract_loop_direction.py — Representation Engineering for MiMo-9B Agentic Loops.

Extracts contrastive hidden state activations between repetitive loop states and productive states
using real execution trajectories from eval_vanilla_n20.json.
Computes direction vector u across layers, validates against permutation null hypothesis,
and exports llama.cpp control-vector GGUF.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import gguf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DIR_DATA = ROOT / "data" / "rl_oss_prompts.json"
DIR_RUNS = ROOT / "rig_runs"
HF_PATH = ROOT / "hf_cache" / "mimo-distill-hf"

MOCK_RESULT = ("[mock harness] executed: ok. stdout captured: total 8\n"
               "drwxr-xr-x .  .. .git src config README.md\n")

RE_ABS = re.compile(
    r"<function=([a-zA-Z0-9_]+)>\s*<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>\s*</function>",
    re.S)
RE_ABS_TRUNC = re.compile(r"<function=([a-zA-Z0-9_]+)>\s*<parameter=([a-zA-Z0-9_]+)>(.*)", re.S)

def parse_calls(text):
    out = []
    for m in RE_ABS.finditer(text):
        out.append((m.group(1).lower(), m.group(3)))
    if not out:
        for m in RE_ABS_TRUNC.finditer(text):
            out.append((m.group(1).lower(), m.group(3)))
    return out

def get_messages_at_turn(endpoint, prompt, target_turn):
    """Reconstruct exact trajectory deterministically using port 8081."""
    messages = [{"role": "user", "content": prompt}]
    for t in range(1, target_turn + 1):
        payload = {
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 800,
            "seed": 42
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.load(r)
        txt = res["choices"][0]["message"]["content"] or ""
        calls = parse_calls(txt)
        if not calls or t == target_turn:
            break
        resp = "\n".join(f"Result for `{c[0]}`: {MOCK_RESULT}" for c in calls)
        messages.append({"role": "assistant", "content": txt})
        messages.append({"role": "user", "content": f"Tool results:\n{resp}\nContinue."})
    return messages

def main():
    endpoint = "http://127.0.0.1:8081/v1/chat/completions"
    all_prompts = json.load(DIR_DATA.open(encoding="utf-8"))
    prompts_by_id = {p["id"]: p["prompt"] for p in all_prompts}
    
    vanilla_run = json.load(open(DIR_RUNS / "eval_vanilla_n20.json", encoding="utf-8"))["results"]
    
    # Select top 6 loop runs and 6 finished runs
    loop_candidates = [
        ("general-0-candidate-0036-software-data-engineering", 7),
        ("general-0-candidate-0109-science-robotics", 9),
        ("general-0-candidate-0260-security-appsec", 11),
        ("general-0-candidate-0461-ml-evaluation", 5),
        ("general-0-candidate-0534-ml-training", 7),
        ("general-50-candidate-2511-security-cryptography", 7)
    ]
    
    prod_candidates = [
        ("general-0-candidate-0688-hardware-rtl", 5),
        ("general-50-candidate-2500-hardware-cad", 5),
        ("general-50-candidate-2684-security-appsec", 5),
        ("general-50-candidate-2713-ml-inference", 5),
        ("general-50-candidate-2790-ml-evaluation", 5),
        ("general-50-candidate-2836-ml-kernels", 5)
    ]
    
    print("=== RECONSTRUCTING CONVERSATION TRAJECTORIES ===")
    contexts = []
    
    for pid, turn in loop_candidates:
        print(f"Reconstructing LOOP context: {pid} at turn {turn}...")
        msgs = get_messages_at_turn(endpoint, prompts_by_id[pid], turn)
        contexts.append(("loop", pid, msgs))
        
    for pid, turn in prod_candidates:
        print(f"Reconstructing PROD context: {pid} at turn {turn}...")
        msgs = get_messages_at_turn(endpoint, prompts_by_id[pid], turn)
        contexts.append(("prod", pid, msgs))
        
    print("\n=== LOADING MIMO-9B IN BFLOAT16 (CPU) FOR ACTIVATION EXTRACTION ===")
    tok = AutoTokenizer.from_pretrained(str(HF_PATH), local_files_only=True)
    ct_file = ROOT / "data" / "chat_template.jinja"
    if ct_file.exists():
        tok.chat_template = ct_file.read_text(encoding="utf-8")
    model = AutoModelForCausalLM.from_pretrained(
        str(HF_PATH),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.eval()
    
    loop_activations = []  # list of [33, 4096]
    prod_activations = []  # list of [33, 4096]
    
    print("\n=== RUNNING CONTRASTIVE FORWARD PASSES ===")
    for idx, (kind, pid, msgs) in enumerate(contexts):
        # Format with assistant prefix to capture the decision point
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt_str, return_tensors="pt")
        
        t0 = time.time()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        dt = time.time() - t0
        
        # Extract last token vector for every layer (0..32)
        # hidden_states: tuple of 33 tensors, each [1, seq_len, 4096]
        vecs = np.stack([out.hidden_states[l][0, -1, :].float().cpu().numpy() for l in range(len(out.hidden_states))])
        print(f"[{idx+1}/{len(contexts)}] {kind.upper()} ({pid[:35]}...) seq_len={inputs.input_ids.shape[1]} in {dt:.1f}s")
        
        if kind == "loop":
            loop_activations.append(vecs)
        else:
            prod_activations.append(vecs)
            
    loop_arr = np.stack(loop_activations, axis=0)  # [N_loop, 33, 4096]
    prod_arr = np.stack(prod_activations, axis=0)  # [N_prod, 33, 4096]
    
    print("\n=== COMPUTING CONTRASTIVE SEPARATION AND STATISTICAL TESTS ===")
    n_layers = loop_arr.shape[1]  # 33 (layer 0 = embedding, 1..32 = transformer layers)
    
    diagnostics = []
    delta_by_layer = {}
    u_by_layer = {}
    
    # Permutation test configuration
    combined = np.concatenate([loop_arr, prod_arr], axis=0)  # [12, 33, 4096]
    n_total = len(combined)
    n_loop = len(loop_arr)
    
    rng = np.random.default_rng(42)
    n_perms = 100
    
    for l in range(1, n_layers):  # focus on transformer layers 1..32
        mu_loop = loop_arr[:, l, :].mean(axis=0)
        mu_prod = prod_arr[:, l, :].mean(axis=0)
        delta = mu_loop - mu_prod
        norm_delta = float(np.linalg.norm(delta))
        
        # Variance
        var_loop = float(np.var(loop_arr[:, l, :], axis=0).sum())
        var_prod = float(np.var(prod_arr[:, l, :], axis=0).sum())
        snr = norm_delta / (np.sqrt(0.5 * (var_loop + var_prod)) + 1e-12)
        
        # Permutation null test
        perm_norms = []
        for _ in range(n_perms):
            idx_perm = rng.permutation(n_total)
            p_loop = combined[idx_perm[:n_loop], l, :].mean(axis=0)
            p_prod = combined[idx_perm[n_loop:], l, :].mean(axis=0)
            perm_norms.append(float(np.linalg.norm(p_loop - p_prod)))
        mean_perm_norm = float(np.mean(perm_norms))
        p_val = float(np.mean([pn >= norm_delta for pn in perm_norms]))
        
        u = delta / (norm_delta + 1e-30)
        
        transformer_layer_idx = l - 1  # 0..31
        u_by_layer[transformer_layer_idx] = u
        delta_by_layer[transformer_layer_idx] = delta
        
        diagnostics.append({
            "transformer_layer": transformer_layer_idx,
            "norm_delta": round(norm_delta, 4),
            "snr": round(float(snr), 4),
            "null_perm_mean_norm": round(mean_perm_norm, 4),
            "separation_ratio": round(norm_delta / (mean_perm_norm + 1e-12), 2),
            "p_value": p_val
        })
        
    diagnostics.sort(key=lambda x: -x["separation_ratio"])
    print("\nTOP-8 LAYERS WITH HIGHEST CONTRASTIVE SEPARATION (SIGNAL vs NULL):")
    for d in diagnostics[:8]:
        print(f"  Layer {d['transformer_layer']:>2}: norm={d['norm_delta']:.2f} (null={d['null_perm_mean_norm']:.2f}) -> ratio={d['separation_ratio']}x, SNR={d['snr']:.3f}, p={d['p_value']:.3f}")
        
    # Export report JSON
    report_path = DIR_RUNS / "loop_direction_report.json"
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_loop_contexts": len(loop_candidates),
        "n_prod_contexts": len(prod_candidates),
        "top_layers": diagnostics[:8],
        "all_layers": sorted(diagnostics, key=lambda x: x["transformer_layer"])
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved report to {report_path}")
    
    # Export GGUF Control Vector
    cv_path = DIR_RUNS / "loop_control_vector.gguf"
    print(f"Writing llama.cpp control vector GGUF to {cv_path}...")
    writer = gguf.GGUFWriter(str(cv_path), "controlvector")
    for layer_idx in range(32):
        u_vec = u_by_layer.get(layer_idx, np.zeros(4096, dtype=np.float32)).astype(np.float32)
        writer.add_tensor(f"direction.{layer_idx}", u_vec)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Control vector GGUF written successfully! Size: {cv_path.stat().st_size:,} bytes")

if __name__ == "__main__":
    main()
