import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\Users\Anto\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"
print("Loading TinyLlama-1.1B...")
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
model.eval()

# Facts to test
target_facts = [
    ("The chemical element with atomic number 79 is", " gold"),
    ("The deepest lake in the world is Lake", " Baikal"),
]

holdout_facts = [
    ("The capital of Burkina Faso is", "Ouagadougou"),
    ("The capital of France is", "Paris"),
    ("Water is made of hydrogen and", "oxygen"),
    ("Calculate 15 + 27 =", "42"),
    ("The creator of the Linux kernel is Linus", "Torvalds"),
]

def evaluate_all(label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    results = {}
    print("--- TARGET FACTS ---")
    for prompt, expected in target_facts:
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok = expected.strip().lower() in gen.lower()
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt} -> {repr(gen)} (Expected: {expected.strip()})")
        results[prompt] = (ok, gen)
        
    print("\n--- HOLDOUT FACTS (COLLATERAL DAMAGE CHECK) ---")
    for prompt, expected in holdout_facts:
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok = expected.strip().lower() in gen.lower()
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt} -> {repr(gen)} (Expected: {expected.strip()})")
        results[prompt] = (ok, gen)
    return results

print("RUNNING BASELINE EVALUATION...")
baseline_res = evaluate_all("BASELINE (BEFORE KNOWLEDGE IMPRINT)")

# -------------------------------------------------------------
# MEMIT KNOWLEDGE IMPRINTING ON LAYER 10 down_proj
# -------------------------------------------------------------
print("\n" + "="*70)
print("EXTRACTING KEYS AND IMPRINTING KNOWLEDGE VIA CLOSED-FORM WSP")
print("="*70)

target_layer = 10
mlp = model.model.layers[target_layer].mlp
W_down = mlp.down_proj.weight # (2048, 5632)

# Collect keys k at input to down_proj for the target prompts
keys = []
values = []

for prompt, expected in target_facts:
    inputs = tok(prompt, return_tensors="pt")
    
    # Hook to capture input to down_proj at last token position
    captured_k = []
    def hook_k(module, args, output):
        # args[0] is input to down_proj, shape (batch, seq_len, 5632)
        k_last = args[0][0, -1, :].detach().clone()
        captured_k.append(k_last)
        
    handle = mlp.down_proj.register_forward_hook(hook_k)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    k = captured_k[0] # (5632,)
    keys.append(k)
    
    # Target value v*:
    # We want down_proj to emit a vector that boosts the target token logits
    target_token_id = tok.convert_tokens_to_ids(expected)
    if target_token_id is None or target_token_id == tok.unk_token_id:
        target_token_id = tok.encode(expected, add_special_tokens=False)[0]
    
    # Target un-embedding vector
    lm_head_w = model.lm_head.weight[target_token_id].detach().clone() # (2048,)
    # Scale v* to have appropriate magnitude for the residual stream
    current_out = W_down @ k # (2048,)
    target_v = current_out + 3.0 * lm_head_w / torch.norm(lm_head_w) * torch.norm(current_out)
    values.append(target_v)
    print(f"Captured key for '{prompt[:35]}...' norm={torch.norm(k):.2f}, target token id={target_token_id} ({expected})")

# Batch MEMIT update:
# Delta W = (V - W K) @ (K^T K + lambda * I)^-1 @ K^T
K = torch.stack(keys, dim=1) # (5632, N)
V = torch.stack(values, dim=1) # (2048, N)
lambda_reg = 1e-3

# Solve least squares for Delta W: Delta W @ K = (V - W @ K)
Residual = V - W_down @ K # (2048, N)
# Gram matrix
G = K.T @ K + lambda_reg * torch.eye(K.shape[1]) # (N, N)
G_inv = torch.linalg.inv(G) # (N, N)
Delta_W = Residual @ G_inv @ K.T # (2048, 5632)

frob_orig = torch.norm(W_down).item()
frob_delta = torch.norm(Delta_W).item()
max_delta = torch.max(torch.abs(Delta_W)).item()

print(f"\nMEMIT Delta W computed:")
print(f"  Frobenius norm of original W: {frob_orig:.4f}")
print(f"  Frobenius norm of Delta W:    {frob_delta:.4f} ({frob_delta/frob_orig*100:.3f}% perturbation)")
print(f"  Max absolute element change:  {max_delta:.6f}")

# Apply update in-place
with torch.no_grad():
    W_down.add_(Delta_W)

print("\nRUNNING POST-IMPRINT EVALUATION...")
post_res = evaluate_all("POST-IMPRINT (AFTER WSP KNOWLEDGE INJECTION)")
