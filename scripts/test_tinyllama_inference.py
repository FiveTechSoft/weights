import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\Users\Anto\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"
print("Loading model and tokenizer...")
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
model.eval()

# Load translated vector
u_data = np.load("data/u_loop_tinyllama_L10.npz")
u = torch.from_numpy(u_data["u"]).float().view(-1, 1) # (2048, 1)

prompts = [
    # General knowledge holdout
    "The capital city of France is",
    "Water is made of hydrogen and",
    "Solve: 15 + 27 =",
    # Repetitive / loop prompt
    "The loop continues and repeats itself. The loop continues and repeats itself. The loop",
    # Agentic prompt
    "<|system|>\nYou are an assistant. To execute a command, use tool: calc(a, b).</s>\n<|user|>\nCalculate 12 * 8.</s>\n<|assistant|>\n"
]

def run_suite(label):
    print(f"\n--- {label} ---")
    results = []
    for p in prompts:
        inputs = tok(p, return_tensors="pt")
        with torch.no_grad():
            t0 = time.time()
            out = model.generate(
                **inputs,
                max_new_tokens=25,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
            dt = time.time() - t0
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        print(f"[{dt:.2f}s] Prompt: {p[:40]}... -> Response: {repr(text.strip())}")
        results.append((p, text))
    return results

print("RUNNING BASELINE (BEFORE SURGERY)...")
res_before = run_suite("BEFORE (Original TinyLlama)")

# Apply LEACE Surgery on Layer 10 down_proj
print("\nAPPLYING LEACE WEIGHT SURGERY ON LAYER 10 down_proj...")
W = model.model.layers[10].mlp.down_proj.weight # (2048, 5632)
proj = u.T @ W # (1, 5632)
delta = u @ proj # (2048, 5632)
energy_before = torch.norm(proj).item()

with torch.no_grad():
    W.sub_(delta)

proj_after = u.T @ W
energy_after = torch.norm(proj_after).item()
print(f"Energy before: {energy_before:.6f}")
print(f"Energy after:  {energy_after:.6e}")
print(f"Max abs delta: {torch.max(torch.abs(delta)).item():.6f}")

print("\nRUNNING POST-SURGERY (AFTER SURGERY)...")
res_after = run_suite("AFTER (Surgically Ablated TinyLlama)")
