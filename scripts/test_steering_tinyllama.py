import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"C:\Users\Anto\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"
print("Loading model and tokenizer...")
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
model.eval()

u_data = np.load("data/u_loop_tinyllama_L10.npz")
u = torch.from_numpy(u_data["u"]).float() # (2048,)

prompt = "The loop continues and repeats itself. The loop continues and repeats itself. The loop"
inputs = tok(prompt, return_tensors="pt")

print("\nPrompt:", prompt)

# 1. Baseline
with torch.no_grad():
    out_base = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
gen_base = tok.decode(out_base[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n[BASELINE]:", repr(gen_base))

# 2. Hook to inject steering along translated vector at Layer 10
for alpha in [-1.0, -2.0, -3.0, 1.0, 2.0]:
    def hook_fn(module, args, output):
        # output is tuple (hidden_states, ...)
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * u.to(h.device)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = model.model.layers[10].register_forward_hook(hook_fn)
    with torch.no_grad():
        out_steered = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
    handle.remove()
    gen_steered = tok.decode(out_steered[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[STEERED alpha={alpha:+.1f}]:", repr(gen_steered))

# Also check knowledge retention under steering
know_prompt = "The chemical symbol for water is H2O. The chemical symbol for gold is"
know_inputs = tok(know_prompt, return_tensors="pt")
for alpha in [0.0, -2.0]:
    if alpha != 0.0:
        def hook_fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h + alpha * u.to(h.device)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        handle = model.model.layers[10].register_forward_hook(hook_fn)
    else:
        handle = None

    with torch.no_grad():
        out_k = model.generate(**know_inputs, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id)
    if handle:
        handle.remove()
    gen_k = tok.decode(out_k[0][know_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n[KNOWLEDGE alpha={alpha:+.1f}]:", repr(gen_k))
