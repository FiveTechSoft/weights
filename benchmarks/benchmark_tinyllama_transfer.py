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

# Multi-layer hook across middle layers (8..15)
class MultiLayerSteering:
    def __init__(self, model, layers, u, alpha):
        self.model = model
        self.layers = layers
        self.u = u
        self.alpha = alpha
        self.handles = []

    def __enter__(self):
        for l in self.layers:
            def make_hook(layer_idx):
                def hook_fn(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    h = h + self.alpha * self.u.to(h.device)
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h
                return hook_fn
            h = self.model.model.layers[l].register_forward_hook(make_hook(l))
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()

benchmarks = [
    # Knowledge holdout
    ("KNOW_FRANCE", "The capital city of France is"),
    ("KNOW_WATER", "Water consists of oxygen and"),
    ("KNOW_MATH", "Calculate 15 + 27 ="),
    ("KNOW_GOLD", "The chemical symbol for gold is"),
    
    # Loop prompts
    ("LOOP_CYCLE", "The loop continues and repeats itself. The loop continues and repeats itself. The loop"),
    ("LOOP_ECHO", "Echo test: ping pong ping pong ping pong ping"),
    
    # Agentic prompts
    ("AGENT_TOOL", "<|system|>\nYou are an agent. If you need a calculation, use [TOOL: calc(expr)].</s>\n<|user|>\nWhat is 24 * 7?</s>\n<|assistant|>\n"),
    ("AGENT_PLAN", "<|system|>\nYou are a planning agent. Output steps in JSON format: {\"steps\": [...]}.</s>\n<|user|>\nPlan how to bake a cake.</s>\n<|assistant|>\n")
]

print("\n" + "="*80)
print("BENCHMARK: TINYLAMA BEFORE VS AFTER TRANSFER STEERING")
print("="*80)

for name, prompt in benchmarks:
    inputs = tok(prompt, return_tensors="pt")
    
    # 1. Baseline
    with torch.no_grad():
        out_base = model.generate(**inputs, max_new_tokens=25, do_sample=False, pad_token_id=tok.eos_token_id)
    text_base = tok.decode(out_base[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    # 2. Steered across layers 8..14 with alpha = -2.5
    with MultiLayerSteering(model, range(8, 15), u, alpha=-2.5):
        with torch.no_grad():
            out_steer = model.generate(**inputs, max_new_tokens=25, do_sample=False, pad_token_id=tok.eos_token_id)
        text_steer = tok.decode(out_steer[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
    print(f"\n--- [{name}] ---")
    print(f"Prompt: {prompt.replace(chr(10), ' ')}")
    print(f"  BEFORE: {repr(text_base)}")
    print(f"  AFTER:  {repr(text_steer)}")
