"""scratch/compare_karpathy_vs_wsplang_generations.py — Comparación directa Karpathy vs WSPLang.

Carga los mismos pesos (models/nanogpt_karpathy/model.safetensors) en:
  1. La implementación canónica de Karpathy (KarpathyGPT).
  2. El runtime de WSPLang (NanoGPT).
Genera texto autoregresivo sobre 8 prompts diversos bajo decodificación Greedy (T=0.0)
y bajo muestreo estocástico controlado (T=0.8, top_k=10, misma semilla).
"""

import json
import sys
from pathlib import Path

R = Path(r"C:\weights")
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "engine"))

import torch
from engine.nanogpt_runtime import GPTConfig, KarpathyGPT, NanoGPT

ckpt_p = R / "models" / "nanogpt_karpathy" / "model.safetensors"
assert ckpt_p.exists(), f"El modelo SafeTensors no existe en {ckpt_p}"

cfg = GPTConfig(
    block_size=256,
    vocab_size=65,
    n_layer=6,
    n_head=6,
    n_embd=384,
    d_ff=1536,
    dropout=0.0,
    bias=True
)

# 1. Instanciar Referencia Original de Karpathy
karpathy_model = KarpathyGPT(cfg)
from safetensors.torch import load_file
sd = load_file(str(ckpt_p))
karpathy_model.load_state_dict(sd, strict=False)
karpathy_model.transformer.wte.weight = karpathy_model.lm_head.weight
karpathy_model.eval()

# 2. Instanciar WSPLang Runtime
wsplang_model = NanoGPT(config=cfg, seed=1337)
wsplang_model.load_checkpoint(ckpt_p)
wsplang_model.model.eval()

# Vocabulario canónico de 65 caracteres de Shakespeare
vocab_str = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
c2i = {ch: i for i, ch in enumerate(vocab_str)}
i2c = {i: ch for i, ch in enumerate(vocab_str)}

# Función de generación canónica de Karpathy (exacta a sample.py)
@torch.no_grad()
def karpathy_generate(model, idx, max_new_tokens=30, temperature=0.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]
        b, t = idx_cond.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx_cond.device)
        tok_emb = model.transformer.wte(idx_cond)
        pos_emb = model.transformer.wpe(pos)
        x = model.transformer.drop(tok_emb + pos_emb)
        for block in model.transformer.h:
            x = block(x)
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x[:, -1, :])

        if temperature == 0.0:
            idx_next = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Lista de 8 prompts de prueba
test_prompts = [
    "ROMEO:\n",
    "To be, or not to be, ",
    "First Citizen:\nWe are ",
    "KING LEAR:\n",
    "Shall I compare thee to a ",
    "What light through yonder ",
    "Friends, Romans, countrymen, ",
    "Speak, speak.\n"
]

results = []
print("=== COMPARACIÓN DIRECTA: ORIGINAL KARPATHY vs WSPLANG ===")

for p in test_prompts:
    p_ids = torch.tensor([[c2i.get(c, 0) for c in p]], dtype=torch.long)
    
    # --- PRUEBA 1: GREEDY T=0.0 ---
    gen_k_greedy = karpathy_generate(karpathy_model, p_ids.clone(), max_new_tokens=25, temperature=0.0)
    gen_w_greedy = wsplang_model.generate(p_ids.clone(), max_new_tokens=25, temperature=0.0)
    
    str_k_greedy = "".join(i2c.get(t, "") for t in gen_k_greedy[0].tolist())
    str_w_greedy = "".join(i2c.get(t, "") for t in gen_w_greedy[0].tolist())
    
    greedy_diff = int((gen_k_greedy != gen_w_greedy).sum().item())
    
    # --- PRUEBA 2: SAMPLE T=0.8, TOP_K=10 (CON LA MISMA SEMILLA) ---
    torch.manual_seed(999)
    gen_k_sample = karpathy_generate(karpathy_model, p_ids.clone(), max_new_tokens=25, temperature=0.8, top_k=10)
    
    torch.manual_seed(999)
    gen_w_sample = wsplang_model.generate(p_ids.clone(), max_new_tokens=25, temperature=0.8, top_k=10)
    
    str_k_sample = "".join(i2c.get(t, "") for t in gen_k_sample[0].tolist())
    str_w_sample = "".join(i2c.get(t, "") for t in gen_w_sample[0].tolist())
    
    sample_diff = int((gen_k_sample != gen_w_sample).sum().item())
    
    entry = {
        "prompt": p,
        "greedy": {
            "karpathy": str_k_greedy[len(p):],
            "wsplang": str_w_greedy[len(p):],
            "diff_count": greedy_diff,
            "identical": bool(greedy_diff == 0)
        },
        "sampled_t08": {
            "karpathy": str_k_sample[len(p):],
            "wsplang": str_w_sample[len(p):],
            "diff_count": sample_diff,
            "identical": bool(sample_diff == 0)
        }
    }
    results.append(entry)
    
    print(f"\nPrompt: {p!r}")
    print(f"  [Greedy Karpathy]: {str_k_greedy[len(p):]!r}")
    print(f"  [Greedy WSPLang]:  {str_w_greedy[len(p):]!r}")
    print(f"  --> Greedy Identical? {greedy_diff == 0} (Diffs: {greedy_diff})")
    print(f"  [Sample Karpathy]: {str_k_sample[len(p):]!r}")
    print(f"  [Sample WSPLang]:  {str_w_sample[len(p):]!r}")
    print(f"  --> Sample Identical? {sample_diff == 0} (Diffs: {sample_diff})")

out_file = R / "lab" / "runs" / "karpathy_vs_wsplang_comparison.json"
out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[OK] Resultados guardados en {out_file}")
