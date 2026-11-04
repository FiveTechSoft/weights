import torch
import json
import numpy as np
from safetensors import safe_open
from pathlib import Path
from transformers import AutoTokenizer

# Load Rosetta matrix
rosetta = np.load("data/rosetta_mimo_to_tinyllama.npz")
T = torch.from_numpy(rosetta["T"]).float() # (2048, 4096)

# Load MiMo vocab
with open("hf_cache/mimo-distill-hf/vocab.json", "r", encoding="utf-8") as f:
    mimo_vocab = json.load(f)

# Load TinyLlama vocab
tiny_tok = AutoTokenizer.from_pretrained(r"C:\Users\Anto\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6")

# Load MiMo lm_head
mimo_shard = Path("hf_cache/mimo-distill-hf/model-00001-of-00004.safetensors")
with safe_open(mimo_shard, framework="pt") as f:
    mimo_lm_head = f.get_tensor("lm_head.weight") # (248320, 4096)

# Load TinyLlama lm_head
tiny_shard = Path(r"C:\Users\Anto\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6\model.safetensors")
with safe_open(tiny_shard, framework="pt") as f:
    tiny_lm_head = f.get_tensor("lm_head.weight") # (32000, 2048)

test_words = ["gold", "Baikal", "Paris", "water", "science", "oxygen", "Linux", "Tolkien"]
print("=" * 70)
print("PROCRUSTES ALIGNMENT OF SPECIFIC CONCEPTS: MiMo-9B -> TinyLlama-1.1B")
print("=" * 70)
for w in test_words:
    # Try with leading space / without
    m_id = mimo_vocab.get("Ġ" + w) or mimo_vocab.get(" " + w) or mimo_vocab.get(w)
    t_id = tiny_tok.convert_tokens_to_ids(" " + w) or tiny_tok.convert_tokens_to_ids(w)
    if m_id and t_id:
        v_mimo = mimo_lm_head[m_id].float()
        v_tiny_actual = tiny_lm_head[t_id].float()
        v_tiny_proj = T @ v_mimo
        cos = torch.dot(v_tiny_proj, v_tiny_actual) / (torch.norm(v_tiny_proj) * torch.norm(v_tiny_actual))
        print(f"Concept '{w}': MiMo id={m_id:6d}, Tiny id={t_id:6d} -> Cosine = {cos.item():.4f}")
    else:
        print(f"Concept '{w}': m_id={m_id}, t_id={t_id}")
