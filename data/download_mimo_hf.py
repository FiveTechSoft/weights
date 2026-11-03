import os
import json
from pathlib import Path

from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = r"C:\weights\hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
snap = snapshot_download(
    repo_id="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
    local_dir=r"C:\weights\hf_cache\mimo-distill-hf",
    allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
    max_workers=4,
)
print("downloaded:", snap)
