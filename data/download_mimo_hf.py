import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="hf_cache/mimo-distill-hf", help="dir destino (local)")
ap.add_argument("--resume", action="store_true", default=True)
a = ap.parse_args()
out = Path(a.out).resolve()
os.environ["HF_HOME"] = str(out.parent)
os.environ["HF_HUB_DISABLE_XET"] = os.environ.get("HF_HUB_DISABLE_XET", "1")
snap = snapshot_download(
    repo_id="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
    local_dir=str(out),
    allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
    max_workers=4,
)
print("downloaded:", snap)
