import os
from pathlib import Path

from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = r"C:\weights\hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
snap = snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    local_dir=r"C:\weights\hf_cache\qwen35-08b-hf",
    max_workers=4,
)
print("stage A descargado:", snap)
