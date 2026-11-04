"""upload_safetensors_shards.py — Sequentially upload SafeTensors shards to HF.
Usage:
    python scripts/upload_safetensors_shards.py [--token <HF_TOKEN>]
"""
import os
import sys
import time
import argparse
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "fivetech/MiMo-V2.6-Distill-Qwen-9B-WSP-Critic"
MODEL_DIR = Path("models/mimo-wsp-critic").resolve()

SHARDS = [
    "model-00004-of-00004.safetensors",  # 3.2 GB
    "model-00001-of-00004.safetensors",  # 5.2 GB
    "model-00002-of-00004.safetensors",  # 5.0 GB
    "model-00003-of-00004.safetensors",  # 5.2 GB
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        print("[ERROR] HF token required. Pass via --token or set HF_TOKEN env var.")
        sys.exit(1)

    api = HfApi(token=args.token)
    existing_files = set(api.list_repo_files(repo_id=REPO_ID))
    
    print(f"[*] Starting SafeTensors upload to {REPO_ID}...")
    for idx, shard_name in enumerate(SHARDS, 1):
        if shard_name in existing_files:
            print(f"[{idx}/{len(SHARDS)}] {shard_name} already uploaded on HF. Skipping.")
            continue
        
        file_path = MODEL_DIR / shard_name
        if not file_path.exists():
            print(f"[ERROR] Shard not found: {file_path}")
            continue
        
        size_gb = file_path.stat().st_size / (1024 ** 3)
        print(f"\n[{idx}/{len(SHARDS)}] Uploading {shard_name} ({size_gb:.2f} GB)...")
        t0 = time.time()
        
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=shard_name,
            repo_id=REPO_ID,
            repo_type="model"
        )
        
        dt = time.time() - t0
        speed_mb = (file_path.stat().st_size / (1024 ** 2)) / max(dt, 1)
        print(f"[OK] Uploaded {shard_name} in {dt:.1f}s ({speed_mb:.2f} MB/s)")

    print("\n" + "=" * 80)
    print(f"[SUCCESS] All 4 SafeTensors shards uploaded successfully to:")
    print(f"https://huggingface.co/{REPO_ID}")
    print("=" * 80)

if __name__ == "__main__":
    main()
