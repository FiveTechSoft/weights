"""publish_to_huggingface.py — Publish MiMo-V2.6-Distill-Qwen-9B-WSP-Critic to Hugging Face.
Usage:
    python scripts/publish_to_huggingface.py --token <HF_TOKEN_WRITE> [--repo FiveTechSoft/MiMo-V2.6-Distill-Qwen-9B-WSP-Critic]
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Upload WSP Critic model to Hugging Face Hub")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API Write Token (hf_...)")
    parser.add_argument("--repo", type=str, default="FiveTechSoft/MiMo-V2.6-Distill-Qwen-9B-WSP-Critic", help="Target HF repo ID")
    parser.add_argument("--private", action="store_true", help="Create as private repository")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub is not installed. Please run: pip install huggingface_hub")
        sys.exit(1)

    model_dir = Path("models/mimo-wsp-critic").resolve()
    if not model_dir.exists():
        print(f"[ERROR] Directory not found: {model_dir}")
        sys.exit(1)

    print("=" * 80)
    print(f"PUBLICANDO MODELO EN HUGGING FACE HUB: {args.repo}")
    print(f"Directorio local: {model_dir}")
    print("=" * 80)

    api = HfApi(token=args.token)

    print(f"[*] Creando repositorio '{args.repo}' en Hugging Face...")
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
    print("[OK] Repositorio creado / verificado.")

    print("[*] Subiendo archivos (shards SafeTensors, configs, tokenizer y README)...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo,
        repo_type="model"
    )

    # Subir también el control vector GGUF para usuarios de llama.cpp
    cv_path = Path("data/u_critic_control_vector.gguf").resolve()
    if cv_path.exists():
        print(f"[*] Subiendo vector de control GGUF adicional: {cv_path.name}...")
        api.upload_file(
            path_or_fileobj=str(cv_path),
            path_in_repo="u_critic_control_vector.gguf",
            repo_id=args.repo,
            repo_type="model"
        )

    print("=" * 80)
    print(f"[EXITO] Modelo publicado exitosamente en:")
    print(f"https://huggingface.co/{args.repo}")
    print("=" * 80)

if __name__ == "__main__":
    main()
