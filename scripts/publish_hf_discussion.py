#!/usr/bin/env python3
"""
Publish discussion post to Hugging Face model repository.

Usage:
    # Dry run (verify content and target):
    python scripts/publish_hf_discussion.py --dry-run

    # Publish with token argument:
    python scripts/publish_hf_discussion.py --token hf_...

    # Publish using HF_TOKEN environment variable:
    set HF_TOKEN=hf_...
    python scripts/publish_hf_discussion.py
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Publish WSP fix discussion to Hugging Face")
    parser.add_argument("--repo-id", default="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", help="Target Hugging Face repo")
    parser.add_argument("--file", default="docs/post_huggingface_discussion.md", help="Markdown file containing post content")
    parser.add_argument("--token", default=None, help="Hugging Face User Access Token (write permission)")
    parser.add_argument("--dry-run", action="store_true", help="Print information without posting")
    args = parser.parse_args()

    content_path = Path(args.file)
    if not content_path.is_file():
        print(f"Error: file not found: {content_path}", file=sys.stderr)
        sys.exit(1)

    text = content_path.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    
    # Extract title from first markdown header
    title = "[Fix & Weight Surgery] Resolving Agentic Tool-Calling Infinite Loops via Surgical 0.24% Rank-1 Projection"
    body = text
    if lines and lines[0].startswith("# "):
        title = lines[0].lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()

    print(f"Target Repo:  https://huggingface.co/{args.repo_id}")
    print(f"Discussions:  https://huggingface.co/{args.repo_id}/discussions")
    print(f"Title:        {title}")
    print(f"Body length:  {len(body)} characters ({len(body.splitlines())} lines)")

    if args.dry_run:
        print("\n[DRY RUN] Everything is validated. To publish, run without --dry-run and provide --token.")
        return

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\nError: No Hugging Face token provided.", file=sys.stderr)
        print("Please provide --token <your_hf_token> or set HF_TOKEN environment variable.", file=sys.stderr)
        print(f"Alternatively, you can open a new discussion manually at: https://huggingface.co/{args.repo_id}/discussions/new", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"\nSubmitting discussion to {args.repo_id}...")
    try:
        discussion = api.create_discussion(
            repo_id=args.repo_id,
            title=title,
            description=body,
            token=token,
            pull_request=False
        )
        print(f"SUCCESS! Discussion created:")
        print(f"URL: {discussion.url}")
    except Exception as e:
        print(f"Failed to create discussion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
