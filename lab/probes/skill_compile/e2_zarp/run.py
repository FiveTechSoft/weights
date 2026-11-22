"""Run E2 against llama.cpp /completion; no tool, no rule in the prompt."""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from generate import SEED, read_split

ARMS = ("base", "placebo", "candidate")
SPLITS = ("sealed", "arithmetic", "dev", "discovery")


def parse_integer(text):
    return int(text.strip()) if re.fullmatch(r"[+-]?\d+", text.strip()) else None


def completion(server, prompt, timeout=120):
    body = json.dumps({"prompt": prompt, "temperature": 0, "seed": SEED,
                       "n_predict": 64, "stream": False, "cache_prompt": False}).encode()
    request = urllib.request.Request(server.rstrip("/") + "/completion", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result.get("content"), str):
        raise ValueError("llama.cpp response missing content")
    return result["content"]


def run(data, split, arm, model_sha, server, output):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", model_sha):
        raise ValueError("model-sha256 must be a 64-character hex SHA256")
    if arm not in ARMS or split not in SPLITS:
        raise ValueError("Invalid arm or split")
    items, dataset_sha = read_split(data, split)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: interrupted runs cannot masquerade as complete runs.
    tmp = output.with_suffix(output.suffix + ".partial")
    if tmp.exists():
        raise FileExistsError(f"Partial file already exists: {tmp}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for item in items:
                raw = completion(server, item["prompt"])
                row = {"id": item["id"], "prompt": item["prompt"], "answer": item["answer"],
                       "raw": raw, "parsed": parse_integer(raw),
                       "correct": parse_integer(raw) == item["answer"],
                       "arm": arm, "split": split, "model_sha256": model_sha.lower(),
                       "dataset_sha256": dataset_sha, "seed": SEED, "n_predict": 64,
                       "temperature": 0}
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
        tmp.rename(output)
    except Exception:
        # Keep partial for diagnostics, but the scorer only accepts final filenames.
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=SPLITS, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8080")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.data, a.split, a.arm, a.model_sha256, a.server, a.out)
