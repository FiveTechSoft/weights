"""Reproduce Simon Willison's weight interpretation experiment on MiMo-9B.

Paper/Demo: "Can Large Language Models Read Weight Matrices?" (MIT / David Bau et al.)
Prompt contains the raw weights of a 17,253 parameter MLP policy network trained with PPO.
The model must identify which Atari game it plays purely from reading the weight matrices.
"""

import argparse
import json
import os
import sys
import time
import urllib.request


def query_server(prompt: str, url: str = "http://127.0.0.1:8081/v1/chat/completions", max_tokens: int = 4096):
    payload = {
        "model": "mimo-v2.6-9b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": True,
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    
    print(f"[*] Sending request to {url} (prompt length: {len(prompt)} chars)...", flush=True)
    t0 = time.time()
    full_content = []
    
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                chunk_str = line[6:]
                if chunk_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    # llama.cpp server might put reasoning in reasoning_content or content
                    token = delta.get("content", "")
                    reasoning_token = delta.get("reasoning_content", "")
                    if reasoning_token:
                        sys.stdout.write(reasoning_token)
                        sys.stdout.flush()
                        full_content.append(reasoning_token)
                    elif token:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        full_content.append(token)
                except json.JSONDecodeError:
                    pass
                    
    total_time = time.time() - t0
    full_text = "".join(full_content)
    print(f"\n\n[*] Completed in {total_time:.2f}s ({len(full_text)} chars generated)", flush=True)
    return full_text, total_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=["pong", "breakout"], default="pong")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    prompt_path = f"data/exp1_{args.game}_prompt.txt"
    if not os.path.exists(prompt_path):
        print(f"Error: Prompt file {prompt_path} not found.")
        sys.exit(1)
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    output_text, duration = query_server(prompt, url=url)
    
    out_file = args.out or f"rig_runs/simon_willison_{args.game}_result.json"
    result = {
        "game_tested": args.game,
        "prompt_path": prompt_path,
        "prompt_chars": len(prompt),
        "duration_seconds": duration,
        "output_text": output_text,
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print(f"[*] Saved result to {out_file}")


if __name__ == "__main__":
    main()
