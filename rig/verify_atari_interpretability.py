"""Verify Atari policy weight interpretability (MIT CSAIL Exp 1 & Simon Willison replication).

Analyzes the raw weight matrices of Pong and Breakout policies:
1. Mathematical signature: computes action variance across output layer W3.
   - Pong: dominated by Action 1 (Up) and Action 2 (Down) (paddle moves vertically).
   - Breakout: dominated by Action 3 (Left, max 5.34) and Action 4 (Right, max 3.06) (paddle moves horizontally).
2. Semantic reasoning: queries the local model (port 8081) with the structural weight observation
   to deduce the exact game title and confidence.
"""

import json
import urllib.request
import numpy as np


def parse_weights(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract W3
    idx_w3 = text.find("W3 (output layer weights)")
    sub_w3 = text[idx_w3:]
    lines = [l.strip() for l in sub_w3.splitlines()[1:] if l.strip()]
    w3_rows = []
    for l in lines:
        parts = l.split()
        if len(parts) == 5:
            try:
                w3_rows.append([float(x) for x in parts])
            except ValueError:
                pass
        if len(w3_rows) == 32:
            break

    # Extract b3 (output biases)
    idx_b3 = text.find("b3 (output layer biases)")
    b3 = []
    if idx_b3 != -1:
        sub_b3 = text[idx_b3:]
        for l in sub_b3.splitlines()[1:]:
            parts = l.strip().split()
            if len(parts) == 5:
                try:
                    b3 = [float(x) for x in parts]
                    break
                except ValueError:
                    pass

    return np.array(w3_rows), np.array(b3) if b3 else np.zeros(5)


def analyze_game_weights(game_name, filename):
    w3, b3 = parse_weights(filename)
    actions = ["0: no-op", "1: up", "2: down", "3: left", "4: right"]
    
    stds = w3.std(axis=0)
    max_abs = np.abs(w3).max(axis=0)
    
    vertical_power = stds[1] + stds[2]
    horizontal_power = stds[3] + stds[4]
    
    inferred = "Pong" if vertical_power > horizontal_power else "Breakout"
    ratio = vertical_power / max(horizontal_power, 1e-6) if vertical_power > horizontal_power else horizontal_power / max(vertical_power, 1e-6)
    
    return {
        "game": game_name,
        "file": filename,
        "actions_std": {actions[i]: round(float(stds[i]), 4) for i in range(5)},
        "actions_max_abs": {actions[i]: round(float(max_abs[i]), 4) for i in range(5)},
        "vertical_activity (up/down)": round(float(vertical_power), 4),
        "horizontal_activity (left/right)": round(float(horizontal_power), 4),
        "inferred_game_from_weights": inferred,
        "axis_dominance_ratio": round(float(ratio), 2),
    }


def query_llm_deduction(game_name, stats, url="http://127.0.0.1:8081/v1/chat/completions"):
    prompt = f"""You are a mechanistic interpretability expert analyzing a reinforcement learning policy network.
A small 17,253 parameter MLP was trained with PPO on an 8x8 playable grid surrounded by walls.
The network outputs 5 action logits:
0: no-op, 1: up (row 0), 2: down (row 9), 3: left (col 0), 4: right (col 9).

From inspecting the output layer W3 (32x5) and its parameter statistics, we observe:
- Standard deviation across the 32 hidden units projecting to each action:
  * Action 0 (no-op): {stats['actions_std']['0: no-op']}
  * Action 1 (up):    {stats['actions_std']['1: up']}
  * Action 2 (down):  {stats['actions_std']['2: down']}
  * Action 3 (left):  {stats['actions_std']['3: left']}
  * Action 4 (right): {stats['actions_std']['4: right']}

- Maximum absolute weight connected to each action:
  * Action 0 (no-op): {stats['actions_max_abs']['0: no-op']}
  * Action 1 (up):    {stats['actions_max_abs']['1: up']}
  * Action 2 (down):  {stats['actions_max_abs']['2: down']}
  * Action 3 (left):  {stats['actions_max_abs']['3: left']}
  * Action 4 (right): {stats['actions_max_abs']['4: right']}

Vertical activity (up/down sum): {stats['vertical_activity (up/down)']}
Horizontal activity (left/right sum): {stats['horizontal_activity (left/right)']}

QUESTION: Which Atari / arcade game was this policy trained to play (e.g. Pong vs Breakout vs Space Invaders)?
Work strictly from these weight distributions.

Answer in this exact format:
Observations: <concise summary of which action weights are large vs suppressed>
Reasoning: <why this motion axis fits the game mechanics>
Game: <game name>
Confidence: <0-100>%
"""
    payload = {
        "model": "mimo-v2.6-9b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Query error: {e}"


def main():
    print("=" * 70)
    print("ATARI POLICY WEIGHT INTERPRETATION (Simon Willison & MIT CSAIL Exp 1)")
    print("=" * 70)

    for game in ["pong", "breakout"]:
        filename = f"data/exp1_{game}_prompt.txt"
        stats = analyze_game_weights(game, filename)
        print(f"\n--- Analysing {game.upper()} Policy Weights ({filename}) ---")
        print(f"Vertical Activity (Up/Down):     {stats['vertical_activity (up/down)']}")
        print(f"Horizontal Activity (Left/Right): {stats['horizontal_activity (left/right)']}")
        print(f"Mathematical Inference:           {stats['inferred_game_from_weights']} (ratio: {stats['axis_dominance_ratio']}x)")
        
        print("\nQuerying local model on port 8081 for semantic deduction...")
        llm_resp = query_llm_deduction(game, stats)
        print("LLM Response:\n" + llm_resp.strip())
        print("-" * 70)


if __name__ == "__main__":
    main()
