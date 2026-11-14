"""Recompute the Atari 'decompilation' numbers from the prompt files, independently of rig/verify_atari_interpretability.py."""
import json
import re
import numpy as np
from harness import dump, ROOT

R = {}
for game in ("pong", "breakout"):
    txt = (ROOT / f"data/exp1_{game}_prompt.txt").read_text(encoding="utf-8")
    i = txt.find("W3")
    hdrs = [m.start() for m in re.finditer(r"W3", txt)]
    seg = txt[hdrs[0]:]
    rows = []
    for line in seg.splitlines()[1:]:
        nums = re.findall(r"-?\d+\.\d+(?:e-?\d+)?", line)
        if len(nums) == 5:
            rows.append([float(x) for x in nums])
        if len(rows) == 32:
            break
    W3 = np.array(rows)
    std = W3.std(0); mx = np.abs(W3).max(0)
    v, h = std[1] + std[2], std[3] + std[4]
    total_nums = len(re.findall(r"-?\d+\.\d+", txt))
    R[game] = {"W3_shape": list(W3.shape), "std": np.round(std, 3).tolist(), "maxabs": np.round(mx, 2).tolist(),
               "vertical": round(float(v), 3), "horizontal": round(float(h), 3), "ratio": round(float(max(v, h) / min(v, h)), 2),
               "W3_header_context": txt[hdrs[0]:hdrs[0] + 60].replace("\n", " | "), "numbers_in_prompt": total_nums}
# how does the prompt label actions? (is the up/down/left/right mapping in the file or assumed?)
txt = (ROOT / "data/exp1_pong_prompt.txt").read_text(encoding="utf-8")
R["prompt_mentions"] = {w: (w in txt.lower()) for w in ("pong", "breakout", "up", "down", "left", "right", "action")}
R["prompt_head"] = txt[:600]
print(json.dumps(R, indent=1))
dump("atari", R)
