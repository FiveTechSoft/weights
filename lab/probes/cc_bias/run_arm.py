"""Run one bench_runner arm with a constant logit bias on one token (lab/probes/cc_bias/PLAN.md).

  python lab/probes/cc_bias/run_arm.py <name> <port> <out_dir> <token_id> <bias> [max_tokens] [gsm_idx_json] [math_idx_json]

bench_runner.ask has no hook for extra request fields and engine/bench_runner.py is not edited here: this wrapper
installs an `ask` with the same body plus llama-server's `logit_bias` ([[token_id, bias]]) into the module before
calling bench_runner.run, which looks `ask` up at call time. The request is otherwise identical (greedy, seed 0,
SUFFIX, max_tokens). The bias is recorded in the report's `extra` field as text so the JSON says what was served.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine import bench_runner as b  # noqa: E402


def make_ask(token_id, bias):
    def ask(url, text, max_tokens, lora=None, extra=""):
        body = {"messages": [{"role": "user", "content": text + b.SUFFIX}], "temperature": 0, "seed": 0,
                "max_tokens": max_tokens, "logit_bias": [[int(token_id), float(bias)]]}
        if lora:
            body["lora"] = [{"id": int(i), "scale": float(s)} for i, s in lora]
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
        ch = r["choices"][0]
        msg = ch["message"]
        content, reasoning = b.split_think(msg.get("content") or "", msg.get("reasoning_content") or "")
        return {"content": content, "reasoning": reasoning, "finish": ch["finish_reason"],
                "tokens": r["usage"]["completion_tokens"],
                "tok_s": round(r.get("timings", {}).get("predicted_per_second", 0), 1)}
    return ask


def main(argv):
    if len(argv) < 5:
        print(__doc__)
        return 2
    name, port, out_dir, token_id, bias = argv[0], int(argv[1]), argv[2], int(argv[3]), float(argv[4])
    max_tokens = int(argv[5]) if len(argv) > 5 else 2048
    gsm_idx = json.loads(argv[6]) if len(argv) > 6 else None
    math_idx = json.loads(argv[7]) if len(argv) > 7 else None
    b.ask = make_ask(token_id, bias)
    # `extra` is only a label here: our ask ignores it, so the prompt is unchanged; the report records the bias
    b.run(name, port, out_dir=out_dir, max_tokens=max_tokens, parallel=2, gsm_idx=gsm_idx, math_idx=math_idx,
          log=lambda m: print(m, flush=True), extra=f"[logit_bias {token_id}:{bias:+g}; prompt unchanged]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
