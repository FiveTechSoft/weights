"""Compare chat models served by llama-server on GSM8K and MATH-500 (protocol: engine/bench_runner.py).

usage: python lab/probes/model_compare/eval.py <name> <port> [gsm_n] [math_n] [parallel]
Writes lab/runs/model_compare/<name>.json (and <name>.partial.jsonl while running).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import bench_runner as benchmarks  # noqa: E402

if __name__ == "__main__":
    a = sys.argv[1:]
    benchmarks.run(a[0], a[1], *(int(x) for x in a[2:]), log=lambda s: print(s, flush=True))
