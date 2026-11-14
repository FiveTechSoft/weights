"""G0 (reported, PLAN.md): greedy agreement of the first 256 tokens on 10 held-out calibration prompts.
usage: agree.py gen <name> <port>   -> lab/runs/cc_weights/agree_<name>.json
       agree.py cmp <a> <b>         -> identical outputs and mean common-prefix share"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "lab" / "runs" / "cc_weights"
from lab.probes.cc_weights.compile import TRACES, SPLIT_SEED, N_FIT  # noqa: E402


def prompts():
    from engine import bench_runner as b
    rows = [json.loads(x) for x in TRACES.read_text(encoding="utf-8").splitlines() if x.strip()]
    order = list(range(len(rows)))
    random.Random(SPLIT_SEED).shuffle(order)
    return [rows[i]["prompt"][: -len(b.SUFFIX)] for i in order[N_FIT:N_FIT + 10]]


def gen(name, port):
    from engine import bench_runner as b
    url = b.endpoint_url(int(port))
    out = [b.ask(url, p, 256) for p in prompts()]
    (RUNS / f"agree_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmp(a, b):
    A, B = (json.loads((RUNS / f"agree_{n}.json").read_text(encoding="utf-8")) for n in (a, b))
    same, share = 0, []
    for x, y in zip(A, B):
        s, t = x["reasoning"] + x["content"], y["reasoning"] + y["content"]
        k = next((i for i, (c, d) in enumerate(zip(s, t)) if c != d), min(len(s), len(t)))
        same += s == t
        share.append(k / max(len(s), len(t), 1))
    res = {"pair": [a, b], "identical": same, "of": len(A), "mean_common_prefix_share": round(sum(share) / len(share), 3)}
    print(json.dumps(res))
    with open(RUNS / "agree.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    gen(a[1], a[2]) if a[0] == "gen" else cmp(a[1], a[2])
