"""Stage-0 premise probe for lab/probes/cc_phi/PLAN.md (Phi-4-mini-reasoning).

  python lab/probes/cc_phi/probe.py items <port> <max_tokens>   -> lab/runs/cc_phi/probe_<max_tokens>.json  (P1, P2, p3)
  python lab/probes/cc_phi/probe.py randvec <out.gguf>          -> same-norm random control vector, d = 3072, layers 1..31 (P3)
  python lab/probes/cc_phi/probe.py compare <port_base_json> <port_vec_json>  (P3: outputs differ?)
  python lab/probes/cc_phi/probe.py offsets                     -> P4 (tokenizer offsets on a sample reasoning text)
The served model is chosen by the shell (lab/runs/cc_phi/probe.sh). Nothing here decides; it prints what the plan asks.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "lab" / "runs" / "cc_phi"
HF = ROOT / "hf_cache" / "phi4mr-hf"


def items(port, max_tokens, suffix=""):
    from engine import bench_runner as b
    RUNS.mkdir(parents=True, exist_ok=True)
    name = f"probe_{max_tokens}{suffix}"
    r = b.run(name, int(port), gsm_n=5, math_n=5, max_tokens=int(max_tokens), parallel=2, out_dir=str(RUNS), log=print)
    d = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    rows = d["items"]
    rep = {"n": len(rows),
           "P1_reasoning_nonempty": sum(1 for x in rows if x["reasoning_chars"] > 0),
           "P1_content_has_think_tag": sum(1 for x in rows if "<think>" in x["tail"] and x["reasoning_chars"] == 0),
           "P2_cut": sum(1 for x in rows if x["finish"] != "stop"),
           "finished_with_boxed_pred": sum(1 for x in rows if x["finish"] == "stop" and x["pred"]),
           "correct": sum(1 for x in rows if x["ok"]),
           "tokens": [x["tokens"] for x in rows]}
    print(json.dumps(rep))
    return rep


def randvec(out, d=3072, layers=32, norm=1.0, seed=2026):
    import numpy as np
    from engine.contrast_vector import write_gguf
    g = np.random.default_rng(seed)
    vecs = {}
    for k in range(1, layers):
        v = g.normal(size=d).astype(np.float32)
        vecs[k] = v * (norm / np.linalg.norm(v))
    write_gguf(vecs, out, "phi4mr")
    print(json.dumps({"out": str(out), "layers": len(vecs), "norm": norm}))


def compare(a, b):
    A = {(x["bench"], x["i"]): x for x in json.loads(Path(a).read_text(encoding="utf-8"))["items"]}
    B = {(x["bench"], x["i"]): x for x in json.loads(Path(b).read_text(encoding="utf-8"))["items"]}
    same_tail = sum(1 for k in A if k in B and A[k]["tail"] == B[k]["tail"])
    same_tokens = sum(1 for k in A if k in B and A[k]["tokens"] == B[k]["tokens"])
    print(json.dumps({"items": len(A), "same_tail": same_tail, "same_tokens": same_tokens,
                      "P3_applied": same_tail < len(A)}))


def offsets():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(HF))
    text = "<think>\nWe need 3 * 4 = 12 apples. So the answer is 12.\n</think>\nThe answer is \\boxed{12}."
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    ok = len(off) == len(enc["input_ids"]) and all(e > s for s, e in off if e) and off[-1][1] == len(text)
    print(json.dumps({"tokens": len(off), "P4_offsets_ok": ok, "first": off[:4], "last": off[-3:],
                      "think_token_ids": tok("<think>", add_special_tokens=False)["input_ids"]}))


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"items": items, "randvec": randvec, "compare": compare, "offsets": offsets}[cmd](*a)
