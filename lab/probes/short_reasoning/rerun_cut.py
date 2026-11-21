"""Re-ask only the items that hit the token cap in a previous bench_runner run, with a larger cap.

Same protocol as engine/bench_runner.py (greedy, seed 0, same prompt, same graders); only max_tokens changes.
Answers one question with no weight edit: how much of a model's score loss is the cap and not the model.

usage: python lab/probes/short_reasoning/rerun_cut.py <prev.json> <new_name> <port> [max_tokens=32768] [parallel=1]
writes lab/runs/short_reasoning/<new_name>.json and prints the corrected accuracy per benchmark.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine import bench_runner as br  # noqa: E402


def main(prev, name, port, max_tokens=32768, parallel=1):
    m5 = br._m5()
    old = json.loads(Path(prev).read_text(encoding="utf-8"))
    cut = [r for r in old["items"] if r["finish"] == "length"]
    gsm = [json.loads(x) for x in (ROOT / "data" / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()]
    math_all = m5.load_items()
    url = br.endpoint_url(port)
    out_dir = ROOT / "lab" / "runs" / "short_reasoning"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{name}.partial.jsonl"
    rows = [json.loads(x) for x in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {(r["bench"], r["i"]) for r in rows}
    print(f"[{name}] {len(cut)} items were cut at {old['max_tokens']}; {len(done)} already re-asked", flush=True)

    import concurrent.futures

    def one(r):
        text = gsm[r["i"]]["question"] if r["bench"] == "gsm8k" else math_all[r["i"]]["problem"]
        a = br.ask(url, text, max_tokens)
        pred = m5.extract_pred(a["content"])
        ok1 = bool(pred) and m5.equivalent(pred, r["gold"])
        ok2 = br.grade_gsm_independent(a["content"], r["gold"]) if r["bench"] == "gsm8k" else (
            bool(pred) and br.simple_norm(pred) == br.simple_norm(r["gold"]))
        return {"bench": r["bench"], "i": r["i"], "gold": r["gold"], "pred": pred, "ok": ok1, "ok_independent": ok2,
                "finish": a["finish"], "tokens": a["tokens"], "tok_s": a["tok_s"], "tail": (a["reasoning"] + a["content"])[-300:]}

    todo = [r for r in cut if (r["bench"], r["i"]) not in done]
    with concurrent.futures.ThreadPoolExecutor(parallel) as ex:
        for row in ex.map(one, todo):
            rows.append(row)
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{name}] {row['bench']} {row['i']} ok={row['ok']} finish={row['finish']} tokens={row['tokens']}", flush=True)

    new = {(r["bench"], r["i"]): r for r in rows}
    summary = {}
    for bench in ("gsm8k", "math500"):
        items = [new.get((r["bench"], r["i"]), r) for r in old["items"] if r["bench"] == bench]
        if not items:
            continue
        summary[bench] = {"n": len(items),
                          "acc_old_cap": round(sum(r["ok"] for r in old["items"] if r["bench"] == bench) / len(items), 3),
                          "acc_new_cap": round(sum(r["ok"] for r in items) / len(items), 3),
                          "still_cut": sum(r["finish"] == "length" for r in items),
                          "reasked": sum((bench, r["i"]) in new for r in items)}
    rep = {"from": str(prev), "old_cap": old["max_tokens"], "new_cap": max_tokens, "summary": summary, "items": rows}
    (out_dir / f"{name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], *(int(x) for x in a[3:]))
