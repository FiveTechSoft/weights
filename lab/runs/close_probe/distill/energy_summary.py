"""Weighted kept energy at rank 16 from a delta_to_lora energy.json (same formula as engine.lora_export) + verdict."""
import json, sys
for p in sys.argv[1:]:
    s = json.load(open(p))
    w = sum(v["norm"] ** 2 for v in s.values())
    k16 = sum(v["kept_at"]["16"] * v["norm"] ** 2 for v in s.values()) / w
    v = "likely" if k16 >= 0.5 else "unlikely" if k16 < 0.2 else "uncertain"
    print(p, f"matrices={len(s)} kept16={k16:.4f} verdict={v} total_delta_norm={w ** 0.5:.1f}")
