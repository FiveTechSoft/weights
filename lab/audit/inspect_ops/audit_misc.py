"""Behavioral drift sensitivity, CLI exit codes on failed asserts, GGUF canned branches."""
import json
import subprocess
from harness import run, dump, MODELS, OUT, ROOT, PY

R = {}
A = MODELS / "rand_1337.safetensors"
B = MODELS / "rand_7.safetensors"
TRAINED = ROOT / "lab/runs/als_adam_small.pt"
NW3 = ROOT / "lab/runs/nanowsp3_pretrained.safetensors"
corpus = "corpus=lab/data/shakespeare_char/input.txt"

# assert_behavioral_drift: does the measured ppl depend on the target model?
for nm, m in (("rand_1337", A), ("rand_7", B), ("als_adam_trained", TRAINED), ("nanowsp3_pretrained", NW3)):
    r = run(f"assert_behavioral_drift {corpus} max_ppl_increase=1.0", model=m)["step"]
    R[f"behav_{nm}"] = {k: r.get("result", {}).get(k) for k in ("measured_perplexity", "verdict")} | {"status": r["status"], "state": r.get("execution_state")}
r = run(f"assert_behavioral_drift {corpus} max_ppl_increase=0.0000001", model=NW3)["step"]
R["behav_nanowsp3_ratio_1e-7"] = {"verdict": r["result"]["verdict"], "ppl": r["result"]["measured_perplexity"]}

# CLI: does a failed assertion / VIOLATION make the program fail (exit code)?
u = OUT / "u64.npz"
progs = {
    "dk_max_angle_fail": f"step assert_davis_kahan transformer.h.0.mlp.c_proj.weight vector={u.as_posix()} alpha=1.0 max_angle_deg=0.001",
    "behav_violation": f"step assert_behavioral_drift {corpus} max_ppl_increase=1.0",
    "missing_tensor": "step audit no.such.tensor",
}
for nm, line in progs.items():
    p = OUT / f"cli_{nm}.wspl"
    p.write_text(f"experiment audit_cli_{nm}:\nmodel: nanoGPT\nsteps:\n    {line}\n", encoding="utf-8")
    cp = subprocess.run([PY, str(ROOT / "engine/wsplang.py"), str(p), "--run", "--model", str(A)], capture_output=True, text=True, cwd=str(ROOT))
    R[f"cli_{nm}"] = {"exit_code": cp.returncode, "stdout_tail": cp.stdout.strip().splitlines()[-3:]}

# GGUF-backend canned strings (reachable when execute_plan_v4 is called directly; CLI --check rejects them)
G = ROOT / "hf_cache/qwen35-08b-gguf/Qwen_Qwen3.5-0.8B-Q6_K.gguf"
for line in ("spectral_audit blk.3.ffn_down.weight", f"assert_weyl blk.3.ffn_down.weight vector={u.as_posix()}",
             f"assert_davis_kahan blk.3.ffn_down.weight vector={u.as_posix()}", "spectral_audit no.such.tensor"):
    r = run(line, model=G)
    R[f"gguf_{line.split()[0]}_{line.split()[1]}"] = {"feas_err": r["feasibility_errors"][:1], "status": r["step"]["status"], "result": r["step"].get("result")}

print(json.dumps(R, indent=1, default=str))
dump("misc", R)
