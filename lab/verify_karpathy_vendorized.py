"""verify_karpathy_vendorized — paridad REAL contra el model.py de karpathy/nanoGPT vendorizado.
Audit-fix 2026-11-04 (retract 26): se reemplaza la tautologia (misma clase 2x) por:
  engine/nanogpt_runtime.KarpathyGPT  vs  vendor/nanoGPT/model.py:GPT (commit 3adf61e)
  tolerancia fp32 (2e-4), NO 0.0-exacto (kernels SDPA difieren; 0.0 exacto seria sospechoso).
Manifest: lab/vendor/NANOGPT_COMMIT.txt
"""
import hashlib, json, sys
from pathlib import Path
import torch

R = Path(r"C:\weights")
sys.path.insert(0, str(R / "engine"))
sys.path.insert(0, str(R / "lab" / "vendor" / "nanoGPT"))

VENDOR_COMMIT = (R / "lab" / "vendor" / "NANOGPT_COMMIT.txt").read_bytes().decode("utf-8", "ignore").strip().replace("\x00", "")
VDIR = R / "lab" / "vendor" / "nanoGPT"
import model as vend_model  # karpathy/nanoGPT (vendor)
import nanogpt_runtime as ours  # engine/

CFG = dict(n_layer=4, n_head=4, n_embd=128, block_size=64, vocab_size=65)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

res = {"vendor_commit": VENDOR_COMMIT,
       "vendor_model_sha256": sha(VDIR / "model.py"),
       "cfg": CFG}

# vendor GPT
torch.manual_seed(1337)
vend = vend_model.GPT(vend_model.GPTConfig(vocab_size=CFG["vocab_size"], block_size=CFG["block_size"],
                                           n_layer=CFG["n_layer"], n_head=CFG["n_head"], n_embd=CFG["n_embd"]))

# engine KarpathyGPT (misma config, sin load de checkpoint)
torch.manual_seed(1337)
ours_ = ours.KarpathyGPT(ours.GPTConfig(vocab_size=CFG["vocab_size"], block_size=CFG["block_size"],
                          n_layer=CFG["n_layer"], n_head=CFG["n_head"], n_embd=CFG["n_embd"]))

vend.eval(); ours_.eval()
x = torch.randint(0, CFG["vocab_size"], (3, 47))
# nanoGPT forward devuelve solo el ultimo token si targets=None (modo inferencia):
# para paridad completa hay que pasar targets (label tensor) y comparar logits/loss completos.
y = torch.randint(0, CFG["vocab_size"], x.shape)
x2 = x.clone(); x2[:, 32] = (x2[:, 32] + 7) % 65
y2 = torch.randint(0, CFG["vocab_size"], x.shape)
with torch.no_grad():
    lv, los_v = vend(x, y)
    ret_o = ours_(x, y)
    lo, los_o = (ret_o[0], ret_o[1]) if isinstance(ret_o, tuple) else (ret_o, None)
    lv2, _ = vend(x2, y2)
    ret_o2 = ours_(x2, y2)
    lo2 = ret_o2[0] if isinstance(ret_o2, tuple) else ret_o2
diff = float((lv - lo).abs().max())
res["max_logits_diff"] = diff
res["cos"] = round(float((lv.flatten() * lo.flatten()).sum() /
                         (lv.flatten().norm() * lo.flatten().norm() + 1e-30)), 9)
res["shapes_match"] = (tuple(lv.shape) == tuple(lo.shape))
# separacion causal real (mascara): mutacion en pos 32 solo debe afectar posiciones >= 32
res["causal_purity_diff_v_past"] = float((lv[:, :32] - lv2[:, :32]).abs().max())
res["causal_purity_v_future"] = float((lv[:, 32:] - lv2[:, 32:]).abs().max())
res["loss_vend"] = round(float(los_v), 6)
res["loss_ours"] = round(float(los_o) if los_o is not None else -1.0, 6)

TOL = 2e-4
res["parity_verdict"] = ("PASSES_fp32" if res["max_logits_diff"] <= TOL else
                        ("CLOSE_BUT_NOT_EXACT" if diff < 0.05 else "DIVERGES"))
res["interpretacion"] = ("paridad REAL contra model.py vendorizado (no tautologia); 0.0 exacto habria sido sospechoso"
                           if diff > 0 and diff <= 0.05 else "verdict")
print(json.dumps(res, indent=2, ensure_ascii=False))
sys.path.insert(0, str(R))
outp = R / "rig_runs" / "karpathy_vendorized_parity.json"
outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("guardado", outp)
