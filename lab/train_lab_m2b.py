"""LAB M-2b: finetune-trace vs cirugia closed-form sobre el paciente REAL NanoWSP-3 (pretrained PPL 14.3).
Dataset: wspx RECALL (universo exacto 200). Trace de Adam sobre train (144). Cirugia: ridge closed-form
en el readout (output.weight + biases... output TIED) sobre features de W0 (h-final activada por el prompt).
Prohibido cirugia: usar W*/gradiente/optimizer-state. Protocolo 3-brazos + juez wspx certify (EXACT only si 256..).
Salida: lab/runs/m2b_results.json + record.
"""
import json, math, time, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R = Path(r"C:\weights")
sys.path.insert(0, str(R / "lab"))
sys.path.insert(0, str(R))
from wspx_datasets import recall  # noqa: E402
from engine.record import ExperimentRecord, validate, infer_tier  # noqa: E402

OUT = R / "lab" / "runs"
cseed = json  # placeholder for lint friendliness if needed
CFG = json.loads((R / "lab" / "runs" / "nanowsp3_train.json").read_text(encoding="utf-8"))

# reconstruct params del trace (W0_flat y snapshots_flat)
tr = np.load(OUT / "nanowsp3_trace.npz", allow_pickle=False)
ns = json.loads(str(tr["names"]))
names = ns
shapes_raw = json.loads(str(tr["shapes"]))
shapes = {k: tuple(v) for k, v in (shapes_raw if isinstance(shapes_raw, dict) else json.loads(shapes_raw)).items()}

def unflat(flat):
    parts, off = {}, 0
    base = {"q": "q", "k": "k", "v": "v", "o": "o", "min_": "min_", "mout": "mout",
            "an": "an", "mn": "mn"}
    for nm in names:
        n = int(np.prod(shapes[nm]))
        key = nm
        if "." in nm and nm.split(".")[1].isdigit():
            k_, idx = nm.split(".")
            key = f"{base.get(k_, k_)}.{idx}.weight"
        parts[key] = torch.tensor(flat[off:off + n].reshape(shapes[nm]), dtype=torch.float32)
        off += n
    assert off == len(flat)
    return parts

W0p = unflat(tr["W0_flat"])
# entrenado real: ultimo snapshot
snap = tr["snapshots_flat"]
Wtrp = unflat(snap[-1])
print("snapshot final en step", int(tr["snap_steps"][-1]))

d_model, n_layers, n_heads, d_head, d_ff = 256, 6, 8, 32, 512

def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * g

def rope(x):
    b, s, h, d = x.shape
    mx = torch.max_seq_len if False else 256
    theta = 10000.0
    inv = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
    t = torch.arange(s, dtype=torch.float32)
    ang = torch.outer(t, inv)
    fc = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.float().reshape(b, s, h, d // 2, 2))
    return torch.view_as_real(xc * fc.view(1, s, 1, d // 2)).flatten(3)

def fwd(x, P):
    B, S = x.shape
    h = F.embedding(x, P["embed.weight"])
    for l in range(n_layers):
        hn = rms(h, P[f"an.{l}.weight"])
        q = (hn @ P[f"q.{l}.weight"].T).view(B, S, n_heads, d_head)
        k = (hn @ P[f"k.{l}.weight"].T).view(B, S, n_heads, d_head)
        v = (hn @ P[f"v.{l}.weight"].T).view(B, S, n_heads, d_head)
        q, k = rope(q), rope(k)
        s_ = q @ k.transpose(-2, -1) / math.sqrt(d_head)
        mask = torch.triu(torch.full((S, S), float("-inf")), 1)
        att = torch.softmax(s_ + mask, dim=-1)
        h = h + (att @ v).transpose(1, 2).contiguous().view(B, S, d_model) @ P[f"o.{l}.weight"].T
        hm = rms(h, P[f"mn.{l}.weight"])
        h = h + F.gelu(hm @ P[f"min_.{l}.weight"].T) @ P[f"mout.{l}.weight"].T
    hf = rms(h, P["fn.weight"])
    logits = hf @ P["out.weight"].T
    return logits, hf

def encode(prompt):
    return torch.tensor(list(prompt.encode("ascii", "ignore")), dtype=torch.long)

ds = recall()
train_rows, uni = ds["train"], ds["universe"]
tr_idx = {r["prompt"] for r in train_rows}

def prep(rows):
    xs, ys = [], []
    for r in rows:
        tok = encode(r["prompt"])
        xs.append(tok)
        ys.append(ord(r["target"]))
    L = max(len(x) for x in xs)
    X = torch.zeros(len(xs), L, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, L - len(x):] = x
    return X, torch.tensor(ys, dtype=torch.long)

def score(P, rows, batch=16):
    correct = 0
    for i in range(0, len(rows), batch):
        X, y = prep(rows[i:i + batch])
        with torch.no_grad():
            lg, _ = fwd(X, P)
        pred = lg[:, -1].argmax(-1)
        correct += int((pred == y).sum())
    return correct / max(len(rows), 1)

# ---------- brazo 1: Adam finetune (train-only) ----------
P = {k: v.clone() for k, v in W0p.items()}
for k in ("embed.weight", "out.weight"):
    P[k].requires_grad_(True)
opt = torch.optim.Adam([P["out.weight"]], lr=3e-3)
Xtr, ytr = prep(train_rows)
t0 = time.time()
trace = []
TRAINABLE = ["out.weight", "mout.5.weight", "mn.5.weight", "fn.weight"]
for k in TRAINABLE:
    P[k].requires_grad_(True)
others = [v for k, v in P.items() if not v.requires_grad]
opt = torch.optim.Adam([P[k] for k in TRAINABLE], lr=3e-3)
for t in range(600):
    lg, _ = fwd(Xtr, P)
    loss = F.cross_entropy(lg[:, -1], ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    if t % 100 == 0 or t == 599:
        trace.append({"step": t + 1, "loss": round(float(loss.detach()), 4)})
tr_time = time.time() - t0
P_adam = {k: v.detach() for k, v in P.items()}

# Readout closed-form surgery: features h_final congeladas de W0 sobre train-prompt; ridge margin-coded
with torch.no_grad():
    Hf_tr = []
    for i in range(0, len(train_rows), 32):
        Xb, _ = prep(train_rows[i:i + 32])
        _, hfb = fwd(Xb, W0p)
        Hf_tr.append(hfb[:, -1])
    Hf_tr = torch.cat(Hf_tr)  # [n,256]
print("features", tuple(Hf_tr.shape), f"adam finetune {tr_time:.0f}s")

# objetivos margin: logit target = +9, resto = -9 (deduccion: CE logit big-margin)
lam = 1.0
T = torch.full((Hf_tr.shape[0], 256), -9.0)
T[torch.arange(len(train_rows)), ytr] = 9.0
Hn = Hf_tr.numpy().astype(np.float64)
Rm = T.numpy().astype(np.float64)
Wout_surg = np.linalg.solve(Hn.T @ Hn + lam * np.eye(256), Hn.T @ Rm)  # W@h.T approx linear readout
# cirugia solo toca out.weight, num neutro: P[...] += delta normalizado por h a proj
def apply_wout(Pp, Wnew):
    Pp = {k: v.clone() for k, v in Pp.items()}
    Pp["out.weight"] = torch.tensor(Wnew, dtype=torch.float32)
    return Pp
P_surg_biased = apply_wout(W0p, Wout_surg.T)  # logits = h @ W.T? fwd: h @ P["out.weight"].T => out.weight [256(vocab), 256hid] (as in engine: [vocab, d_model]

# placebo: misma ||Delta W_out||, dir aleatoria
delta_surg = torch.tensor(Wout_surg.T, dtype=torch.float32) - W0p["out.weight"]
delta_norm = float(delta_surg.norm())
rng = np.random.default_rng(21)
rnd = torch.tensor(rng.normal(size=W0p["out.weight"].shape), dtype=torch.float32)
rnd = rnd * (delta_norm / (rnd.norm() + 1e-12))
Wplc = W0p["out.weight"] + rnd
P_plac = apply_wout(W0p, Wplc)

answers = {}
res = {
    "dataset": "wspx_RECALL", "patient": "nanowsp3-pretrained-PPL14",
    "adam_steps": 600, "trace": trace,
    "acc": {
        "w0_base": score(W0p, uni),
        "adam_finetune": score(P_adam, uni),
        "closed_form_readout": score(P_surg_biased, uni),
        "placebo_norm_matched": score(P_plac, uni),
    },
    "seconds": round(tr_time, 1),
}
for r in uni:
    Xb, _ = prep([r])
    with torch.no_grad():
        lg, _ = fwd(Xb, P_surg_biased)
    answers[r["prompt"]] = chr(int(lg[0, -1].argmax()))
from certify import universe_eval  # noqa: E402
cert = universe_eval(ds, answers)
res["certify"] = cert
tier = infer_tier({"trained": max(res["acc"]["adam_finetune"], res["acc"]["w0_base"]),
                   "candidate": res["acc"]["closed_form_readout"], "placebo": res["acc"]["placebo_norm_matched"]},
                  chance=res["acc"]["w0_base"])
res["surgery_tier"] = tier

r = ExperimentRecord(
    experiment_id="e2026.117.m2b-readout-recall",
    model_id="nanowsp3-pretrained-PPL14",
    objective="cirugia readout closed-form vs finetune Adam en RECALL (lookup con paciente real)",
    baseline={"nombre": "W0 pretrain (PPL14)", "modelo": "nanowsp3"},
    edicion={"program": f"ridge-readout margin-coded lambda=1.0 (certify={cert['verdict']})",
             "operaciones": ["out.weight:=solve(HtH+lam, Ht*R)"],
             "tensor": "out.weight", "ruta": "lab/train_lab_m2b.py"},
    ejecucion={"estado": "ok"},
    resultados=res["acc"],
)
d = r.to_dict() if hasattr(r, "to_dict") else r.__dict__
d.update({"hypothesis": "cirugia closed-form readout sustituye finetune en RECALL",
          "evidence": {**cert, "acc": res["acc"], "ruta": "lab/runs/m2b_results.json"},
          "benchmark_version": "wspx-RECALL + lab-m2b",
          "candidate": "closed-form-readout", "placebo": "norm-matched-random",
          "surgery_tier": tier, "tier_inferred": True,
          "claim_class": "causal_learn", "placebo_required": True,
          "conclusion": "REJECTED" if tier == "UNREACHABLE" else "SUPPORTED",
          "veredicto": "discard" if tier == "UNREACHABLE" else "keep"})
errs = validate(d)
assert not errs, errs
outp = R / "lab" / "runs" / "records" / "e2026.117.m2b-readout-recall.json"
outp.write_text(json.dumps(d, indent=1), encoding="utf-8")
res["record"] = str(outp)
print(json.dumps(res, indent=2))
(OUT / "m2b_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("guardado m2b_results.json + record:", tier, d["conclusion"])
