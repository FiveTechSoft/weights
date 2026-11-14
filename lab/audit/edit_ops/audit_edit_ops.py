"""Audit of WSPLang weight-modifying ops (engine/wsplang.py execute_plan_v4).

Runs every op on COPIES of small models in a temp dir and measures the effect from the
files on disk (before/after), not from what the op reports. Never touches originals:
hashes of the source models are checked at the end.

Run:  C:\\Users\\Anto\\.venvs\\weights-cuda\\Scripts\\python.exe lab/audit/edit_ops/audit_edit_ops.py
Output: lab/audit/edit_ops/results.json
"""
import hashlib
import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from engine.wsplang import (parse_args_lines, validate, plan_run, execute_plan_v4,  # noqa: E402
                            check_feasibility)
from engine import wsp_arch  # noqa: E402

OUT = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="wsp_edit_audit_"))
ALS = ROOT / "lab/runs/als_adam_small.pt"
NANOWSP = ROOT / "models/nanowsp/model.safetensors"
NANOWSP3 = ROOT / "lab/runs/nanowsp3_pretrained.safetensors"
PROTECT = [ALS, NANOWSP, NANOWSP3, ROOT / "models/nanowsp/model_baseline.safetensors",
           ROOT / "models/nanowsp2/model.safetensors", ROOT / "data/nanowsp3_weights.safetensors",
           ROOT / "data/nanowsp2_weights.safetensors"]
PRE = {}
OVR = dict(block=64, layers=2, heads=4, d=64, ff=256, dropout=0.0, bias=False, vocab=65)
torch.manual_seed(0)
RESULTS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] if Path(p).exists() else None


PRE.update({str(p): sha(p) for p in PROTECT})


def P(p):
    return Path(p).as_posix()


def copy(src, name):
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "model.safetensors"  # als_adam_small.pt has a .pt suffix -> wsplang treats it as non-safetensors
    shutil.copy2(src, dst)
    return dst


def snap(p):
    return {k: v.float().clone() for k, v in load_file(str(p)).items()}


def diff(a, b):
    out = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out[k] = "added" if k not in a else "removed"
            continue
        if a[k].shape != b[k].shape:
            out[k] = f"shape {tuple(a[k].shape)}->{tuple(b[k].shape)}"
            continue
        n = int((a[k] != b[k]).sum())
        if n:
            out[k] = {"changed": n, "of": a[k].numel(), "max_abs": float((a[k] - b[k]).abs().max())}
    return out


def run(steps, model, out_dir=None, allow_write=True, budget=None, dry=False, extra=""):
    code = "experiment audit:\nmodel: nanoGPT\nbudgets:\n"
    if budget is not None:
        code += f"    max_modified_weights = {budget}\n"
    code += f"    allow_write = {str(allow_write).lower()}\n{extra}steps:\n"
    code += "".join(f"    step {s}\n" for s in steps)
    prog = parse_args_lines(code)
    st, err = validate(prog)
    feas_err, feas_warn = check_feasibility(prog, st, str(model) if model else None)
    res = execute_plan_v4(plan_run(prog, st), prog, model_path=str(model) if model else None,
                          out_dir=str(out_dir) if out_dir else None, dry_run=dry)
    return {"validate_errors": err, "feasibility_errors": feas_err, "feasibility_warnings": feas_warn,
            "results": [{k: v for k, v in r.items() if k != "args"} for r in res]}


# ---------------------------------------------------------------- behavior helpers
_TXT = (ROOT / "lab/data/shakespeare_char/input.txt").read_text(encoding="utf-8")
with safe_open(str(ALS), "pt") as f:
    CHARS = json.loads(f.metadata()["vocab_chars"])
STOI = {c: i for i, c in enumerate(CHARS)}
VAL = torch.tensor([STOI[c] for c in _TXT[int(0.9 * len(_TXT)):]], dtype=torch.long)
g = torch.Generator().manual_seed(123)
IX = torch.randint(len(VAL) - 65, (64,), generator=g)
XB = torch.stack([VAL[i:i + 64] for i in IX])
YB = torch.stack([VAL[i + 1:i + 65] for i in IX])


def nanogpt_loss(path):
    m = wsp_arch.load_arch_file(ROOT / "programs/arch/nanogpt.wsparch", overrides=OVR, seed=1337)
    try:
        m.load(path)
    except Exception as e:  # noqa: BLE001
        sd = load_file(str(path))
        # bypass the tie check to still measure behavior
        try:
            m.load_state_dict(sd, strict=False)
        except Exception as e2:  # noqa: BLE001
            return {"load_error": str(e), "fallback_error": str(e2)}
        m.eval()
        with torch.no_grad():
            _, loss = m(XB, YB)
        return {"load_error": str(e)[:160], "loss_ignoring_tie": round(float(loss), 4)}
    m.eval()
    with torch.no_grad():
        _, loss = m(XB, YB)
    return {"loss": round(float(loss), 4)}


def nanowsp_probs(path, pairs):
    from engine.nanowsp_runtime import NanoWSP
    m = NanoWSP()
    m.load_checkpoint(path)
    out = []
    for p, t in pairs:
        ids = m.tokenizer.encode(p, add_bos=True)
        logits, _ = m.forward(ids)
        pr = F.softmax(logits[0, -1], -1)
        tid = m.tokenizer.encode(t, add_bos=False)[0]
        out.append({"prompt": p, "target": t, "p_target": round(float(pr[tid]), 5),
                    "argmax_is_target": int(pr.argmax()) == tid,
                    "argmax": m.tokenizer.decode([int(pr.argmax())])})
    return out


def nanowsp_argmax(path, prompts):
    from engine.nanowsp_runtime import NanoWSP
    m = NanoWSP()
    m.load_checkpoint(path)
    return [int(m.forward(m.tokenizer.encode(p, add_bos=True))[0][0, -1].argmax()) for p in prompts]


def save_vec(name, v):
    p = TMP / f"{name}.npz"
    np.savez(p, u=np.asarray(v, dtype=np.float32))
    return p


def energy(W, u):
    u = torch.as_tensor(u).float()
    u = u / u.norm()
    return round(float((u @ W).norm()), 6)


def case(name):
    def deco(fn):
        try:
            RESULTS[name] = fn()
        except Exception as e:  # noqa: BLE001
            RESULTS[name] = {"harness_exception": repr(e), "tb": traceback.format_exc()[-1500:]}
        print(f"[{name}] done", flush=True)
        return fn
    return deco


TENS = "transformer.h.0.mlp.c_proj.weight"  # [64, 256], u lives in R^64
W0 = snap(ALS)[TENS]
U_TOP = torch.linalg.svd(W0)[0][:, 0].numpy()  # top left singular vector: high energy direction
U_PATH = save_vec("u_top", U_TOP)
BASE_LOSS = nanogpt_loss(ALS)
RESULTS["_baseline"] = {"als_loss": BASE_LOSS, "tmp": str(TMP), "energy_u_top": energy(W0, U_TOP)}


# ---------------------------------------------------------------- unimplemented ops
@case("unimplemented_ops")
def _():
    m = copy(ALS, "unimpl")
    h = sha(m)
    out = {}
    for s in [f"poke {TENS} 0 0 1.0", f"poke-weight {TENS} 0 0 1.0", f"intervene project_out tensor={TENS}",
              "rollback", f"distributed_memit tensor={TENS} prompt=ab target=c"]:
        r = run([s], m, out_dir=TMP / "unimpl_out")
        out[s.split()[0]] = {"status": r["results"][0]["status"], "feasibility_errors": [e[:90] for e in r["feasibility_errors"]]}
    # intervene section with a function-call expression
    code = ("experiment audit:\nmodel: nanoGPT\nbudgets:\n    allow_write = true\nintervene:\n"
            f"    project_out(tensor={TENS}, vector={P(U_PATH)}, alpha=1.0)\n")
    prog = parse_args_lines(code)
    st, err = validate(prog)
    res = execute_plan_v4(plan_run(prog, st), prog, model_path=str(m), out_dir=str(TMP / "unimpl_out"))
    out["intervene_section"] = {"validate_errors": err, "parsed_steps": len(st), "executed_steps": len(res)}
    out["model_unchanged"] = sha(m) == h
    out["out_dir_files"] = [p.name for p in (TMP / "unimpl_out").glob("*")] if (TMP / "unimpl_out").exists() else []
    return out


# ---------------------------------------------------------------- project_out / project_in
@case("project_out")
def _():
    out = {}
    m = copy(ALS, "po")
    od = TMP / "po_out"
    r = run([f"project_out {TENS} vector={P(U_PATH)} alpha=1.0 top_cols=1.0 top_rows=0"], m, out_dir=od)
    f = od / m.name
    out["full_rank1"] = {"reported": {k: r["results"][0].get("result", {}).get(k) for k in
                                      ("modified_parameters", "directional_energy_before", "directional_energy_after", "written_to_disk")},
                         "status": r["results"][0]["status"], "feas": r["feasibility_errors"],
                         "measured_energy_after": energy(snap(f)[TENS], U_TOP) if f.exists() else None,
                         "diff": diff(snap(m), snap(f)) if f.exists() else None,
                         "loss": nanogpt_loss(f) if f.exists() else None}
    # defaults (alpha=0.05, top_cols=0.127, top_rows=50)
    od2 = TMP / "po_out2"
    r = run([f"project_out {TENS} vector={P(U_PATH)}"], m, out_dir=od2)
    f2 = od2 / m.name
    out["defaults"] = {"reported_mod": r["results"][0].get("result", {}).get("modified_parameters"),
                       "reported_red": r["results"][0].get("result", {}).get("energy_reduction_pct"),
                       "measured_energy_before": energy(W0, U_TOP),
                       "measured_energy_after": energy(snap(f2)[TENS], U_TOP),
                       "diff": diff(snap(m), snap(f2))}
    # budget
    od3 = TMP / "po_budget"
    r = run([f"project_out {TENS} vector={P(U_PATH)} alpha=1.0 top_cols=1.0 top_rows=0"], m, out_dir=od3, budget=100)
    out["budget_100"] = {"status": r["results"][0]["status"], "error": r["results"][0].get("error"),
                         "file_written": (od3 / m.name).exists()}
    # allow_write false / dry run / no out_dir
    r = run([f"project_out {TENS} vector={P(U_PATH)} alpha=1.0"], m, out_dir=TMP / "po_nw", allow_write=False)
    out["allow_write_false"] = {"state": r["results"][0].get("execution_state"), "file": (TMP / "po_nw" / m.name).exists()}
    r = run([f"project_out {TENS} vector={P(U_PATH)} alpha=1.0"], m, out_dir=TMP / "po_dry", dry=True)
    out["dry_run"] = {"state": r["results"][0].get("execution_state"), "file": (TMP / "po_dry" / m.name).exists()}
    h = sha(m)
    r = run([f"project_out {TENS} vector={P(U_PATH)} alpha=1.0"], m, out_dir=None)
    out["no_out_dir"] = {"status": r["results"][0]["status"], "state": r["results"][0].get("execution_state"),
                         "input_unchanged": sha(m) == h}
    # forbidden
    r = run([f"project_out lm_head.weight vector={P(save_vec('u65', np.random.randn(65)))} alpha=1.0"], m, out_dir=TMP / "po_f1")
    out["forbidden_lm_head"] = {"status": r["results"][0]["status"]}
    od4 = TMP / "po_wte"
    r = run([f"project_out transformer.wte.weight vector={P(TMP / 'u65.npz')} alpha=1.0 top_cols=1.0 top_rows=0"], m, out_dir=od4)
    out["tied_embedding_wte"] = {"status": r["results"][0]["status"], "written": (od4 / m.name).exists(),
                                 "diff": diff(snap(m), snap(od4 / m.name)) if (od4 / m.name).exists() else None,
                                 "loss": nanogpt_loss(od4 / m.name) if (od4 / m.name).exists() else None}
    r = run([f"project_out transformer.ln_f.weight vector={P(U_PATH)} alpha=1.0"], m, out_dir=TMP / "po_ln")
    out["norm_ln_f"] = {"status": r["results"][0]["status"], "error": (r["results"][0].get("error") or "")[:120]}
    # project_in
    od5 = TMP / "pi_out"
    r = run([f"project_in {TENS} vector={P(U_PATH)} alpha=0.5 top_cols=1.0 top_rows=0"], m, out_dir=od5)
    out["project_in_alpha0.5"] = {"status": r["results"][0]["status"], "energy_before": energy(W0, U_TOP),
                                  "energy_after": energy(snap(od5 / m.name)[TENS], U_TOP),
                                  "loss": nanogpt_loss(od5 / m.name)}
    return out


# ---------------------------------------------------------------- delta_unlearn
@case("delta_unlearn")
def _():
    out = {}
    m = copy(ALS, "du")
    od = TMP / "du_out"
    r = run([f"delta_unlearn {TENS} key={P(U_PATH)} eta=1.0 target=0"], m, out_dir=od)
    f = od / m.name
    rr = r["results"][0]
    out["eta1"] = {"status": rr["status"], "reported_after": rr.get("result", {}).get("energy_after"),
                   "measured_before": energy(W0, U_TOP),
                   "measured_after": energy(snap(f)[TENS], U_TOP) if f.exists() else None,
                   "diff": diff(snap(m), snap(f)) if f.exists() else None, "loss": nanogpt_loss(f) if f.exists() else None}
    r = run([f"delta_unlearn {TENS} key={P(U_PATH)} eta=1.0"], m, out_dir=TMP / "du_b", budget=10)
    out["budget_10"] = {"status": r["results"][0]["status"], "error": r["results"][0].get("error"),
                        "file": (TMP / "du_b" / m.name).exists()}
    r = run([f"delta_unlearn lm_head.weight key={P(TMP / 'u65.npz')}"], m, out_dir=TMP / "du_f")
    out["forbidden"] = r["results"][0]["status"]
    # nonzero target: does k^T W hit target?
    od2 = TMP / "du_t"
    r = run([f"delta_unlearn {TENS} key={P(U_PATH)} eta=1.0 target=0.5"], m, out_dir=od2)
    W2 = snap(od2 / m.name)[TENS]
    u = torch.tensor(U_TOP).float()
    out["target_0.5"] = {"measured_kTW_mean": round(float((u @ W2).mean()), 5),
                         "measured_kTW_std": round(float((u @ W2).std()), 6)}
    return out


# ---------------------------------------------------------------- leace_erase
@case("leace_erase")
def _():
    out = {}
    m = copy(ALS, "le")
    h0 = sha(m)
    before = snap(m)
    r = run([f"leace_erase {TENS} vector={P(U_PATH)}"], m, out_dir=None, budget=1)
    rr = r["results"][0]
    after = snap(m)
    out["no_out_dir_budget1"] = {"status": rr["status"], "state": rr.get("execution_state"),
                                 "INPUT_FILE_OVERWRITTEN": sha(m) != h0,
                                 "measured_energy_after": energy(after[TENS], U_TOP),
                                 "diff": diff(before, after), "loss": nanogpt_loss(m),
                                 "feas_warn": r["feasibility_warnings"]}
    m2 = copy(ALS, "le2")
    r = run([f"leace_erase lm_head.weight vector={P(TMP / 'u65.npz')}"], m2, out_dir=TMP / "le2_out")
    f = TMP / "le2_out" / m2.name
    out["forbidden_lm_head"] = {"status": r["results"][0]["status"], "written": f.exists(),
                                "diff": diff(snap(m2), snap(f)) if f.exists() else None,
                                "loss": nanogpt_loss(f) if f.exists() else None}
    return out


# ---------------------------------------------------------------- spectral_decay
@case("spectral_decay")
def _():
    out = {}
    m = copy(ALS, "sd")
    od = TMP / "sd_out"
    r = run([f"spectral_decay {TENS} decay=0.9"], m, out_dir=od, budget=10)
    f = od / m.name
    a, b = snap(m), snap(f) if f.exists() else None
    out["decay0.9_budget10"] = {"status": r["results"][0]["status"], "written": f.exists(),
                                "max_ratio_err": float(((b[TENS] - 0.9 * a[TENS]).abs().max())) if b else None,
                                "diff": diff(a, b) if b else None, "loss": nanogpt_loss(f) if b else None}
    r = run(["spectral_decay lm_head.weight decay=0.5"], m, out_dir=TMP / "sd_f")
    f2 = TMP / "sd_f" / m.name
    out["forbidden_lm_head"] = {"status": r["results"][0]["status"], "written": f2.exists(),
                                "diff": diff(a, snap(f2)) if f2.exists() else None}
    return out


# ---------------------------------------------------------------- decoupled_delta
@case("decoupled_delta")
def _():
    m = copy(ALS, "dd")
    k = np.random.RandomState(1).randn(256)
    v = np.random.RandomState(2).randn(64)
    kp, vp = save_vec("k256", k), save_vec("v64", v)
    od = TMP / "dd_out"
    r = run([f"decoupled_delta {TENS} erase_vector={P(U_PATH)} write_key={P(kp)} write_value={P(vp)} beta_erase=1.0 beta_write=1.0"],
            m, out_dir=od, budget=1)
    f = od / m.name
    W1 = snap(f)[TENS]
    u = torch.tensor(U_TOP).float(); u /= u.norm()
    kk = torch.tensor(k).float(); kk /= kk.norm()
    vv = torch.tensor(v).float(); vv /= vv.norm()
    vo = vv - (u @ vv) * u
    expected = W0 - torch.outer(u, u @ W0) + torch.outer(vo, kk)
    return {"status": r["results"][0]["status"], "reported": r["results"][0].get("result"),
            "measured_energy_after": energy(W1, U_TOP), "max_err_vs_formula": float((W1 - expected).abs().max()),
            "write_readback_cos(W1 k, v_ortho)": round(float(F.cosine_similarity((W1 @ kk)[None], vo[None])), 4),
            "diff": diff(snap(m), snap(f)), "loss": nanogpt_loss(f)}


# ---------------------------------------------------------------- jepa_steer
@case("jepa_steer")
def _():
    m = copy(ALS, "js")
    tl = save_vec("tl64", np.random.RandomState(3).randn(64))
    od = TMP / "js_out"
    r = run([f"jepa_steer {TENS} target_latent={P(tl)} lambda=0.1"], m, out_dir=od, budget=1)
    f = od / m.name
    # does the edit steer REAL activations (inputs of c_proj) toward the latent?
    mdl = wsp_arch.load_arch_file(ROOT / "programs/arch/nanogpt.wsparch", overrides=OVR, seed=1337)
    mdl.load(ALS)
    with torch.no_grad():
        H = mdl.input_of("transformer.h.0.mlp.c_proj", XB[:8]).reshape(-1, 256)
    t = torch.tensor(np.load(tl)["u"]).float(); t /= t.norm()
    W1 = snap(f)[TENS]
    cos0 = float(F.cosine_similarity(H @ W0.t(), t[None], dim=1).mean())
    cos1 = float(F.cosine_similarity(H @ W1.t(), t[None], dim=1).mean())
    return {"status": r["results"][0]["status"], "reported_cos": r["results"][0].get("result", {}).get("latent_alignment_cosine"),
            "real_activation_cos_before": round(cos0, 4), "real_activation_cos_after": round(cos1, 4),
            "diff": diff(snap(m), snap(f)), "loss": nanogpt_loss(f)}


# ---------------------------------------------------------------- permute_align
@case("permute_align")
def _():
    m = copy(ALS, "pa")
    T2 = "transformer.h.0.mlp.c_fc.weight"  # [256, 64]
    W = snap(m)[T2]
    perm = torch.randperm(256, generator=torch.Generator().manual_seed(7))
    ref = TMP / "pa_ref.npz"
    np.savez(ref, w=W[perm].numpy())  # target rows = source rows permuted
    od = TMP / "pa_out"
    r = run([f"permute_align {T2} target_reference={P(ref)} mode=rows"], m, out_dir=od, budget=1)
    f = od / m.name
    Wa = snap(f)[T2]
    Wt = W[perm]
    return {"status": r["results"][0]["status"], "reported": {k: r["results"][0].get("result", {}).get(k) for k in
                                                              ("similarity_before_permutation", "similarity_after_permutation")},
            "measured_rows_matching_reference": int((Wa == Wt).all(dim=1).sum()),
            "measured_mean_rowcos_aligned_vs_ref": round(float(F.cosine_similarity(Wa, Wt, dim=1).mean()), 4),
            "rows_unchanged_vs_source": int((Wa == W).all(dim=1).sum()),
            "diff": diff(snap(m), snap(f)), "loss_after_single_tensor_permute": nanogpt_loss(f)}


# ---------------------------------------------------------------- tie_weights
@case("tie_weights")
def _():
    out = {}
    m = copy(ALS, "tw")
    h = sha(m)
    a = snap(m)
    r = run(["tie_weights source=transformer.h.0.attn.c_proj.weight target=transformer.h.1.attn.c_proj.weight"],
            m, out_dir=TMP / "tw_out", budget=1)
    out["out_dir_ignored"] = {"status": r["results"][0]["status"], "INPUT_FILE_OVERWRITTEN": sha(m) != h,
                              "out_dir_written": (TMP / "tw_out").exists(), "diff": diff(a, snap(m)), "loss": nanogpt_loss(m)}
    m2 = copy(ALS, "tw2")
    a2 = snap(m2)
    r = run(["tie_weights source=transformer.h.0.mlp.c_proj.weight target=lm_head.weight"], m2)
    out["forbidden_target_lm_head"] = {"status": r["results"][0]["status"], "error": (r["results"][0].get("error") or "")[:150],
                                       "diff": diff(a2, snap(m2))}
    m3 = copy(ALS, "tw3")
    a3 = snap(m3)
    r = run(["tie_weights source=transformer.ln_f.weight target=new.tensor"], m3)
    out["creates_new_tensor"] = {"status": r["results"][0]["status"], "diff": diff(a3, snap(m3))}
    m4 = copy(ALS, "tw4")
    h4 = sha(m4)
    r = run(["tie_weights source=transformer.h.0.attn.c_proj.weight target=transformer.h.1.attn.c_proj.weight"], m4, allow_write=False)
    out["allow_write_false"] = {"state": r["results"][0].get("execution_state"), "unchanged": sha(m4) == h4}
    return out


# ---------------------------------------------------------------- imprint_association / imprint_batch / cov_imprint
OTHER = ["the ", "hello wor", "abc", "zz", "Mary had a", "12+3=", "qwerty", "XYZ", "foo bar", "lorem"]


@case("imprint_association")
def _():
    out = {}
    m = copy(NANOWSP, "ia")
    h = sha(m)
    a = snap(m)
    pairs = [("Paris is in", "Q")]
    before = nanowsp_probs(m, pairs)
    am0 = nanowsp_argmax(m, OTHER)
    r = run(['imprint_association prompt="Paris is in" target=Q layer=1 eta=5.0'], m, out_dir=TMP / "ia_out", budget=1)
    out["single_char"] = {"status": r["results"][0]["status"], "INPUT_FILE_OVERWRITTEN": sha(m) != h,
                          "out_dir_written": (TMP / "ia_out").exists(), "before": before,
                          "after": nanowsp_probs(m, pairs), "diff": diff(a, snap(m)),
                          "collateral_argmax_changed": sum(x != y for x, y in zip(am0, nanowsp_argmax(m, OTHER))),
                          "of": len(OTHER)}
    m2 = copy(NANOWSP, "ia2")
    pairs2 = [("capital:", "R"), ("capital:R", "o"), ("capital:Ro", "m"), ("capital:Rom", "e")]
    b2 = nanowsp_probs(m2, pairs2)
    r = run(['imprint_association prompt=capital: target=Rome layer=1 eta=5.0'], m2)
    out["multi_char"] = {"status": r["results"][0]["status"], "before": b2, "after": nanowsp_probs(m2, pairs2)}
    m3 = copy(ALS, "ia3")
    h3 = sha(m3)
    r = run(['imprint_association prompt=abc target=d'], m3)
    out["on_nanogpt_checkpoint"] = {"status": r["results"][0]["status"], "error": (r["results"][0].get("error") or "")[:120],
                                    "unchanged": sha(m3) == h3}
    return out


@case("imprint_batch")
def _():
    m = copy(NANOWSP, "ib")
    pairs = [("hello", "w"), ("abc", "q"), ("xyz", "m"), ("key1", "Z"), ("key2", "Y")]
    b = nanowsp_probs(m, pairs)
    am0 = nanowsp_argmax(m, OTHER)
    a = snap(m)
    spec = ";".join(f"{p}->{t}" for p, t in pairs)
    r = run([f"imprint_batch pairs={spec} layer=1 eta=5.0"], m, budget=1)
    aft = nanowsp_probs(m, pairs)
    return {"status": r["results"][0]["status"], "reported": r["results"][0].get("result"),
            "recall_before": sum(x["argmax_is_target"] for x in b), "recall_after": sum(x["argmax_is_target"] for x in aft),
            "p_before": [x["p_target"] for x in b], "p_after": [x["p_target"] for x in aft],
            "collateral_argmax_changed": sum(x != y for x, y in zip(am0, nanowsp_argmax(m, OTHER))),
            "diff": diff(a, snap(m))}


@case("cov_imprint")
def _():
    out = {}
    for op in ("cov_imprint", "memit_imprint"):
        m = copy(NANOWSP, op)
        pairs = [("Paris is in", "Q")]
        b = nanowsp_probs(m, pairs)
        a = snap(m)
        r = run([f'{op} layers.1.mlp_out.weight prompt="Paris is in" target=Q'], m, out_dir=TMP / f"{op}_out", budget=1)
        out[op] = {"status": r["results"][0]["status"], "error": r["results"][0].get("error"),
                   "out_dir_written": (TMP / f"{op}_out").exists(),
                   "before": b, "after": nanowsp_probs(m, pairs), "diff": diff(a, snap(m))}
    # file= batch
    m = copy(NANOWSP, "ci_batch")
    pairs = [("hello", "w"), ("abc", "q"), ("xyz", "m"), ("key1", "Z"), ("key2", "Y")]
    bf = TMP / "pairs.json"
    bf.write_text(json.dumps([{"prompt": p, "target": t} for p, t in pairs]))
    b = nanowsp_probs(m, pairs)
    r = run([f"cov_imprint layers.1.mlp_out.weight file={P(bf)}"], m)
    aft = nanowsp_probs(m, pairs)
    out["batch5"] = {"status": r["results"][0]["status"], "recall_before": sum(x["argmax_is_target"] for x in b),
                     "recall_after": sum(x["argmax_is_target"] for x in aft),
                     "p_before": [x["p_target"] for x in b], "p_after": [x["p_target"] for x in aft]}
    # on a nanoGPT checkpoint: what gets written?
    m = copy(ALS, "ci_gpt")
    a = snap(m)
    r = run(["cov_imprint layers.1.mlp_out.weight prompt=abc target=d"], m)
    d = diff(a, snap(m))
    out["on_nanogpt_checkpoint"] = {"status": r["results"][0]["status"], "error": (r["results"][0].get("error") or "")[:120],
                                    "tensors_added": sorted(k for k, v in d.items() if v == "added"),
                                    "n_changed_entries": len(d), "loss": nanogpt_loss(m)}
    return out


# ---------------------------------------------------------------- compile_synthetic
@case("compile_synthetic")
def _():
    from engine.nanowsp3_runtime import NanoWSP3
    from lab.wspx_datasets import recall
    m = copy(NANOWSP3, "cs")
    h = sha(m)
    outp = TMP / "cs_out" / "edited.safetensors"
    r = run([f"compile_synthetic dataset=wspx_recall lambda_base=30 model={P(m)} out={P(outp)}"], m, budget=1)
    rr = r["results"][0]
    ds = recall()

    def acc(path, rows):
        mm = NanoWSP3()
        mm.load_checkpoint(path)
        hits = 0
        for row in rows:
            with torch.no_grad():
                lg, _ = mm.forward([ord(c) for c in row["prompt"]])
            hits += int(lg[0, -1].argmax()) == ord(row["target"])
        return hits / len(rows)
    res = {"status": rr["status"], "error": rr.get("error"), "input_unchanged": sha(m) == h,
           "reported": {k: rr.get("result", {}).get(k) for k in ("train_accuracy_before", "train_accuracy_after",
                                                                 "holdout_accuracy_before", "holdout_accuracy_after", "weights_written")}}
    if outp.exists():
        res["diff"] = diff(snap(m), snap(outp))
        res["measured_train_before"] = acc(m, ds["train"])
        res["measured_train_after"] = acc(outp, ds["train"])
        res["measured_holdout_after"] = acc(outp, ds["holdout"])
    return res


# ---------------------------------------------------------------- rosetta_align / cross_project
@case("rosetta_cross_project")
def _():
    rng = np.random.RandomState(5)
    Es = rng.randn(400, 32).astype(np.float32)
    Q, _ = np.linalg.qr(rng.randn(48, 32))  # [48,32] orthonormal columns
    Et = (Es @ Q.T).astype(np.float32)
    sp, tp = TMP / "Es.npz", TMP / "Et.npz"
    np.savez(sp, e=Es); np.savez(tp, e=Et)
    rp = TMP / "rosetta.npz"
    r = run([f"rosetta_align source={P(sp)} target={P(tp)} out={P(rp)}"], None)
    T = np.load(rp)["T"] if rp.exists() else None
    u = rng.randn(32).astype(np.float32)
    up = save_vec("u32", u)
    xp = TMP / "xlated.npz"
    r2 = run([f"cross_project vector={P(up)} rosetta={P(rp)} out={P(xp)}"], None)
    ut = np.load(xp)["u"] if xp.exists() else None
    exp = Q @ (u / np.linalg.norm(u)); exp /= np.linalg.norm(exp)
    rp_nw = TMP / "rosetta_nw.npz"
    r3 = run([f"rosetta_align source={P(sp)} target={P(tp)} out={P(rp_nw)}"], None, allow_write=False)
    return {"rosetta_status": r["results"][0]["status"], "feas": r["feasibility_errors"],
            "reported_cos": r["results"][0].get("result", {}).get("mean_anchor_cosine"),
            "max_abs_T_minus_Q": float(np.abs(T - Q).max()) if T is not None else None,
            "cross_status": r2["results"][0]["status"],
            "cos_translated_vs_truth": float(ut @ exp) if ut is not None else None,
            "allow_write_false_file": rp_nw.exists(), "modifies_weights": False}


# ---------------------------------------------------------------- export_control_vector / layer_envelope
def gguf_tensors(p):
    import gguf
    rd = gguf.GGUFReader(str(p))
    return {t.name: np.array(t.data, dtype=np.float32) for t in rd.tensors}


@case("export_control_vector")
def _():
    v = np.random.RandomState(9).randn(64)
    vp = save_vec("cv64", v)
    gp = TMP / "cv.gguf"
    r = run([f"export_control_vector {P(vp)} out={P(gp)} scale=2.0"], None, allow_write=False, dry=True)
    t = gguf_tensors(gp) if gp.exists() else {}
    names = sorted(t, key=lambda s: int(s.split(".")[1]))
    vn = v / np.linalg.norm(v)
    return {"status": r["results"][0]["status"], "written_despite_allow_write_false_and_dry_run": gp.exists(),
            "n_tensors": len(t), "first_last": [names[0], names[-1]] if names else None,
            "values_ok": bool(np.allclose(t[names[0]], 2.0 * vn, atol=1e-6)) if names else None,
            "llama_common_dll_string": "invalid (zero) direction tensor layer index in %s"}


@case("layer_envelope")
def _():
    vp = TMP / "cv64.npz"
    out = {}
    files = {}
    for alpha in ("0.04", "4.0"):
        gp = TMP / f"env_{alpha}.gguf"
        r = run([f"layer_envelope layers=0..3 peak=2 bandwidth=1.0 alpha={alpha} vector={P(vp)} out={P(gp)} n_layers=4"],
                None, allow_write=False)
        files[alpha] = gguf_tensors(gp) if gp.exists() else None
        out[f"alpha_{alpha}"] = {"status": r["results"][0]["status"], "feas": r["feasibility_errors"][:1],
                                 "written": gp.exists(), "weights": r["results"][0].get("result", {}).get("weights_distribution")}
    if files["0.04"] and files["4.0"]:
        out["alpha_changes_output"] = any(not np.array_equal(files["0.04"][k], files["4.0"][k]) for k in files["0.04"])
        out["tensor_names"] = sorted(files["0.04"])
        out["norms"] = {k: round(float(np.linalg.norm(v)), 4) for k, v in files["0.04"].items()}
    # failure path of the GGUF writer (exception inside the try): make GGUFWriter raise
    import gguf
    orig = gguf.GGUFWriter

    def boom(*a, **k):
        raise RuntimeError("simulated writer failure")
    gguf.GGUFWriter = boom
    try:
        r = run([f"multi_layer_envelope layers=0..3 vector={P(vp)} out={P(TMP / 'env_fail.gguf')}"], None)
    finally:
        gguf.GGUFWriter = orig
    out["writer_failure_path"] = {"status": r["results"][0]["status"], "error": r["results"][0].get("error")}
    # pre-existing repo artifacts: do they carry the invalid direction.0 too?
    for p in (ROOT / "data/loop_control_vector.gguf", ROOT / "data/u_critic_control_vector.gguf"):
        if p.exists():
            out[f"repo_artifact_{p.name}"] = sorted(gguf_tensors(p), key=lambda s: int(s.split(".")[1]))[:2]
    return out


# ---------------------------------------------------------------- fwp_overlay / surprise_fwp / vicreg_preserve
@case("fwp_overlay")
def _():
    m = copy(ALS, "fwp")
    h = sha(m)
    op_ = TMP / "fwp.npz"
    r = run([f"fwp_overlay {TENS} vector={P(U_PATH)} alpha=0.5 out={P(op_)}"], m, allow_write=False, dry=True)
    z = np.load(op_, allow_pickle=True) if op_.exists() else None
    return {"status": r["results"][0]["status"], "written_despite_dry_run": op_.exists(),
            "npz_keys": sorted(z.files) if z is not None else None, "model_unchanged": sha(m) == h,
            "is_active_inference_hook": r["results"][0].get("result", {}).get("is_active_inference_hook"),
            "grep_consumers": "no reader of FastWeightOverlay npz in engine/ (see report)"}


@case("surprise_fwp")
def _():
    r = run(['surprise_fwp prompt="the loop the loop the loop the loop" threshold=0.3'], None)
    return {"status": r["results"][0]["status"], "result": r["results"][0].get("result"),
            "feas_warn": r["feasibility_warnings"]}


@case("vicreg_preserve")
def _():
    m = copy(ALS, "vr")
    h = sha(m)
    r = run([f"vicreg_preserve {TENS}"], m, out_dir=TMP / "vr_out")
    return {"status": r["results"][0]["status"], "model_unchanged": sha(m) == h,
            "result": {k: r["results"][0].get("result", {}).get(k) for k in ("effective_rank", "vicreg_certified")}}


# ---------------------------------------------------------------- originals untouched
RESULTS["_originals_unchanged"] = {str(p): PRE[str(p)] == sha(p) for p in PROTECT}
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str), encoding="utf-8")
print("wrote", OUT / "results.json", "tmp:", TMP)
