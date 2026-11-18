"""Builds the arm table (lab/probes/weight2bench/PLAN.md), computes every weight-only feature and paired target, and
the readings R1-R3. Writes lab/runs/weight2bench/arms.json and summary.json. CPU, no forward pass.

  python lab/probes/weight2bench/run.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lab.probes.weight2bench import features as F  # noqa: E402

OUT = ROOT / "lab" / "runs" / "weight2bench"

FAMILIES = {
    "q9": dict(hf="hf_cache/qwen35-9b-hf", head="lm_head.weight", norm="model.language_model.norm.weight",
               close=248069, open=248068, own="hf_cache/cc/cc_v2.gguf"),
    "phi": dict(hf="hf_cache/phi4mr-hf", head="model.embed_tokens.weight", norm="model.norm.weight",
                close=49631, open=881, own="hf_cache/cc_phi/cc_phi.gguf"),
    "llama": dict(hf="hf_cache/r1llama8b-hf", head="lm_head.weight", norm="model.norm.weight",
                  close=128014, open=128013, own="hf_cache/cc_llama/cc_llama.gguf"),
    "q08": dict(hf="hf_cache/qwen35-08b-hf", head="model.language_model.embed_tokens.weight",
                norm="model.language_model.norm.weight", close=248069, open=248068,
                own="hf_cache/act_map/cvec08/cc_own.gguf"),
}

R = "lab/runs"


def A(id, fam, kind, art, scale, arm, base, label, set_, auc_row, placebo=None, circular=False, layers=None):
    return dict(id=id, family=fam, kind=kind, artefact=art, scale=scale, arm=f"{R}/{arm}.json", base=f"{R}/{base}.json",
                label=label, set=set_, auc_row=auc_row, placebo=placebo, circular=circular, layers=layers)


def arms():
    rows = []
    # ---- q9
    rows += [A("q9_vec_dev", "q9", "vec", "hf_cache/cc/cc_v2.gguf", 1.0, "cc_v2/dev_vec", "cc_v2/dev_base", "real", "dev", True,
               "hf_cache/cc/cc_v2.placebo2026.gguf", circular=True),
             A("q9_pl_dev", "q9", "vec", "hf_cache/cc/cc_v2.placebo2026.gguf", 1.0, "cc_v2/dev_placebo", "cc_v2/dev_base", "placebo", "dev", True),
             A("q9_no31_dev", "q9", "vec", "hf_cache/cc/cc_no31.gguf", 1.0, "cc_v2/dev_no31", "cc_v2/dev_base", "variant", "dev", True, circular=True),
             A("q9_comp_dev", "q9", "compiled", "hf_cache/cc_op_v2/vector/fit.pt", 1.0, "cc_v2/weights/dev_merged", "cc_v2/dev_base", "real", "dev", True,
               "hf_cache/cc_op_v2/placebo/fit.pt", circular=True),
             A("q9_comp_pl_dev", "q9", "compiled", "hf_cache/cc_op_v2/placebo/fit.pt", 1.0, "cc_v2/weights/dev_placebo_merged", "cc_v2/dev_base", "placebo", "dev", True),
             A("q9_v1_dev", "q9", "vec", "hf_cache/cc/cc.gguf", 1.0, "cc/dev_vec", "cc/dev_base", "real", "dev_v1", True, "hf_cache/cc/cc.placebo2026.gguf"),
             A("q9_v1pl_dev", "q9", "vec", "hf_cache/cc/cc.placebo2026.gguf", 1.0, "cc/dev_placebo", "cc/dev_base", "placebo", "dev_v1", True)]
    for sc in ("0.25", "0.5", "1.0", "2.0"):
        rows.append(A(f"q9_vec_screen_{sc}", "q9", "vec", "hf_cache/cc/cc_v2.gguf", float(sc), f"cc_v2/screen_s{sc}", "cc_v2/screen_base", "dose", "screen", False, circular=True))
    for k in range(1, 9):
        rows.append(A(f"q9_pl{k}_screen", "q9", "vec", f"hf_cache/cc/cc_v2.placebo{k}.gguf", 1.0, f"cc_v2/screen_pl{k}", "cc_v2/screen_base", "placebo", "screen", True))
    rows += [A("q9_vec_sealed", "q9", "vec", "hf_cache/cc/cc_v2.gguf", 1.0, "sealed_cc_v2/vec", "sealed_cc_v2/base", "real", "sealed", False, circular=True),
             A("q9_pl_sealed", "q9", "vec", "hf_cache/cc/cc_v2.placebo2026.gguf", 1.0, "sealed_cc_v2/placebo", "sealed_cc_v2/base", "placebo", "sealed", False),
             A("q9_comp_sealed", "q9", "compiled", "hf_cache/cc_op_v2/vector/fit.pt", 1.0, "sealed_ccw_v2/merged", "sealed_ccw_v2/base", "real", "sealed2", False, circular=True),
             A("q9_comp_pl_sealed", "q9", "compiled", "hf_cache/cc_op_v2/placebo/fit.pt", 1.0, "sealed_ccw_v2/placebo_merged", "sealed_ccw_v2/base", "placebo", "sealed2", False),
             A("q9_vec_sealed2", "q9", "vec", "hf_cache/cc/cc_v2.gguf", 1.0, "sealed_ccw_v2/vec", "sealed_ccw_v2/base", "real", "sealed2", False, circular=True)]
    # ---- phi (vector served at 0.5)
    rows += [A("phi_vec_dev", "phi", "vec", "hf_cache/cc_phi/cc_phi.gguf", 0.5, "cc_phi/dev_vec", "cc_phi/dev_base", "real", "dev", True,
               "hf_cache/cc_phi/cc_phi.placebo2026.gguf", circular=True),
             A("phi_pl_dev", "phi", "vec", "hf_cache/cc_phi/cc_phi.placebo2026.gguf", 0.5, "cc_phi/dev_placebo", "cc_phi/dev_base", "placebo", "dev", True),
             A("phi_comp_dev", "phi", "compiled", "hf_cache/cc_phi_op/vector/fit.pt", 1.0, "cc_phi/weights/dev_merged", "cc_phi/dev_base", "real", "dev", True,
               "hf_cache/cc_phi_op/placebo/fit.pt", circular=True),
             A("phi_comp_pl_dev", "phi", "compiled", "hf_cache/cc_phi_op/placebo/fit.pt", 1.0, "cc_phi/weights/dev_placebo_merged", "cc_phi/dev_base", "placebo", "dev", True)]
    for sc in ("0.25", "0.5", "1.0", "2.0"):
        rows.append(A(f"phi_vec_screen_{sc}", "phi", "vec", "hf_cache/cc_phi/cc_phi.gguf", float(sc), f"cc_phi/screen_s{sc}", "cc_phi/screen_base", "dose", "screen", False, circular=True))
    rows += [A("phi_vec_sealed", "phi", "vec", "hf_cache/cc_phi/cc_phi.gguf", 0.5, "sealed_cc_phi/vec", "sealed_cc_phi/base", "real", "sealed", False, circular=True),
             A("phi_pl_sealed", "phi", "vec", "hf_cache/cc_phi/cc_phi.placebo2026.gguf", 0.5, "sealed_cc_phi/placebo", "sealed_cc_phi/base", "placebo", "sealed", False),
             A("phi_comp_sealed", "phi", "compiled", "hf_cache/cc_phi_op/vector/fit.pt", 1.0, "sealed_ccw_phi/merged", "sealed_ccw_phi/base", "real", "sealed2", False, circular=True),
             A("phi_comp_pl_sealed", "phi", "compiled", "hf_cache/cc_phi_op/placebo/fit.pt", 1.0, "sealed_ccw_phi/placebo_merged", "sealed_ccw_phi/base", "placebo", "sealed2", False),
             A("phi_vec_sealed2", "phi", "vec", "hf_cache/cc_phi/cc_phi.gguf", 0.5, "sealed_ccw_phi/vec", "sealed_ccw_phi/base", "real", "sealed2", False, circular=True)]
    # ---- llama (vector served at 0.5; dose 0.25 / 0.35; sealed 3 at 0.35)
    rows += [A("llama_vec_dev", "llama", "vec", "hf_cache/cc_llama/cc_llama.gguf", 0.5, "cc_llama/dev_vec", "cc_llama/dev_base", "real", "dev", True,
               "hf_cache/cc_llama/cc_llama.placebo2026.gguf", circular=True),
             A("llama_vec_dev_0.25", "llama", "vec", "hf_cache/cc_llama/cc_llama.gguf", 0.25, "cc_llama/dev_vec_s0.25", "cc_llama/dev_base", "dose", "dev", False, circular=True),
             A("llama_vec_dev_0.35", "llama", "vec", "hf_cache/cc_llama/cc_llama.gguf", 0.35, "cc_llama/dev_vec_s0.35", "cc_llama/dev_base", "dose", "dev", False, circular=True),
             A("llama_pl_dev", "llama", "vec", "hf_cache/cc_llama/cc_llama.placebo2026.gguf", 0.5, "cc_llama/dev_placebo", "cc_llama/dev_base", "placebo", "dev", True),
             A("llama_comp_dev", "llama", "compiled", "hf_cache/cc_llama_op/vector/fit.pt", 1.0, "cc_llama/weights/dev_merged", "cc_llama/dev_base", "real", "dev", True,
               "hf_cache/cc_llama_op/placebo/fit.pt", circular=True),
             A("llama_comp_pl_dev", "llama", "compiled", "hf_cache/cc_llama_op/placebo/fit.pt", 1.0, "cc_llama/weights/dev_placebo_merged", "cc_llama/dev_base", "placebo", "dev", True)]
    for sc in ("0.25", "0.5", "1.0", "2.0"):
        rows.append(A(f"llama_vec_screen_{sc}", "llama", "vec", "hf_cache/cc_llama/cc_llama.gguf", float(sc), f"cc_llama/screen_s{sc}", "cc_llama/screen_base", "dose", "screen", False, circular=True))
    rows += [A("llama_vec_sealed", "llama", "vec", "hf_cache/cc_llama/cc_llama.gguf", 0.5, "sealed_cc_llama/vec", "sealed_cc_llama/base", "real", "sealed", False, circular=True),
             A("llama_pl_sealed", "llama", "vec", "hf_cache/cc_llama/cc_llama.placebo2026.gguf", 0.5, "sealed_cc_llama/placebo", "sealed_cc_llama/base", "placebo", "sealed", False),
             A("llama_comp_sealed", "llama", "compiled", "hf_cache/cc_llama_op/vector/fit.pt", 1.0, "sealed_ccw_llama/merged", "sealed_ccw_llama/base", "real", "sealed2", False, circular=True),
             A("llama_comp_pl_sealed", "llama", "compiled", "hf_cache/cc_llama_op/placebo/fit.pt", 1.0, "sealed_ccw_llama/placebo_merged", "sealed_ccw_llama/base", "placebo", "sealed2", False),
             A("llama_vec_sealed2", "llama", "vec", "hf_cache/cc_llama/cc_llama.gguf", 0.5, "sealed_ccw_llama/vec", "sealed_ccw_llama/base", "real", "sealed2", False, circular=True),
             A("llama_vec_sealed3", "llama", "vec", "hf_cache/cc_llama/cc_llama.gguf", 0.35, "sealed3_cc_llama/vec", "sealed3_cc_llama/base", "dose", "sealed3", False, circular=True),
             A("llama_pl_sealed3", "llama", "vec", "hf_cache/cc_llama/cc_llama.placebo2026.gguf", 0.35, "sealed3_cc_llama/placebo", "sealed3_cc_llama/base", "placebo", "sealed3", False)]
    # ---- q08 (test 100 GSM8K + 50 MATH; dev 64 + 32)
    C08 = "act_map/cc08"; V08 = "hf_cache/act_map/cvec08"
    rows += [A("q08_own_test", "q08", "vec", f"{V08}/cc_own.gguf", 1.0, f"{C08}/test_own", f"{C08}/test_base", "real", "test", True, circular=True),
             A("q08_port_test", "q08", "vec", f"{V08}/cc_port.gguf", 2.0, f"{C08}/test_port", f"{C08}/test_base", "variant", "test", True, f"{V08}/cc_port.placebo2026.gguf"),
             A("q08_embport_test", "q08", "vec", f"{V08}/cc_embport.gguf", 0.5, f"{C08}/test_embport", f"{C08}/test_base", "variant", "test", True),
             A("q08_portpl_test", "q08", "vec", f"{V08}/cc_port.placebo2026.gguf", 2.0, f"{C08}/test_placebo", f"{C08}/test_base", "placebo", "test", True),
             A("q08_exp8_test", "q08", "cols", "data/exp8_edit_recovered.npz", 1.0, f"{C08}/test_exp8", f"{C08}/test_base", "real", "test", True, "data/exp8_placebo.npz", layers=[15, 19]),
             A("q08_exp8_rerun", "q08", "cols", "data/exp8_edit_recovered.npz", 1.0, "exp8_prune/test_full", f"{C08}/test_base", "real", "test", False, layers=[15, 19]),
             A("q08_exp8pl_test", "q08", "cols", "data/exp8_placebo.npz", 1.0, f"{C08}/test_exp8_placebo", f"{C08}/test_base", "placebo", "test", True, layers=[15, 19]),
             A("q08_d4k_test", "q08", "cols", "data/exp8_d4k.npz", 1.0, "exp8_prune/test_d4k", f"{C08}/test_base", "variant", "test", True, layers=[15, 19]),
             A("q08_d4kc_test", "q08", "cols", "data/exp8_d4k_complement.npz", 1.0, "exp8_prune/test_d4k_comp", f"{C08}/test_base", "variant", "test", True, layers=[15, 19])]
    for arm in ("own", "port", "embport"):
        for sc in ("0.25", "0.5", "1.0", "2.0", "4.0"):
            rows.append(A(f"q08_{arm}_dev_{sc}", "q08", "vec", f"{V08}/cc_{arm}.gguf", float(sc), f"{C08}/dev_{arm}_s{sc}", f"{C08}/dev_base", "dose", "dev", False, circular=(arm == "own")))
    return rows


def load_dirs(row):
    p = ROOT / row["artefact"]
    if row["kind"] == "vec":
        return F.load_cvec(p, row["scale"])
    if row["kind"] == "compiled":
        return F.load_compiled(p, row["scale"])
    return F.load_cols(p, row["layers"], row["scale"])


def readouts(fam):
    from engine import weight_diff as wd
    spec = FAMILIES[fam]
    rd = wd.LocalSafetensors(ROOT / spec["hf"])
    g = rd.get(spec["norm"]).float().numpy()
    rc = rd.get_rows(spec["head"], spec["close"], 1).float().numpy().ravel()
    ro = rd.get_rows(spec["head"], spec["open"], 1).float().numpy().ravel()
    return F.readout(rc, g), F.readout(ro, g)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    ro_cache, own_cache = {}, {}
    missing = []
    for a in arms():
        arm_p, base_p = ROOT / a["arm"], ROOT / a["base"]
        if not (arm_p.exists() and base_p.exists() and (ROOT / a["artefact"]).exists()):
            missing.append((a["id"], [str(x) for x in (arm_p, base_p, ROOT / a["artefact"]) if not x.exists()]))
            continue
        fam = a["family"]
        if fam not in ro_cache:
            ro_cache[fam] = readouts(fam)
            own_cache[fam] = F.load_cvec(ROOT / FAMILIES[fam]["own"], 1.0)
        dirs = load_dirs(a)
        D, layers = F.stack(dirs)
        feats = F.mag_features(D)
        feats.update(F.cos_features(D, *ro_cache[fam]))
        feats["cos_own_contrast"] = None if a["circular"] else F.cos_between(dirs, own_cache[fam])
        feats["cos_vs_placebo"] = F.cos_between(dirs, load_dirs(dict(a, artefact=a["placebo"]))) if a["placebo"] else None
        if a["kind"] == "compiled":
            fs = json.loads((ROOT / a["artefact"]).with_name("fit_summary.json").read_text(encoding="utf-8"))
            feats["rel_dW"] = float(np.mean([v["rel_norm_dW"] for v in fs["layers"].values()]))
            rep = (ROOT / a["artefact"]).parents[1] / "report.json"
            feats["G0"] = json.loads(rep.read_text(encoding="utf-8"))["G0"]["median"] if rep.exists() else None
        t = F.target_row(arm_p, base_p)
        rows.append({**{k: a[k] for k in ("id", "family", "kind", "label", "set", "scale", "auc_row")},
                     "is_real": 0 if a["label"] == "placebo" else 1, "layers": len(layers), **feats, **t})
    (OUT / "arms.json").write_text(json.dumps({"rows": rows, "missing": missing}, indent=1), encoding="utf-8")

    feats = [k for k in rows[0] if k not in ("id", "family", "kind", "label", "set", "scale", "auc_row", "is_real", "layers",
                                             "d_gsm", "d_gsm_g2", "d_math", "d_math_g2", "d_tok", "d_cut", "n")]
    targets = ["d_gsm", "d_math", "d_tok", "d_cut"]
    fams = sorted({r["family"] for r in rows})

    def pairs(sel, f, t):
        xy = [(r[f], r[t]) for r in sel if r.get(f) is not None and r.get(t) is not None]
        return [x for x, _ in xy], [y for _, y in xy]

    sp = {}
    for t in targets:
        sp[t] = {}
        for f in feats:
            cell = {}
            for fam in fams + ["pooled"]:
                sel = rows if fam == "pooled" else [r for r in rows if r["family"] == fam]
                x, y = pairs(sel, f, t)
                cell[fam] = {"rho": F.spearman(x, y) if len(x) >= 3 else None, "n": len(x)}
            sp[t][f] = cell
    au = {}
    arows = [r for r in rows if r["auc_row"]]
    for f in feats:
        cell = {}
        for fam in fams + ["pooled"]:
            sel = arows if fam == "pooled" else [r for r in arows if r["family"] == fam]
            s = [r[f] for r in sel if r.get(f) is not None]; y = [r["is_real"] for r in sel if r.get(f) is not None]
            cell[fam] = {"auc": F.auc(s, y) if y and 0 < sum(y) < len(y) else None, "n_pos": int(sum(y)), "n_neg": int(len(y) - sum(y))}
        au[f] = cell
    lofo = {}
    for t in ("d_gsm", "d_math"):
        for f in feats:
            rho = sp[t][f]["pooled"]["rho"]
            if rho is not None and not np.isnan(rho) and abs(rho) > 0.5:
                lofo[f"{t}:{f}"] = {}
                for fam in fams:
                    x, y = pairs([r for r in rows if r["family"] != fam], f, t)
                    lofo[f"{t}:{f}"][f"without_{fam}"] = {"rho": F.spearman(x, y), "n": len(x)}
    summary = {"n_rows": len(rows), "n_auc_rows": len(arows), "families": {fam: sum(r["family"] == fam for r in rows) for fam in fams},
               "spearman": sp, "auc": au, "lofo": lofo, "missing": missing}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "auc_rows": len(arows), "missing": missing}, indent=1))
    for t in targets:
        print(f"== Spearman pooled vs {t}")
        for f in feats:
            c = sp[t][f]["pooled"]
            print(f"  {f:18s} rho {c['rho'] if c['rho'] is None else round(c['rho'], 3)!s:>7} n {c['n']}")
    print("== AUC real vs placebo (pooled)")
    for f in feats:
        c = au[f]["pooled"]
        print(f"  {f:18s} auc {c['auc'] if c['auc'] is None else round(c['auc'], 3)!s:>7} pos {c['n_pos']} neg {c['n_neg']}")
    if lofo:
        print("== leave-one-family-out", json.dumps(lofo, indent=1))


if __name__ == "__main__":
    main()
