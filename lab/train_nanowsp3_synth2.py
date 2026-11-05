"""LAB S2: binding composicional con conservacion de lenguaje y evaluacion sin atajo (ENTRENAMIENTO gradiente).

Continuacion de S1 (seccion 31) bajo la peticion vigente del propietario. S1 quedo como negativo
en dos limites: (a) regresion severa de Alice (6.11 -> 17.04) por mezcla anti-olvido de 0.7%;
(b) conjunto sistematica F4 invalidado porque el placebo empataba (copia en contexto inducida
por exposicion a plantillas). S2 ataca esos dos limites SIN tocar el paciente original y SIN
elegir hiperparametros con el conjunto de confirmacion.

REGLAS PRE-REGISTRADAS (antes de ver resultados):
  1. DESARROLLO vs RESERVADO:
     - DEV: plantillas F1-F3 (las de S1) con pares nuevos + PRIMERA mitad del holdout real de
       Alice (dev-Alice, chars 0-20000 de la particion reservada). Solo aqui se elige la mezcla.
     - RESERVADO: familias F5/F6/F7 (NO vistas en train) con pares nuevos + SEGUNDA mitad del
       holdout de Alice (confirm-Alice, chars 20000-40000). Se reporta solo tras fijar la mezcla.
  2. SELECCION DE MEZCLA (solo en DEV): candidatas {10%, 40%} de contenido Alice en el stream.
     Regla: entre las que cumplan dev-Alice PPL <= 9.17 (1.5x base 6.11), elegir la de mejor
     dev-binding estricto; empate -> mayor mezcla Alice. Si ninguna cumple, elegir la de menor
     dev-Alice PPL y reportar el tradeoff como hallazgo.
  3. Hiperparametros FIJOS de S1 (no se ajustan): Adam lr 5e-4, acum 16, 320 pasos, clip 1.0.
  4. Semillas: 3 para ft y 3 para placebo en RESERVADO (mismas semillas emparejadas por brazo).
  5. Familias RESERVADAS anti-atajo (la premisa NO repite la frase de la consulta):
     F5: "The {X} found a {A} near the river. The {Y} found a {B} near the mill. What did the {X} find? The {X} found a"
     F6 (recencia adversaria): "The {A} is with the {X}. The {B} is with the {Y}. The {X} keeps the" -> correcta {A};
        un copiador por recencia (ultimo objeto mencionado) elige {B} (incorrecta).
     F7: "A {A} belongs to a {X}. A {B} belongs to a {Y}. Who has the {A}? The {X} has the" -> correcta {A}.
     Se reportan TODAS las familias, gane ft o no; no se descarta ninguna a posteriori ni se
     seleccionan solo diferencias favorables. ft > placebo en una familia = evidencia A FAVOR de
     aprendizaje de la regla ahi; ft ~= placebo = atajo no descartado (negativo preservado);
     ft < placebo tambien se reporta. Ninguna mejora puntual elimina un atajo sin su control.
     Variacion entre semillas reportada por brazo (3 semillas emparejadas ft/placebo).
  6. Metricas: conteos por orden y estricto (azar 32/64 y 8/32 bajo independencia no asumida;
     scorer aleatorio empirico incluido), por familia y agregado; regresion confirm-Alice PPL con
     cota pre-registrada 9.17 (1.5x base 6.11) reportada como pasa/no pasa por brazo.
     Las mitades dev/confirm de Alice son disjuntas con guarda de 128 chars entre ellas y cada
     ventana de eval usa contexto de su propia mitad (sin solape de contexto entre seleccion y
     confirmacion).
  7. Resultado negativo valido; nada aqui es evidencia de LLM general.

Salida: lab/runs/nanowsp3_synth2.json. Checkpoints gitignored. Reanudable por fases.
"""
import argparse, json, math, random, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "engine")); sys.path.insert(0, str(R / "lab"))
from engine.nanowsp3_runtime import NanoWSP3
from probe_nanowsp3_retriever_baseline import NGram, load_final
from train_nanowsp3_synth1 import (nonce_pool, f1, f2, f3, make_train, make_items, ids_of,
                                   cont_ppl, paired_eval, random_scorer_eval, SEED, LR, ACCUM,
                                   TOTAL_STEPS, MAXLEN)

GUARD = 128  # mitad real calculada en main; guarda entre mitades
MIX_CANDIDATES = [0.10, 0.40]
DEV_ALICE_PPL_BOUND = 9.17  # 1.5x base 6.11, pre-registrado

def f5(X, Y, A, B): return f"The {X} found a {A} near the river. The {Y} found a {B} near the mill. What did the {X} find? The {X} found a {A}."
def f6(X, Y, A, B): return f"The {A} is with the {X}. The {B} is with the {Y}. The {X} keeps the {A}."
def f7(X, Y, A, B): return f"A {A} belongs to a {X}. A {B} belongs to a {Y}. Who has the {A}? The {X} has the {A}."

def query_s2(fam, X, Y, A, B):
    if fam is f5:
        return (f"The {X} found a {A} near the river. The {Y} found a {B} near the mill. What did the {X} find? The {X} found a", f" {A}", f" {B}")
    if fam is f6:
        return (f"The {A} is with the {X}. The {B} is with the {Y}. The {X} keeps the", f" {A}", f" {B}")
    if fam is f7:
        return (f"A {A} belongs to a {X}. A {B} belongs to a {Y}. Who has the {A}? The {X} has the", f" {A}", f" {B}")
    raise ValueError(fam)

def make_items_s2(rng, pairs, fam, n, train_set):
    items, used = [], set()
    while len(items) < n * 2:
        (X, A), (Y, B) = rng.choice(pairs), rng.choice(pairs)
        if len({X, Y, A, B}) < 4:
            continue
        pair_items = []
        for first, second in ((A, B), (B, A)):
            pre, good, bad = query_s2(fam, X, Y, first, second)
            if len(pre) + 6 > MAXLEN or pre in used or any(pre in t for t in train_set):
                pair_items = []; break
            used.add(pre)
            pair_items.append({"fam": fam.__name__, "prefix": pre, "good": good, "bad": bad})
        items.extend(pair_items)
    return items[:n * 2]

def alice_stream(train_txt, n):
    return [train_txt[i:i+100] for i in range(0, min(len(train_txt) - 101, n * 100), 100)][:n]

def train_one(synth_sents, alice_chunks, seed, ckpt_path, state_path, max_seconds):
    m = load_final()
    for k in m.weights:
        m.weights[k] = m.weights[k].clone().requires_grad_(True)
    params = list(m.weights.values())
    opt = torch.optim.Adam(params, lr=LR)
    rng_np = np.random.default_rng(seed)
    step = 0
    if state_path.exists():
        st = torch.load(state_path)
        for k in m.weights:
            m.weights[k].data = st["weights"][k]
        opt.load_state_dict(st["opt"]); step = st["step"]
        rng_np.bit_generator.state = st["rng"]
    stream = synth_sents + alice_chunks
    t0 = time.time(); losses = []
    while step < TOTAL_STEPS:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(ACCUM):
            s = stream[rng_np.integers(len(stream))]
            ids = ids_of(s)[:MAXLEN]
            if len(ids) < 9:
                continue
            lg, _ = m.forward(ids[:-1])
            loss = F.cross_entropy(lg[0], torch.tensor(ids[1:]))
            (loss / ACCUM).backward(); acc += float(loss) / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); step += 1; losses.append(acc)
        if time.time() - t0 > max_seconds:
            break
    done = step >= TOTAL_STEPS
    if done:
        for k in m.weights:
            m.weights[k] = m.weights[k].detach()
        m.save_checkpoint(ckpt_path)
    torch.save({"weights": {k: v.detach() for k, v in m.weights.items()},
                "opt": opt.state_dict(), "step": step, "rng": rng_np.bit_generator.state}, state_path)
    return {"done": done, "step": step, "loss_last20": round(float(np.mean(losses[-20:])), 4) if losses else None}

def ppl_slice(m, txt):
    ids = ids_of(txt)[:2500]; tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids) - 65, 64):
            lg, _ = m.forward(ids[i:i+64])
            tot += float(F.cross_entropy(lg[0], torch.tensor(ids[i+1:i+65]), reduction="sum").item()); n += 64
    return math.exp(tot / max(n, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=90.0)
    args = ap.parse_args()
    state_dir = R / "lab" / "runs"
    out_path = state_dir / "nanowsp3_synth2.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "experimento": "S2 binding con conservacion de lenguaje y eval sin atajo (gradiente)",
        "reglas": "pre-registradas en docstring", "fases": {}}

    raw = (R / "lab" / "data" / "corpus.txt").read_text(encoding="utf-8", errors="replace")
    data = raw[15000:15000+400000].encode("ascii", "ignore").decode()
    n_hold = min(40000, max(len(data)//10, 1000))
    train_txt, hold_txt = data[:-n_hold], data[-n_hold:]
    half = len(hold_txt)//2
    dev_alice, confirm_alice = hold_txt[:half-GUARD], hold_txt[half:]  # mitades disjuntas con guarda

    rng = random.Random(SEED)
    pool = nonce_pool(rng, train_txt)
    ents, objs = pool[:80], pool[80:200]
    all_pairs = [(X, A) for X in ents for A in objs]
    rng.shuffle(all_pairs)
    train_pairs = all_pairs[:400]
    eval_pairs = [p for p in all_pairs if p not in set(train_pairs)]
    dev_pairs, reserved_pairs = eval_pairs[:2000], eval_pairs[2000:]  # pares disjuntos dev/reservado
    train_sents = make_train(rng, train_pairs, 3000, train_txt)
    train_set = set(train_sents)
    dev_items = make_items(rng, dev_pairs, [f1, f2, f3], 48, train_set)
    rng_pl = random.Random(SEED + 1)
    placebo_sents = []
    for s in train_sents:
        m2 = re.match(r"(.+ )([a-z]+)\.$", s)
        w = objs[rng_pl.randrange(len(objs))]
        placebo_sents.append(m2.group(1) + w + "." if m2 else s)

    # FASE 1b: re-eval de PPLs Alice con la particion corregida (bug: cortes fijos 20000/40000
    # excedian el holdout real de ~13.6k; dev era el holdout completo y confirm quedaba vacio)
    if out.get("mix") and "mix_ppl_corregido" not in out["fases"]:
        devres = out["fases"]["mix_dev"]
        for tag in list(devres):
            m = NanoWSP3(); m.load_checkpoint(state_dir / f"nanowsp3_synth2_sel_{tag}.safetensors")
            devres[tag]["dev_alice_ppl_v2"] = round(ppl_slice(m, dev_alice), 2)
        ok = [t for t, r in devres.items() if r["dev_alice_ppl_v2"] <= DEV_ALICE_PPL_BOUND]
        if ok:
            chosen = max(ok, key=lambda t: (devres[t]["dev_binding"]["strict_ok"], float(t[1:])))
        else:
            chosen = min(devres, key=lambda t: devres[t]["dev_alice_ppl_v2"])
        out["fases"]["mix_ppl_corregido"] = {"dev_alice_ppl_v2": {t: r["dev_alice_ppl_v2"] for t, r in devres.items()},
                                             "mix_v1": out["mix"], "mix_v2": chosen,
                                             "nota": "holdout real ~13.6k; mitades disjuntas con guarda 128"}
        out["mix"] = chosen
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print("PPLs dev corregidos:", out["fases"]["mix_ppl_corregido"])
        if chosen != out["fases"]["mix_ppl_corregido"]["mix_v1"]:
            print("AVISO: la mezcla elegida cambia con la particion corregida")
        return
    # FASE 1: seleccion de mezcla SOLO con DEV
    if "mix" not in out:
        devres = out["fases"].setdefault("mix_dev", {})
        for mix in MIX_CANDIDATES:
            tag = f"m{int(mix*100)}"
            if tag in devres and devres[tag].get("done"):
                continue
            n_alice = int(3000 * mix / (1 - mix))
            r = train_one(train_sents, alice_stream(train_txt, n_alice), SEED + 100,
                          state_dir / f"nanowsp3_synth2_sel_{tag}.safetensors",
                          state_dir / f"nanowsp3_synth2_sel_{tag}.state.pt", args.max_seconds)
            devres[tag] = r
            if not r["done"]:
                out["estado"] = f"PARCIAL seleccion mezcla {tag} paso {r['step']}/{TOTAL_STEPS}"
                out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(out["estado"]); return
            m = NanoWSP3(); m.load_checkpoint(state_dir / f"nanowsp3_synth2_sel_{tag}.safetensors")
            sc = lambda p, g: cont_ppl(m, p, g)
            devres[tag]["dev_binding"] = paired_eval(sc, dev_items)
            devres[tag]["dev_alice_ppl"] = round(ppl_slice(m, dev_alice), 2)
            out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(f"mezcla {tag} evaluada en dev"); return
        ok = [t for t, r in devres.items() if r["dev_alice_ppl"] <= DEV_ALICE_PPL_BOUND]
        if ok:
            chosen = max(ok, key=lambda t: (devres[t]["dev_binding"]["strict_ok"], float(t[1:])))
        else:
            chosen = min(devres, key=lambda t: devres[t]["dev_alice_ppl"])
            out["mix_tradeoff"] = "ninguna mezcla cumplio la cota dev-Alice; se eligio la de menor PPL dev"
        out["mix"] = chosen
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False)); print("mezcla elegida:", chosen); return

    mix = out["mix"]; n_alice = int(3000 * float(mix[1:]) / 100 / (1 - float(mix[1:]) / 100))
    alice_chunks = alice_stream(train_txt, n_alice)

    # FASE 2: entrenamiento en RESERVADO, 3 semillas emparejadas por brazo
    res_tr = out["fases"].setdefault("train_reserved", {})
    for seed_i in range(3):
        for arm, sents in (("ft", train_sents), ("placebo", placebo_sents)):
            tag = f"{arm}_s{seed_i}"
            if tag in res_tr and res_tr[tag].get("done"):
                continue
            r = train_one(sents, alice_chunks, SEED + 200 + seed_i,
                          state_dir / f"nanowsp3_synth2_{tag}.safetensors",
                          state_dir / f"nanowsp3_synth2_{tag}.state.pt", args.max_seconds)
            res_tr[tag] = r
            out["estado"] = f"PARCIAL reservado {tag} paso {r['step']}/{TOTAL_STEPS}"
            out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
            if not r["done"]:
                print(out["estado"]); return
            print(f"completado {tag}"); return

    # FASE 3: evaluacion en RESERVADO
    eval_sets = {fam.__name__: make_items_s2(rng, reserved_pairs, fam, 64, train_set) for fam in (f5, f6, f7)}
    res_eval = {}
    for tag in res_tr:
        m = NanoWSP3(); m.load_checkpoint(state_dir / f"nanowsp3_synth2_{tag}.safetensors")
        sc = lambda p, g: cont_ppl(m, p, g)
        res_eval[tag] = {fname: paired_eval(sc, items) for fname, items in eval_sets.items()}
        res_eval[tag]["confirm_alice_ppl"] = round(ppl_slice(m, confirm_alice), 2)
    m0 = load_final()
    sc0 = lambda p, g: cont_ppl(m0, p, g)
    res_eval["base"] = {fname: paired_eval(sc0, items) for fname, items in eval_sets.items()}
    res_eval["base"]["confirm_alice_ppl"] = round(ppl_slice(m0, confirm_alice), 2)
    ng = NGram(train_txt + "\n" + "\n".join(train_sents), 8)
    scn = lambda p, g: ng.cont_ppl(p, g)
    res_eval["ngram8"] = {fname: paired_eval(scn, items) for fname, items in eval_sets.items()}
    res_eval["azar_empirico"] = {fname: random_scorer_eval(random.Random(SEED + 300 + i), items)
                                 for i, (fname, items) in enumerate(eval_sets.items())}
    # confirm-Alice PPL corregido (la fase previa lo computo sobre una particion vacia)
    for tag in list(res_tr):
        m = NanoWSP3(); m.load_checkpoint(state_dir / f"nanowsp3_synth2_{tag}.safetensors")
        res_eval[tag]["confirm_alice_ppl_v2"] = round(ppl_slice(m, confirm_alice), 2)
        res_eval[tag]["confirm_alice_ppl_v1_invalido"] = res_eval[tag].pop("confirm_alice_ppl")
    res_eval["base"]["confirm_alice_ppl_v2"] = round(ppl_slice(m0, confirm_alice), 2)
    res_eval["base"]["confirm_alice_ppl_v1_invalido"] = res_eval["base"].pop("confirm_alice_ppl")
    res_eval["confirm_alice_cota"] = DEV_ALICE_PPL_BOUND
    out["evaluacion_reservado"] = res_eval
    out["estado"] = "COMPLETO"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["evaluacion_reservado"], indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
