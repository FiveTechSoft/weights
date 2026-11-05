"""LAB S3: ¿la DIVERSIDAD de familias de plantillas en train induce generalizacion a familias
nuevas? (ENTRENAMIENTO por gradiente; continuacion acotada de S1/S2)

HIPOTESIS (NO establecida, a medir): el negativo S2 en familias nuevas se debio a entrenar solo
con 3 familias de plantillas; entrenar con 12 podria inducir la operacion abstracta
"entidad consultada -> buscar su premisa -> copiar su atributo". Alternativas: falta de
capacidad, de datos, o la evaluacion. Este experimento SOLO separa diversidad: presupuesto
emparejado (mismos 320 pasos, acum 16, 3000 frases, mezcla 40% Alice, hiperparametros fijos S1).

REGLAS PRE-REGISTRADAS (antes de ver resultados):
  1. Familias de CONFIRMACION FRESCAS fijadas AHORA (F8/F9/F10, abajo), no usadas en S1/S2 ni
     en desarrollo. F5/F6/F7 (de S2) y F1-F3 quedan como DIAGNOSTICO, no como confirmacion.
  2. Brazos (presupuesto emparejado; 2 semillas cada uno):
       div    = 12 familias (F1-F3 + G1-G9), regla consistente
       fam3   = solo F1-F3 (como S1/S2), regla consistente  -> separa diversidad de solo re-correr
       pladiv = 12 familias, regla rota (respuesta final aleatoria) -> separa exposicion de regla
     Referencias sin entrenar: base (snapshot 1500), ngram8 (Alice + corpus diverso), azar empirico.
  3. Criterio de apoyo a la hipotesis: div supera a pladiv en familias FRESCAS (F8-F10) en
     AMBAS semillas (por orden y estricto, conteos). Cualquier otro resultado = sin apoyo.
     TODAS las familias y brazos se reportan, ganen o no; ningun negativo se descarta.
  4. Guarda de lenguaje: mezcla 40% Alice (fijada en S2 por dev); confirm-Alice PPL con cota
     9.17 (1.5x base) reportada por brazo (misma particion confirm que S2, mitad alta del
     holdout unico de 12.660 chars, sha256 442dd5a2...; jamas usada en seleccion).
  5. Items: 64 pares por familia (128 con ambos ordenes), pares entidad-atributo de la particion
     reservada (disjuntos de train y de dev), ausencia de prefijos en train verificada por codigo.
  6. Azar: 64/128 por orden; estricto ~16/64 bajo independencia NO asumida; azar empirico incluido.
  7. Nada aqui es evidencia de LLM general; negativo valido. Paciente: COPIA de snapshot 1500,
     runtime engine/nanowsp3_runtime.py (sin cambios upstream al momento de S3).

Familias de confirmacion FRESCAS:
  F8: "{X} packed a {A} for the trip. {Y} packed a {B} for the fair. What did {X} pack? {X} packed a" -> {A}
  F9: "The {A} was given to the {X}. The {B} was given to the {Y}. Who received the {A}? The" -> {X}  (direccion inversa: atributo->entidad)
  F10 (recencia adversaria): "Near the tree sat a {X} with a {A}. Near the fence sat a {Y} with a {B}. The {X} had a" -> {A}

Salida: lab/runs/nanowsp3_synth3.json. Checkpoints gitignored. Reanudable.
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
from train_nanowsp3_synth1 import (nonce_pool, f1, f2, f3, ids_of, cont_ppl, paired_eval,
                                   random_scorer_eval, SEED, LR, ACCUM, TOTAL_STEPS, MAXLEN)
from train_nanowsp3_synth2 import f5, f6, f7, query_s2, alice_stream

# --- familias train-only G1-G9 (forma re-declaracion; respuesta = ultima palabra) ---
def g1(X, Y, A, B): return f"{X} bought a {A} at the market. {Y} bought a {B} at the fair. {X} bought a {A}."
def g2(X, Y, A, B): return f"There is a {A} with the {X}. There is a {B} with the {Y}. There is a {A} with the {X}."
def g3(X, Y, A, B): return f"The {X} lost a {A} in the garden. The {Y} lost a {B} in the hall. The {X} lost a {A}."
def g4(X, Y, A, B): return f"A {X} carries a {A}. A {Y} carries a {B}. A {X} carries a {A}."
def g5(X, Y, A, B): return f"The {X} owns the {A} now. The {Y} owns the {B} now. The {X} owns the {A} now."
def g6(X, Y, A, B): return f"{X} gave the {A} to the cook. {Y} gave the {B} to the maid. {X} gave the {A} to the cook."
def g7(X, Y, A, B): return f"My {X} drew a {A}. My {Y} drew a {B}. My {X} drew a {A}."
def g8(X, Y, A, B): return f"The {X} wants a {A} today. The {Y} wants a {B} today. The {X} wants a {A} today."
def g9(X, Y, A, B): return f"A {A} sat near the {X}. A {B} sat near the {Y}. A {A} sat near the {X}."
G_FAMS = [g1, g2, g3, g4, g5, g6, g7, g8, g9]
FAMS12 = [f1, f2, f3] + G_FAMS

# --- familias FRESCAS de confirmacion ---
def f8(X, Y, A, B): return f"{X} packed a {A} for the trip. {Y} packed a {B} for the fair. What did {X} pack? {X} packed a {A}."
def f9(X, Y, A, B): return f"The {A} was given to the {X}. The {B} was given to the {Y}. Who received the {A}? The {X} received the {A}."
def f10(X, Y, A, B): return f"Near the tree sat a {X} with a {A}. Near the fence sat a {Y} with a {B}. The {X} had a {A}."

def query_s3(fam, X, Y, A, B):
    if fam is f8:
        return (f"{X} packed a {A} for the trip. {Y} packed a {B} for the fair. What did {X} pack? {X} packed a", f" {A}", f" {B}")
    if fam is f9:
        return (f"The {A} was given to the {X}. The {B} was given to the {Y}. Who received the {A}? The", f" {X}", f" {Y}")
    if fam is f10:
        return (f"Near the tree sat a {X} with a {A}. Near the fence sat a {Y} with a {B}. The {X} had a", f" {A}", f" {B}")
    raise ValueError(fam)

def make_items_generic(rng, pairs, fam, n, train_set, qfun):
    items, used = [], set()
    while len(items) < n * 2:
        (X, A), (Y, B) = rng.choice(pairs), rng.choice(pairs)
        if len({X, Y, A, B}) < 4:
            continue
        pair_items = []
        for first, second in ((A, B), (B, A)):
            pre, good, bad = qfun(fam, X, Y, first, second)
            if len(pre) + 6 > MAXLEN or pre in used or any(pre in t for t in train_set):
                pair_items = []; break
            used.add(pre)
            pair_items.append({"fam": fam.__name__, "prefix": pre, "good": good, "bad": bad})
        items.extend(pair_items)
    return items[:n * 2]

def make_train_fams(rng, pairs, fams, n):
    sents, seen = [], set()
    while len(sents) < n:
        (X, A), (Y, B) = rng.choice(pairs), rng.choice(pairs)
        if len({X, Y, A, B}) < 4:
            continue
        s = rng.choice(fams)(X, Y, A, B)
        if len(s) > MAXLEN or s in seen:
            continue
        seen.add(s); sents.append(s)
    return sents

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
    out_path = state_dir / "nanowsp3_synth3.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {
        "experimento": "S3 diversidad de familias vs generalizacion a familias nuevas (gradiente)",
        "reglas": "pre-registradas en docstring", "fases": {}}

    raw = (R / "lab" / "data" / "corpus.txt").read_text(encoding="utf-8", errors="replace")
    data = raw[15000:15000+400000].encode("ascii", "ignore").decode()
    n_hold = min(40000, max(len(data)//10, 1000))
    train_txt, hold_txt = data[:-n_hold], data[-n_hold:]
    half = len(hold_txt)//2
    confirm_alice = hold_txt[half:]

    rng = random.Random(SEED)
    pool = nonce_pool(rng, train_txt)
    ents, objs = pool[:80], pool[80:200]
    all_pairs = [(X, A) for X in ents for A in objs]
    rng.shuffle(all_pairs)
    train_pairs = all_pairs[:400]
    reserved_pairs = [p for p in all_pairs if p not in set(train_pairs)][2000:]

    sents_div = make_train_fams(rng, train_pairs, FAMS12, 3000)
    sents_fam3 = make_train_fams(rng, train_pairs, [f1, f2, f3], 3000)
    rng_pl = random.Random(SEED + 1)
    sents_pladiv = []
    for s in sents_div:
        m2 = re.match(r"(.+ )([a-z]+)\.$", s)
        w = objs[rng_pl.randrange(len(objs))]
        sents_pladiv.append(m2.group(1) + w + "." if m2 else s)
    alice_chunks = alice_stream(train_txt, 2000)  # 40% de 3000/0.6 -> mismo ratio que S2 m40

    # FASE 1: entrenamientos (2 semillas x 3 brazos), uno por llamada
    res_tr = out["fases"].setdefault("train", {})
    for seed_i in range(2):
        for arm, sents in (("div", sents_div), ("fam3", sents_fam3), ("pladiv", sents_pladiv)):
            tag = f"{arm}_s{seed_i}"
            if tag in res_tr and res_tr[tag].get("done"):
                continue
            r = train_one(sents, alice_chunks, SEED + 400 + seed_i,
                          state_dir / f"nanowsp3_synth3_{tag}.safetensors",
                          state_dir / f"nanowsp3_synth3_{tag}.state.pt", args.max_seconds)
            res_tr[tag] = r
            out["estado"] = f"PARCIAL {tag} paso {r['step']}/{TOTAL_STEPS}"
            out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
            if not r["done"]:
                print(out["estado"]); return
            print(f"completado {tag}"); return

    # FASE 2: evaluacion
    train_set = set(sents_div) | set(sents_fam3)
    fresh = {fam.__name__: make_items_generic(rng, reserved_pairs, fam, 64, train_set, query_s3)
             for fam in (f8, f9, f10)}
    diag = {fam.__name__: make_items_generic(rng, reserved_pairs, fam, 64, train_set, query_s2)
            for fam in (f5, f6, f7)}
    from train_nanowsp3_synth1 import query_prefix_and_cands
    interp = {"interp_F1F3": make_items_generic(rng, reserved_pairs, f1, 64, train_set, query_prefix_and_cands)}
    eval_sets = {**fresh, **diag, **interp}
    res_eval = out.get("evaluacion", {})
    t_eval = time.time()

    def _ck(etiqueta):
        out["evaluacion"] = res_eval
        out["estado"] = f"PARCIAL eval {etiqueta}"
        out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        if time.time() - t_eval > args.max_seconds:
            print(out["estado"]); return True
        return False

    for tag in res_tr:
        if tag in res_eval:
            continue
        m = NanoWSP3(); m.load_checkpoint(state_dir / f"nanowsp3_synth3_{tag}.safetensors")
        sc = lambda p, g: cont_ppl(m, p, g)
        res_eval[tag] = {fname: paired_eval(sc, items) for fname, items in eval_sets.items()}
        res_eval[tag]["confirm_alice_ppl"] = round(ppl_slice(m, confirm_alice), 2)
        if _ck(tag):
            return
    if "base" not in res_eval:
        m0 = load_final()
        sc0 = lambda p, g: cont_ppl(m0, p, g)
        res_eval["base"] = {fname: paired_eval(sc0, items) for fname, items in eval_sets.items()}
        res_eval["base"]["confirm_alice_ppl"] = round(ppl_slice(m0, confirm_alice), 2)
        if _ck("base"):
            return
    if "ngram8" not in res_eval:
        ng = NGram(train_txt + "\n" + "\n".join(sents_div), 8)
        scn = lambda p, g: ng.cont_ppl(p, g)
        res_eval["ngram8"] = {fname: paired_eval(scn, items) for fname, items in eval_sets.items()}
        if _ck("ngram8"):
            return
    if "azar_empirico" not in res_eval:
        res_eval["azar_empirico"] = {fname: random_scorer_eval(random.Random(SEED + 500 + i), items)
                                     for i, (fname, items) in enumerate(eval_sets.items())}
        if _ck("azar_empirico"):
            return
    res_eval["confirm_alice_cota"] = 9.17
    out["evaluacion"] = res_eval
    out["estado"] = "COMPLETO"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res_eval, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
