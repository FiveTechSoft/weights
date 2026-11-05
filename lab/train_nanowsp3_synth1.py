"""LAB S1: aprendizaje de binding composicional desde datos sinteticos (ENTRENAMIENTO por gradiente).

Peticion del propietario del repo (2026-11-05): continuar con entrenamiento sintetico hacia un
LLM pequeno que se comporte como modelo de lenguaje. S1 mide si el paciente (NanoWSP-3,
snapshot 1500) puede APRENDER la regla de ligadura entidad->atributo desde ejemplos y
TRANSFERIRLA a combinaciones reservadas. Es ENTRENAMIENTO por gradiente (Adam) sobre una COPIA
del checkpoint; el original queda intacto. NO es cirugia ni closed-form. La regla de la tarea
la disenamos nosotros: lo que se mide es si el gradiente induce generalizacion sistematica a
combinaciones reservadas, no una capacidad general de LLM.

HIPOTESIS (no probada a priori, a medir): el fine-tune puede generalizar la regla a pares y
plantillas nuevas; un brazo closed-form solo podria insertar asociaciones conocidas. Ambas
direcciones del resultado son validas (negativo incluido).

DATOS SINTETICOS: palabras nonce CVC generadas con semilla y verificadas (por codigo) ausentes
del corpus Alice. Familias de plantillas:
  F1: "The {X} has a {A}. The {Y} has a {B}. The {X} has a {A}."
  F2: "{X} put the {A} in the box. {Y} put the {B} in the cup. {X} put the {A} in the box."
  F3: "The {X} keeps a {A}. The {Y} keeps a {B}. What does the {X} keep? The {X} keeps a {A}."
  F4 (SOLO eval sistematica): "The {A} belongs to the {X}. The {B} belongs to the {Y}. The {A} belongs to the {X}."
Splits: TRAIN usa pares (X,A) y plantillas F1-F3; EVAL-INTERPOLACION usa pares NUEVOS (palabras
vistas en otros emparejamientos) con F1-F3; EVAL-SISTEMATICA usa pares nuevos con F4 (familia
no vista). Ninguna (premisa,consulta) de eval aparece en train (verificado por codigo).

BRAZOS:
  ft       = fine-tune Adam (lr 5e-4, batch-acum 16, loss LM completa) sobre Alice+sintetico mezclado.
  placebo  = mismo procedimiento con ligaduras barajadas por frase (regla inconsistente):
             separa "exponerse a palabras nonce" de "aprender la regla".
  base     = snapshot 1500 sin tocar (referencia).
  ngram8   = control recuperador: n-grama orden 8 contado sobre Alice+sintetico-train.

METRICAS PRE-REGISTRADAS (antes de ver resultados):
  M1. Accuracy pareada por item: prefijo hasta la consulta, comparacion PPL de las dos
      continuaciones candidatas (" {A}" vs " {B}", mismo prefijo). Cada item en DOS ordenes
      (A-primero/B-primero). Se reportan CONTEOS por orden (k/64 por conjunto) y estricto
      (ambos ordenes correctos). Referencia de azar: 50% por orden; 25% estricto SOLO bajo
      independencia entre ordenes (no asumida: se reporta ademas un scorer aleatorio sembrado
      como referencia empirica y el brazo placebo como referencia de "sin regla").
  M2. Accuracy en split de train (capacidad de memorizar lo ensenado).
  M3. Regresion: PPL en holdout REAL de Alice (misma particion que probes previos; base 6.11)
      tras el entrenamiento, por brazo.
Salida: lab/runs/nanowsp3_synth1.json. Checkpoints: lab/runs/nanowsp3_synth1_{ft,placebo}.safetensors
(gitignored). Determinista (semillas fijas). Reanudable (--max-seconds para trocear en llamadas).
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "engine")); sys.path.insert(0, str(R / "lab"))
from engine.nanowsp3_runtime import NanoWSP3
from probe_nanowsp3_retriever_baseline import NGram, load_final

SEED = 1234
LR = 5e-4
ACCUM = 16
TOTAL_STEPS = 320          # pasos de optimizador (x16 frases = 5120 frases-paso)
MAXLEN = 126

def nonce_pool(rng, train_txt, n=240):
    cons, vow = "bcdfghjklmnpqrstvwxz", "aeiou"
    out, seen = [], set()
    while len(out) < n:
        L = rng.choice([3, 3, 4])
        w = (rng.choice(cons) + rng.choice(vow) + rng.choice(cons)
             + (rng.choice(vow) if L == 4 else ""))
        if w in seen or w in train_txt:
            continue
        seen.add(w); out.append(w)
    return out

def f1(X, Y, A, B): return f"The {X} has a {A}. The {Y} has a {B}. The {X} has a {A}."
def f2(X, Y, A, B): return f"{X} put the {A} in the box. {Y} put the {B} in the cup. {X} put the {A} in the box."
def f3(X, Y, A, B): return f"The {X} keeps a {A}. The {Y} keeps a {B}. What does the {X} keep? The {X} keeps a {A}."
def f4(X, Y, A, B): return f"The {A} belongs to the {X}. The {B} belongs to the {Y}. The {A} belongs to the {X}."
TRAIN_FAMS = [f1, f2, f3]

def query_prefix_and_cands(fam, X, Y, A, B):
    """Prefijo hasta la consulta y (correcta, incorrecta); la respuesta es el PRIMER atributo."""
    if fam is f1:
        pre = f"The {X} has a {A}. The {Y} has a {B}. The {X} has a"
    elif fam is f2:
        pre = f"{X} put the {A} in the box. {Y} put the {B} in the cup. {X} put the"
    elif fam is f3:
        pre = f"The {X} keeps a {A}. The {Y} keeps a {B}. What does the {X} keep? The {X} keeps a"
    elif fam is f4:
        pre = f"The {A} belongs to the {X}. The {B} belongs to the {Y}. The {A} belongs to the"
        return pre, f" {X}", f" {Y}"
    else:
        raise ValueError(fam)
    return pre, f" {A}", f" {B}"

def make_items(rng, pairs, fams, n, train_sentences=None):
    items, used = [], set()
    while len(items) < n:
        (X, A), (Y, B) = rng.choice(pairs), rng.choice(pairs)
        if len({X, Y, A, B}) < 4:
            continue
        fam = rng.choice(fams)
        pair_items = []
        for order in ((A, B), (B, A)):
            pre, good, bad = query_prefix_and_cands(fam, X, Y, order[0], order[1])
            if len(pre) + 6 > MAXLEN or pre in used:
                pair_items = []
                break
            if train_sentences is not None and any(pre in t for t in train_sentences):
                pair_items = []
                break
            used.add(pre)
            pair_items.append({"fam": fam.__name__, "prefix": pre, "good": good, "bad": bad, "order": f"{order[0]}-first"})
        items.extend(pair_items)
        if len(items) >= n * 2:
            break
    return items[:n * 2]

def make_train(rng, pairs, n, train_txt):
    sents, seen = [], set()
    while len(sents) < n:
        (X, A), (Y, B) = rng.choice(pairs), rng.choice(pairs)
        if len({X, Y, A, B}) < 4:
            continue
        fam = rng.choice(TRAIN_FAMS)
        s = fam(X, Y, A, B)
        if len(s) > MAXLEN or s in seen:
            continue
        seen.add(s); sents.append(s)
    return sents

def ids_of(s):
    return [ord(c) for c in s if ord(c) < 256]

def cont_ppl(m, ctx, tgt):
    ids = ids_of(ctx + tgt)[:127]
    k = len(ids_of(ctx))
    if len(ids) <= k + 1:
        return float("nan")
    with torch.no_grad():
        lg, _ = m.forward(ids[:-1])
        losses = F.cross_entropy(lg[0], torch.tensor(ids[1:]), reduction="none")
        return math.exp(float(losses[k-1:].mean()))

def paired_eval(scorer, items):
    per_order = sum(1 for it in items if scorer(it["prefix"], it["good"]) < scorer(it["prefix"], it["bad"]))
    strict = 0
    for i in range(0, len(items) - 1, 2):
        a, b = items[i], items[i + 1]
        ok_a = scorer(a["prefix"], a["good"]) < scorer(a["prefix"], a["bad"])
        ok_b = scorer(b["prefix"], b["good"]) < scorer(b["prefix"], b["bad"])
        strict += (ok_a and ok_b)
    return {"per_order_ok": per_order, "per_order_n": len(items),
            "strict_ok": strict, "strict_n": len(items) // 2}

def random_scorer_eval(rng, items):
    per_order = sum(1 for _ in items if rng.random() < 0.5)
    strict = sum(1 for _ in range(len(items) // 2) if rng.random() < 0.25)
    return {"per_order_ok": per_order, "per_order_n": len(items),
            "strict_ok": strict, "strict_n": len(items) // 2}

def ppl_holdout(m, hold_txt):
    ids = ids_of(hold_txt[:2500]); tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids) - 65, 64):
            lg, _ = m.forward(ids[i:i+64])
            tot += float(F.cross_entropy(lg[0], torch.tensor(ids[i+1:i+65]), reduction="sum").item()); n += 64
    return math.exp(tot / max(n, 1))

def train_arm(name, sents, alice_chunks, ckpt_path, state_path, max_seconds, rng_np):
    m = load_final()
    for k in m.weights:
        m.weights[k] = m.weights[k].clone().requires_grad_(True)
    params = list(m.weights.values())
    opt = torch.optim.Adam(params, lr=LR)
    step = 0
    if state_path.exists():
        st = torch.load(state_path)
        for k in m.weights:
            m.weights[k].data = st["weights"][k]
        opt.load_state_dict(st["opt"]); step = st["step"]
        rng_np.bit_generator.state = st["rng"]
    t0 = time.time(); losses = []
    stream = sents + alice_chunks          # mezcla sintetico + Alice (anti-olvido)
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
        opt.step(); step += 1; losses.append(round(acc, 4))
        if time.time() - t0 > max_seconds:
            break
    done = step >= TOTAL_STEPS
    if done:
        for k in m.weights:
            m.weights[k] = m.weights[k].detach()
        m.save_checkpoint(ckpt_path)
    torch.save({"weights": {k: v.detach() for k, v in m.weights.items()},
                "opt": opt.state_dict(), "step": step, "rng": rng_np.bit_generator.state}, state_path)
    return {"arm": name, "steps_done": step, "steps_total": TOTAL_STEPS, "done": done,
            "loss_last20_mean": round(float(np.mean(losses[-20:])), 4) if losses else None,
            "ckpt": str(ckpt_path.name) if done else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=90.0)
    args = ap.parse_args()

    raw = (R / "lab" / "data" / "corpus.txt").read_text(encoding="utf-8", errors="replace")
    data = raw[15000:15000+400000].encode("ascii", "ignore").decode()
    n_hold = min(40000, max(len(data)//10, 1000))
    train_txt, hold_txt = data[:-n_hold], data[-n_hold:]

    rng = random.Random(SEED)
    pool = nonce_pool(rng, train_txt)
    ents, objs = pool[:80], pool[80:200]
    all_pairs = [(X, A) for X in ents for A in objs]
    rng.shuffle(all_pairs)
    train_pairs = all_pairs[:400]           # pares de entrenamiento
    eval_pairs = [p for p in all_pairs if p not in set(train_pairs)]  # combinaciones NUEVAS de palabras vistas

    train_sents = make_train(rng, train_pairs, 3000, train_txt)
    train_set = set(train_sents)
    eval_interp = make_items(rng, eval_pairs, TRAIN_FAMS, 64, train_set)
    eval_syst = make_items(rng, eval_pairs, [f4], 64, train_set)
    # control de contaminacion: ningun prefijo de eval en las frases de train
    for it in eval_interp + eval_syst:
        assert not any(it["prefix"] in s for s in train_set), "eval contaminado"

    alice_chunks = [train_txt[i:i+100] for i in range(0, 60000, 3000)]  # 20 trozos Alice anti-olvido
    out = {"experimento": "S1 binding composicional desde datos sinteticos (ENTRENAMIENTO gradiente)",
           "hipotesis": "a medir; ambas direcciones validas",
           "metricas": "pre-registradas en docstring",
           "n_train_sents": len(train_sents), "n_train_pairs": len(train_pairs),
           "n_eval_interp_items": len(eval_interp), "n_eval_syst_items": len(eval_syst)}

    # Brazo placebo: mismas frases, respuesta final barajada entre los dos atributos de la frase
    rng_pl = random.Random(SEED + 1)
    # placebo: mismas premisas, respuesta final sustituida por un objeto del pool (regla rota)
    placebo_sents = []
    for s in train_sents:
        import re as _re
        m2 = _re.match(r"(.+ )([a-z]+)\.$", s)
        # localizar los dos atributos de la frase es fragil; en su lugar barajar la ULTIMA palabra
        # con una palabra-objeto del pool (rompe la regla dentro de la frase)
        w = objs[rng_pl.randrange(len(objs))]
        placebo_sents.append(m2.group(1) + w + "." if m2 else s)

    state_dir = R / "lab" / "runs"
    arms = {}
    for name, sents in (("ft", train_sents), ("placebo", placebo_sents)):
        st_path = state_dir / f"nanowsp3_synth1_{name}.state.pt"
        ck_path = state_dir / f"nanowsp3_synth1_{name}.safetensors"
        arms[name] = train_arm(name, sents, alice_chunks, ck_path, st_path,
                               args.max_seconds, np.random.default_rng(SEED + (0 if name == "ft" else 7)))
        out[f"train_{name}"] = arms[name]
        if not arms[name]["done"]:
            out["estado"] = f"PARCIAL: reanudar (brazo {name} paso {arms[name]['steps_done']}/{TOTAL_STEPS})"
            (state_dir / "nanowsp3_synth1.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
            print(json.dumps(out, indent=1, ensure_ascii=False)); return

    # evaluacion
    res_eval = {}
    for name in ("ft", "placebo"):
        m = NanoWSP3(); m.load_checkpoint(state_dir / f"nanowsp3_synth1_{name}.safetensors")
        sc = lambda p, g: cont_ppl(m, p, g)
        res_eval[name] = {"interp": paired_eval(sc, eval_interp),
                          "sistematica": paired_eval(sc, eval_syst),
                          "alice_holdout_ppl": round(ppl_holdout(m, hold_txt), 2)}
    m0 = load_final()
    sc0 = lambda p, g: cont_ppl(m0, p, g)
    res_eval["base"] = {"interp": paired_eval(sc0, eval_interp),
                        "sistematica": paired_eval(sc0, eval_syst),
                        "alice_holdout_ppl": round(ppl_holdout(m0, hold_txt), 2)}
    ng = NGram(train_txt + "\n" + "\n".join(train_sents), 8)
    scn = lambda p, g: ng.cont_ppl(p, g)
    res_eval["ngram8"] = {"interp": paired_eval(scn, eval_interp),
                          "sistematica": paired_eval(scn, eval_syst)}
    res_eval["azar_empirico"] = {"interp": random_scorer_eval(random.Random(SEED + 2), eval_interp),
                                 "sistematica": random_scorer_eval(random.Random(SEED + 3), eval_syst)}
    # accuracy en train para ft (memorizacion de lo ensenado), con su propio scorer
    import re as _re
    rng_bad = random.Random(SEED + 4)
    m_ft = NanoWSP3(); m_ft.load_checkpoint(state_dir / "nanowsp3_synth1_ft.safetensors")
    sc_ft = lambda p, g: cont_ppl(m_ft, p, g)
    train_items = []
    for snt in train_sents:
        if len(train_items) >= 64:
            break
        m2 = _re.match(r"(.+ )([a-z]+)\.$", snt)
        if m2:
            bad_w = rng_bad.choice(objs)
            if bad_w != m2.group(2):
                train_items.append({"prefix": m2.group(1), "good": " " + m2.group(2), "bad": " " + bad_w})
    res_eval["ft"]["train_memorizacion"] = (
        {"per_order_ok": sum(1 for it in train_items if sc_ft(it["prefix"], it["good"]) < sc_ft(it["prefix"], it["bad"])),
         "per_order_n": len(train_items)} if train_items else None)

    out["evaluacion"] = res_eval
    out["estado"] = "COMPLETO"
    (state_dir / "nanowsp3_synth1.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
