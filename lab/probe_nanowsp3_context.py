"""LAB probe: NanoWSP-3 (checkpoint final, SIN reentrenar) — contexto, combinacion nueva, recuperacion.

Pregunta aprobada por el propietario del repo (2026-11-05): ¿responde al contexto y combina elementos de formas nuevas,
o solo recupera fragmentos del corpus? Modelo y configuracion INTACTOS; solo evaluacion.

Metricas pre-registradas (antes de ver resultados):
  A. Discriminacion sobre holdout REAL (ultimos 12.660 chars filtrados, nunca entrenados):
     PPL natural vs palabras permutadas vs ruido uniforme (base, sin cirugia).
  B. Sensibilidad al contexto: en 200 posiciones del holdout, PPL de los siguientes 32 chars dado
     (i) 64 chars previos, (ii) 8 chars previos, (iii) los 64 previos barajados por palabras.
     Contexto util <=> PPL(i) < PPL(ii) < PPL(iii).
  C. Generacion con prompts NUEVOS (verificado por codigo que NO aparecen en el corpus) vs
     prompts del corpus: % ventanas de 8 chars presentes en el corpus (metrica de copia ya usada),
     match verbatim maximo, y validez lexica contra lista embebida de ~180 palabras ingleses
     comunes (no contra el vocabulario del corpus, para no autovalidarse).
  D. Control: si en prompts nuevos la copia cae pero la lexicalidad se mantiene, hay combinacion
     no literal; si la copia se mantiene igual de alta, es costura de fragmentos.

Todo greedy (temp 0), determinista. Resultado negativo = resultado valido.
Salida: lab/runs/nanowsp3_context_probe.json
"""
import json, math, random, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "engine"))
from engine.nanowsp3_runtime import NanoWSP3

RANDOM_SEED = 42
WORDS_COMMON = set("""a i the of and to in is was he she it for on as are with his her they at be this have from or had by not but what all were we when your can said there use an each which do their if will up other about out many then them these so some would make like him into time has look two more write go see number no way could people my than first been call who its now find long down day did get come made may part over new sound take only little work know place year live me back give most very after thing our just name good say where help through much before line right too mean old any same tell child well large must big even such because turn here why ask went read need land different home us move try kind hand picture again change off play spell air away animal house point page letter mother answer found study still learn should world high every near add food between own below country plant last school father keep tree never start city earth eye light thought head under story saw left few while along might close something seem next hard open example begin life always those both paper together got group often run""".split())


def load_final():
    z = np.load(R / "lab" / "runs" / "nanowsp3_trace.npz")
    names = json.loads(str(z["names"]))
    shapes = json.loads(json.loads(str(z["shapes"])))
    m = NanoWSP3()
    km = {"embed.weight": "embedding.weight", "out.weight": "output.weight", "fn.weight": "final_norm.weight"}
    for l in range(6):
        for a, b in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")):
            km[f"{a}.{l}"] = f"layers.{l}.attention.{b}.weight"
        km[f"an.{l}"] = f"layers.{l}.attn_norm.weight"; km[f"mn.{l}"] = f"layers.{l}.mlp_norm.weight"
        km[f"min_.{l}"] = f"layers.{l}.mlp_in.weight"; km[f"mout.{l}"] = f"layers.{l}.mlp_out.weight"
    flat = z["snapshots_flat"][-1]; off = 0
    for k in names:
        n = int(np.prod(shapes[k]))
        m.weights[km.get(k, k)] = torch.tensor(flat[off:off+n].reshape(shapes[k])); off += n
    return m


def ppl_of(m, ids, limit=2500):
    ids = ids[:limit]; tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids) - 65, 64):
            lg, _ = m.forward(ids[i:i+64])
            tot += float(F.cross_entropy(lg[0], torch.tensor(ids[i+1:i+65]), reduction="sum").item()); n += 64
    return math.exp(tot / max(n, 1))


def cont_ppl(m, ctx, tgt):
    """PPL del target condicionado al contexto (ctx+tgt <= 127 chars)."""
    ids = [ord(c) for c in (ctx + tgt) if ord(c) < 256][:127]
    k = len([c for c in ctx if ord(c) < 256])
    if len(ids) <= k + 1:
        return float("nan")
    with torch.no_grad():
        lg, _ = m.forward(ids[:-1])
        y = torch.tensor(ids[1:])
        losses = F.cross_entropy(lg[0], y, reduction="none")
        return math.exp(float(losses[k-1:].mean()))


def copy_stats(gen_tail, train_txt):
    grams = {gen_tail[i:i+8] for i in range(max(0, len(gen_tail) - 7))}
    inc = sum(1 for g in grams if g in train_txt)
    best = 0
    for i in range(len(gen_tail)):
        for L in range(best + 1, len(gen_tail) - i + 1):
            if gen_tail[i:i+L] in train_txt: best = L
            else: break
    words = re.findall(r"\b[a-zA-Z]+\b", gen_tail.lower())
    valid = [w for w in words if (len(w) == 1 and w in ("a", "i")) or w in WORDS_COMMON]
    return {"pct_8char_en_train": round(100 * inc / max(len(grams), 1), 1),
            "match_verbatim_max_chars": best,
            "lexical_common_pct": round(100 * len(valid) / max(len(words), 1), 1)}


def main():
    m = load_final()
    raw = (R / "lab" / "data" / "corpus.txt").read_text(encoding="utf-8", errors="replace")
    data = raw[15000:15000+400000].encode("ascii", "ignore").decode()
    ids = np.frombuffer(data.encode("ascii", "ignore"), dtype=np.uint8).astype(np.int64)
    n_hold = min(40000, max(len(ids)//10, 1000))
    train_txt, hold_txt = data[:-n_hold], data[-n_hold:]
    hold_ids = [int(t) for t in ids[-n_hold:]]

    res = {"modelo": "NanoWSP-3 snapshot 1500 (base, sin cirugia, sin reentrenar)",
           "metricas": "pre-registradas en docstring del script"}

    # A. discriminacion en holdout REAL
    nat = hold_txt[:2500]
    words = nat.split()
    rng = random.Random(RANDOM_SEED); sh = words[:]; rng.shuffle(sh)
    noise = "".join(random.Random(RANDOM_SEED).choice("abcdefghijklmnopqrstuvwxyz \n.,!?'") for _ in range(2500))
    res["A_discriminacion_holdout_real"] = {
        "natural": round(ppl_of(m, hold_ids), 2),
        "permutado": round(ppl_of(m, [ord(c) for c in " ".join(sh)]), 2),
        "ruido": round(ppl_of(m, [ord(c) for c in noise]), 2)}

    # B. sensibilidad al contexto (200 posiciones del holdout)
    rng2 = random.Random(7)
    positions = sorted(rng2.sample(range(100, len(hold_txt) - 40), 200))
    p_full, p_short, p_shuf = [], [], []
    for pos in positions:
        ctx64 = hold_txt[max(0, pos-64):pos]
        tgt = hold_txt[pos:pos+32]
        w = ctx64.split(); rng2.shuffle(w)
        p_full.append(cont_ppl(m, ctx64, tgt))
        p_short.append(cont_ppl(m, ctx64[-8:], tgt))
        p_shuf.append(cont_ppl(m, " ".join(w), tgt))
    res["B_sensibilidad_contexto"] = {
        "ppl_ctx64": round(float(np.nanmean(p_full)), 2),
        "ppl_ctx8": round(float(np.nanmean(p_short)), 2),
        "ppl_ctx64_barajado": round(float(np.nanmean(p_shuf)), 2),
        "n_posiciones": len(positions)}

    # C/D. prompts nuevos vs corpus
    corpus_prompts = ["Alice was ", "The Queen of Hearts ", "She said to the "]
    novel_prompts = ["The Rabbit opened the ", "She ordered the Hatter to ",
                     "The Queen drank tea with the Mouse and ", "A grinning cat counted seven "]
    for tag, plist in (("corpus", corpus_prompts), ("nuevos", novel_prompts)):
        rows = []
        for p in plist:
            assert tag == "corpus" or p not in train_txt, f"prompt nuevo contaminado: {p!r}"
            g = m.generate(p, max_new_tokens=90, temperature=0.0)
            tail = g[len(p):]
            st = copy_stats(tail, train_txt)
            st["prompt"] = p; st["tail"] = tail[:80]
            rows.append(st)
        res[f"C_prompts_{tag}"] = rows
    out = R / "lab" / "runs" / "nanowsp3_context_probe.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
