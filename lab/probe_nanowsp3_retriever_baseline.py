"""LAB probe: NanoWSP-3 vs RECUPERADOR BASELINE directo (completa la comparacion aprobada 2026-11-05 08:27).

Parte pendiente de la evaluacion aprobada: comparar el modelo contra un recuperador directo
sobre el MISMO paciente y los MISMOS datos, sin reentrenar el modelo ni crear otro modelo de
generacion. Los baselines son contadores/recuperadores sobre el corpus de train (no son el
paciente; no se toca el checkpoint).

Metricas pre-registradas (antes de ver resultados):
  R1. n-grama de caracteres orden 8 (contexto 7) con backoff estupido (lambda 0.4, piso
      unigrama add-0.05), contado SOLO sobre train_txt:
        R1a. PPL en holdout real natural/permutado/ruido (mismo protocolo y semillas que el
             probe de contexto, seccion A).
        R1b. PPL condicional de los mismos 200 objetivos de 32 chars bajo ctx64 / ctx8 /
             ctx64-barajado (mismas posiciones, semilla 7). Prediccion pre-registrada: un
             orden-8 solo usa los ultimos 7 chars, asi que su degradacion con contexto
             largo barajado debe ser pequena comparada con la del modelo (6.26 -> 8.00).
  R2. Recuperador de sufijo mas largo (longest-suffix-match) sobre train_txt: dado el
      contexto, busca el sufijo mas largo presente en train y emite el char que le seguia
      (ocurrencia elegida con RNG sembrado); sin match, retrocede acortando; piso: char
      mas frecuente del corpus. Genera 90 chars con los MISMOS 7 prompts del probe anterior
      y se le aplican las MISMAS metricas de copia/verbatim/lexico. Define el TECHO de copia
      de un recuperador puro (por construccion ~100% 8-gramas en corpus).
  R3. Prueba composicional con respuestas comprobables, SIN reentrenar ni cambiar de tarea:
      binding de palabras nonce (inexistentes en el corpus, verificado por codigo).
      Plantillas:
        T1: "The {X} has a {A}. The {Y} has a {B}. The {X} has a"  ->  " {A}" vs " {B}"
        T2: "{X} put the {A} in the box. {Y} put the {B} in the cup. {X} put the" -> " {A}" vs " {B}"
      8 items por plantilla, cada uno en los DOS ordenes (A-primero / B-primero): acierto
      estricto = acierta en ambos ordenes (anula sesgo lexico por continuacion; azar=50%).
      Control: el mismo scoring con R1. Si el modelo no supera azar estricto, es un
      resultado negativo valido y se reporta como limite, no se inventa otra prueba.

Interpretacion pre-registrada (limites): R1/R2 son DOS recuperadores concretos, no todos los
posibles; superarlos no prueba un mecanismo, y empatar en metricas de copia tampoco lo prueba.
La comparacion solo separa lo que el modelo hace de lo que estos recuperadores hacen con los
mismos datos. Determinista (temp 0, semillas fijas). Salida: lab/runs/nanowsp3_retriever_baseline.json
"""
import json, math, random, re, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "engine"))
from engine.nanowsp3_runtime import NanoWSP3

RANDOM_SEED = 42
NGRAM_ORDER = 8          # contexto 7
BACKOFF_LAMBDA = 0.4
UNIGRAM_ALPHA = 0.05
VOCAB = 256
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


def cont_ppl_model(m, ctx, tgt):
    ids = [ord(c) for c in (ctx + tgt) if ord(c) < 256][:127]
    k = len([c for c in ctx if ord(c) < 256])
    if len(ids) <= k + 1:
        return float("nan")
    with torch.no_grad():
        lg, _ = m.forward(ids[:-1])
        y = torch.tensor(ids[1:])
        losses = F.cross_entropy(lg[0], y, reduction="none")
        return math.exp(float(losses[k-1:].mean()))


class NGram:
    """n-grama de chars con backoff estupido. Cuenta SOLO train."""
    def __init__(self, text, order):
        self.order = order
        self.counts = [defaultdict(Counter) for _ in range(order)]  # nivel k: contexto k chars -> Counter
        self.unigram = Counter(text)
        self.total_uni = sum(self.unigram.values())
        for lvl in range(1, order):
            for i in range(lvl, len(text)):
                self.counts[lvl][text[i-lvl:i]][text[i]] += 1

    def p(self, c, hist):
        for lvl in range(min(self.order - 1, len(hist)), 0, -1):
            ctx = hist[-lvl:]
            cnt = self.counts[lvl].get(ctx)
            if cnt and cnt.get(c, 0) > 0:
                lam = BACKOFF_LAMBDA ** (min(self.order - 1, len(hist)) - lvl)
                return lam * cnt[c] / sum(cnt.values())
        lam = BACKOFF_LAMBDA ** min(self.order - 1, len(hist))
        return lam * (self.unigram.get(c, 0) + UNIGRAM_ALPHA) / (self.total_uni + UNIGRAM_ALPHA * VOCAB)

    def cont_ppl(self, ctx, tgt):
        s, hist = 0.0, ctx
        for c in tgt:
            p = max(self.p(c, hist), 1e-12)
            s += -math.log(p); hist += c
        return math.exp(s / max(len(tgt), 1))

    def ppl_text(self, text, limit=2500):
        text = text[:limit]; s, n = 0.0, 0
        for i in range(0, len(text) - 65, 64):
            for j, c in enumerate(text[i+1:i+65]):
                p = max(self.p(c, text[:i+1+j][-64:]), 1e-12)
                s += -math.log(p); n += 1
        return math.exp(s / max(n, 1))


def retrieve_gen(train_txt, prompt, n_chars, seed):
    rng = random.Random(seed)
    floor_char = Counter(train_txt).most_common(1)[0][0]
    out = []
    for _ in range(n_chars):
        ctx = prompt + "".join(out)
        c = None
        for L in range(min(32, len(ctx)), 0, -1):
            sfx = ctx[-L:]
            occs = []
            start = train_txt.find(sfx)
            while start != -1 and start + L < len(train_txt):
                occs.append(start + L)
                start = train_txt.find(sfx, start + 1)
            if occs:
                c = train_txt[occs[rng.randrange(len(occs))]]
                break
        out.append(c if c is not None else floor_char)
    return "".join(out)


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


def binding_items():
    pool = ["wug", "mip", "dor", "zav", "tob", "lom", "rux", "gom",
            "kaj", "veb", "nof", "yat", "hux", "dax", "wiv", "queb"]
    items = []
    for i in range(8):
        X, Y = pool[(2*i) % 16], pool[(2*i + 5) % 16]
        A, B = pool[(2*i + 9) % 16], pool[(2*i + 13) % 16]
        assert len({X, Y, A, B}) == 4
        items.append((X, Y, A, B))
    return items


def t1(X, Y, A, B):
    return f"The {X} has a {A}. The {Y} has a {B}. The {X} has a"

def t2(X, Y, A, B):
    return f"{X} put the {A} in the box. {Y} put the {B} in the cup. {X} put the"


def main():
    m = load_final()
    raw = (R / "lab" / "data" / "corpus.txt").read_text(encoding="utf-8", errors="replace")
    data = raw[15000:15000+400000].encode("ascii", "ignore").decode()
    n_hold = min(40000, max(len(data)//10, 1000))
    train_txt, hold_txt = data[:-n_hold], data[-n_hold:]

    ng = NGram(train_txt, NGRAM_ORDER)
    res = {"modelo": "NanoWSP-3 snapshot 1500 (intacto) vs baselines de recuperacion sobre train",
           "metricas": "pre-registradas en docstring del script"}

    # R1a. PPL n-grama en las tres condiciones de la seccion A del probe anterior
    nat = hold_txt[:2500]
    words = nat.split()
    rng = random.Random(RANDOM_SEED); sh = words[:]; rng.shuffle(sh)
    noise = "".join(random.Random(RANDOM_SEED).choice("abcdefghijklmnopqrstuvwxyz \n.,!?'") for _ in range(2500))
    res["R1a_ngram8_holdout"] = {
        "natural": round(ng.ppl_text(hold_txt), 2),
        "permutado": round(ng.ppl_text(" ".join(sh)), 2),
        "ruido": round(ng.ppl_text(noise), 2)}

    # R1b. mismas 200 posiciones, mismas tres condiciones de contexto
    rng2 = random.Random(7)
    positions = sorted(rng2.sample(range(100, len(hold_txt) - 40), 200))
    b_full, b_short, b_shuf = [], [], []
    for pos in positions:
        ctx64 = hold_txt[max(0, pos-64):pos]
        tgt = hold_txt[pos:pos+32]
        w = ctx64.split(); rng2.shuffle(w)
        b_full.append(ng.cont_ppl(ctx64, tgt))
        b_short.append(ng.cont_ppl(ctx64[-8:], tgt))
        b_shuf.append(ng.cont_ppl(" ".join(w), tgt))
    res["R1b_ngram8_contexto"] = {
        "ppl_ctx64": round(float(np.mean(b_full)), 2),
        "ppl_ctx8": round(float(np.mean(b_short)), 2),
        "ppl_ctx64_barajado": round(float(np.mean(b_shuf)), 2),
        "n_posiciones": len(positions)}

    # R2. recuperador de sufijo mas largo, mismos 7 prompts
    corpus_prompts = ["Alice was ", "The Queen of Hearts ", "She said to the "]
    novel_prompts = ["The Rabbit opened the ", "She ordered the Hatter to ",
                     "The Queen drank tea with the Mouse and ", "A grinning cat counted seven "]
    for tag, plist in (("corpus", corpus_prompts), ("nuevos", novel_prompts)):
        rows = []
        for p in plist:
            g = retrieve_gen(train_txt, p, 90, RANDOM_SEED)
            st = copy_stats(g, train_txt)
            st["prompt"] = p; st["tail"] = g[:80]
            rows.append(st)
        res[f"R2_recuperador_prompts_{tag}"] = rows

    # R3. binding con palabras nonce (comprobacion de ausencia en corpus)
    for w in set(sum(([X, Y, A, B] for X, Y, A, B in binding_items()), [])):
        assert w not in train_txt, f"palabra nonce contaminada: {w}"
    items = binding_items()
    det_model, det_ng = [], []
    strict_model = strict_ng = 0
    for X, Y, A, B in items:
        for tmpl, name in ((t1, "T1"), (t2, "T2")):
            ok_m = ok_n = True
            for first, second in ((A, B), (B, A)):  # dos ordenes; respuesta correcta = el ligado a X
                prefix = tmpl(X, Y, first, second)
                good, bad = f" {first}", f" {second}"
                pm_good = cont_ppl_model(m, prefix, good); pm_bad = cont_ppl_model(m, prefix, bad)
                pn_good = ng.cont_ppl(prefix, good);   pn_bad = ng.cont_ppl(prefix, bad)
                ok_m &= pm_good < pm_bad; ok_n &= pn_good < pn_bad
                det_model.append({"tmpl": name, "prefix": prefix, "ppl_correcta": round(pm_good, 2), "ppl_incorrecta": round(pm_bad, 2)})
                det_ng.append({"tmpl": name, "ppl_correcta": round(pn_good, 2), "ppl_incorrecta": round(pn_bad, 2)})
            strict_model += ok_m; strict_ng += ok_n
    n_items = len(items) * 2
    res["R3_binding_nonce"] = {
        "n_items_x2ordenes": n_items,
        "modelo_estricto_aciertos": strict_model,
        "modelo_estricto_pct": round(100 * strict_model / n_items, 1),
        "ngram8_estricto_aciertos": strict_ng,
        "ngram8_estricto_pct": round(100 * strict_ng / n_items, 1),
        "azar_estricto_pct": 50.0,
        "detalle_modelo": det_model, "detalle_ngram8": det_ng}

    out = R / "lab" / "runs" / "nanowsp3_retriever_baseline.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "R3_binding_nonce"}, indent=1, ensure_ascii=False))
    r3 = res["R3_binding_nonce"]
    print(json.dumps({k: v for k, v in r3.items() if not k.startswith("detalle")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
