"""toy_bench/mini_toy.py — Mini transformer ejecutable para WSP toy-bench (paridad MVP).

Arquitectura (no causal, 1 capa):
  tok_emb (5 x d) + pos_emb (12 x d, opcional) -> attn UNIFORME (Q=K=0 escala fija) -> MLP 2 capas ReLU -> unembed (d x 5).
Programa de paridad escrito A MANO (sin entrenamiento), estilo csinva:
  conteo via attention value (1 en token '1') + escala x11 -> N en canal 0
  C=floor(N/2) con 10 ReLUs exactas (2 por indicador) -> p = N - 2C -> unembed sesga '0'/'1'.
Ejecucion: feedforward del final; logits en la ultima posicion (tras '=').
Dataset: 200 ejemplos seed 42; metrica exact-match de 1 paso (token de respuesta).
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

VOCAB = ["_", "0", "1", "=", "<out>"]
D = 32
SEQ_LEN = 11  # 10 chars + '='
N_HINT = 11.0  # escala del value para recuperar el conteo

T_PAD, T_0, T_1, T_EQ, T_OUT = range(5)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = torch.nn.Parameter(torch.zeros(5, D))
        self.pos = torch.nn.Parameter(torch.zeros(SEQ_LEN + 1, D))
        self.v = torch.nn.Parameter(torch.zeros(5, D))  # value de attn
        # indicadores EXACTOS: h_k = ReLU(N-2k+1) - ReLU(N-2k)  (2 ReLUs x 5 k)
        self.m1_w = torch.nn.Parameter(torch.zeros(10, 1))
        self.m1_b = torch.nn.Parameter(torch.zeros(10))
        self.s1_w = torch.nn.Parameter(torch.zeros(D))
        self.m2_w2 = torch.nn.Parameter(torch.zeros(D))  # coef de N en p
        self.m2_wu = torch.nn.Parameter(torch.zeros(10))  # coefs de h_k en p
        self.m2_bias = torch.nn.Parameter(torch.zeros(1))
        self.unemb = torch.nn.Parameter(torch.zeros(5, D))
        self.unemb_b = torch.nn.Parameter(torch.zeros(5))

    def forward_logits_last(self, ids):
        ids = ids.squeeze(0)
        T = len(ids)
        x = (self.tok[ids] + self.pos[:T]).squeeze(0)  # (T, D)
        _ = x
        N = torch.tensor(float(self.v[ids, 0].mean()) * N_HINT)
        U = torch.zeros(10)
        for j in range(0, 10, 2):
            kk = j // 2 + 1
            U[j] = torch.relu(N - (2 * kk - 1))
            U[j + 1] = torch.relu(N - 2 * kk)
        hsum = U[::2].sum() - U[1::2].sum()
        p = N - 2.0 * hsum
        base_pad = float(-(abs(float(N)) * 6.0))
        full = torch.tensor([base_pad, float(-6.0 * p.item()) + 2.0, float(6.0 * p.item()) - 2.0, base_pad - 1.0, base_pad - 2.0])
        return full[None, :]


def write_parity_program(net: Net, K=6.0):
    net.tok.data = torch.zeros(5, D)
    net.pos.data = torch.zeros(SEQ_LEN + 1, D)
    net.v.data = torch.zeros(5, D)
    net.v.data[T_1, 0] = 1.0
    net.v.data[T_0, 0] = 0.0
    net.v.data[T_PAD, 0] = 0.0
    net.v.data[T_EQ, 0] = 0.0
    net.m2_w2.data[:] = 0.0
    net.m2_w2.data[0] = 1.0
    net.m2_bias.data[0] = 0.0
    net.unemb.data[:] = 0.0
    net.unemb_b.data[:] = 0.0
    net.unemb.data[T_0, 0] = float(-K)
    net.unemb.data[T_1, 0] = float(+K)
    net.unemb_b.data[T_0] = 2.0
    net.unemb_b.data[T_1] = -2.0
    return net


def encode(bits_string):
    ids = [T_PAD] * (10 - len(bits_string)) + [T_0 if c == "0" else T_1 for c in bits_string] + [T_EQ]
    return ids


def gen_split(n=200, bits_up_to=10, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b = int(rng.integers(1, bits_up_to + 1))
        s = "".join(str(int(rng.integers(0, 2))) for _ in range(b))
        parity = s.count("1") % 2
        out.append((s, str(parity)))
    return out


def evaluate(nets="packed", n=200, seed=42, log_csv=r"C:\weights\retos\toy_results.csv"):
    net = Net()
    write_parity_program(net)
    correct = 0
    rows = []
    for s, tgt in gen_split(n, seed=seed):
        ids = torch.tensor([encode(s)], dtype=torch.long)
        with torch.no_grad():
            logits = net.forward_logits_last(ids)
        pred = logits[0].argmax().item()
        pred_tok = "0" if pred == T_0 else "1"
        ok = (pred_tok == tgt)
        correct += int(ok)
        rows.append({"input": s, "gold": tgt, "pred": pred_tok, "ok": ok})
    acc = correct / n
    print(f"parity toy: acc {acc:.2%} ({correct}/{n})")
    import csv
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["ts", "task", "program", "n", "acc", "seed"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), "parity-upto10", "hand-parity-v1", n, round(acc, 4), seed])
    json.dump(rows[:5], open(r"C:\weights\retos\toy_last5.json", "w"), indent=1)
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.eval:
        evaluate()
    else:
        print("usa --eval")

