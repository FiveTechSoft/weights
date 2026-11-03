"""toy_bench/mini_toy.py — mini transformer GENERIC para WSP toy-bench (familia: escribir pesos from-scratch).

Arquitectura (documentada, no-causal, torch CPU):
  att (D) = mean_t v[ids_t]                       (broadcast T posiciones)
  att_s = att * s1                                 (program asigna s1)
  h1 = ReLU(W1 @ att_s + b1)                       (U1=12 units)
  h2 = ReLU(W2 @ h1 + Wskip @ att_s + b2)          (U2=6 units)
  logits = Wout @ h2 + bout                        (vocab) -> se lee en la ULTIMA pos

Reglas del programa (driver):
  def set_weights(net, task): asignaciones DIRECTAS (sin optimizers, sin autograd).
  Corre bajo torch.no_grad(); juez >= 0.97 acc (200 ejemplos, seed 42).

CLI:
  python mini_toy.py --eval                         (demo hand-parity)
  python mini_toy.py --task count3 --program driver.py
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

D = 32
U1, U2 = 12, 6
CSV = Path(r"C:\weights\retos\toy_results.csv")
N_HINT = "1"  # programa le pone escala en s1


def make_net(vocab_size, seqlen):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = torch.nn.Parameter(torch.zeros(vocab_size, D))
            self.pos = torch.nn.Parameter(torch.zeros(seqlen, D))
            self.Q = torch.nn.Parameter(torch.zeros(D, D))  # per-position keys ruta (0 => uniforme)
            self.K = torch.nn.Parameter(torch.zeros(D, D))
            self.V = torch.nn.Parameter(torch.zeros(D, D))  # head transformer pura; v simplificada por canal
            self.s1 = torch.nn.Parameter(torch.zeros(D))
            self.W1 = torch.nn.Parameter(torch.zeros(U1, D))
            self.b1 = torch.nn.Parameter(torch.zeros(U1))
            self.W2 = torch.nn.Parameter(torch.zeros(U2, U1))
            self.Wskip = torch.nn.Parameter(torch.zeros(U2, D))
            self.b2 = torch.nn.Parameter(torch.zeros(U2))
            self.Wout = torch.nn.Parameter(torch.zeros(vocab_size, U2))
            self.bout = torch.nn.Parameter(torch.zeros(vocab_size))

        def forward_logits_all(self, ids):
            xs = self.tok[ids] + self.pos[: len(ids)]  # (T, D)
            S = (xs @ self.K.T) @ (xs @ self.Q.T).T / (D ** 0.5)  # (T, T)
            A = torch.softmax(S, dim=-1)
            z = A @ (xs @ self.V.T)  # (T, D): per-position attended vector
            h1 = torch.relu(z @ self.W1.T + self.b1)
            h2 = torch.relu(h1 @ self.W2.T + xs @ self.Wskip.T + self.b2)
            return h2 @ self.Wout.T + self.bout

    return Net()


VR = 0
V0, V1, VQ = 1, 2, 3
VS, VN = 4, 5
VH = 6
VPOWER = ["_", "0", "1", "=", "S", "N", "H"]  # 7 tokens
HANDS = {}


def gen_parity_input(n, seed):
    rng = np.random.default_rng(seed)
    szy = []
    for _ in range(n):
        b = int(rng.integers(1, 11))
        s = "".join(str(int(rng.integers(0, 2))) for _ in range(b))
        szy.append((s, str(s.count("1") % 2)))
    return szy


def encode_parity(s):
    bits = [V1 if c == "1" else V0 for c in s]
    return [VR] * (10 - len(bits)) + bits + [VQ]


def gen_count(n, seed, target_eq):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b = int(rng.integers(4, 11))
        s = "".join(str(int(rng.integers(0, 2))) for _ in range(b))
        cnt = s.count("1")
        if target_eq:
            tgt = "S" if cnt == 3 else "N"
        else:
            tgt = "H" if cnt >= 3 else "N"
        out.append((s, tgt))
    return out


def encode_generic(s):
    bits = [V1 if c == "1" else V0 for c in s]
    return [VR] * (10 - len(bits)) + bits + [VQ]


TASKS = {
    "parity": {"vocab": VPOWER, "seqlen": 11, "generator": lambda n, seed: gen_parity_input(n, seed),
               "encode": encode_parity, "hint": "output '0' even / '1' odd"},
    "count3": {"vocab": VPOWER, "seqlen": 11, "generator": lambda n, seed: gen_count(n, seed, True),
               "encode": encode_generic, "hint": "output 'S' si count=='3' else 'N'"},
    "geq3": {"vocab": VPOWER, "seqlen": 11, "generator": lambda n, seed: gen_count(n, seed, False),
             "encode": encode_generic, "hint": "output 'H' si count>=3 else 'N'"},
}


def _hand_parity(net, task):
    """arch nueva: z0 = (mean de xs·V) con V=e0, tok[V1,0]=1: z0 = count/11: pesos 11/1 para N"""
    net.tok.data.fill_(0.0)
    net.tok.data[V1, 0] = 1.0
    net.pos.data.fill_(0.0)
    net.Q.data.fill_(0.0)
    net.K.data.fill_(0.0)
    net.V.data.fill_(0.0)
    net.V.data[0, 0] = 1.0
    net.W1.data.fill_(0.0)
    net.b1.data.fill_(0.0)
    for k in range(1, 6):
        ev = 2 * (k - 1)
        net.W1.data[ev, 0] = 11.0
        net.b1.data[ev] = 1.0 - 2 * k
        net.W1.data[ev + 1, 0] = 11.0
        net.b1.data[ev + 1] = -2.0 * k
    net.W1.data[10, 0] = 11.0  # passthrough N
    net.b1.data[10] = 0.0
    net.W2.data.fill_(0.0)
    net.Wskip.data.fill_(0.0)
    net.W2.data[0, 10] = 1.0
    for k in range(1, 6):
        ev = 2 * (k - 1)
        net.W2.data[0, ev] = -2.0
        net.W2.data[0, ev + 1] = 2.0
    net.b2.data.fill_(0.0)
    net.Wout.data.fill_(0.0)
    net.Wout.data[V0, 0] = -6.0
    net.Wout.data[V1, 0] = 6.0
    net.Wout.data[VR, 0] = -6.0
    net.Wout.data[VQ, 0] = -6.0
    net.bout.data.fill_(-7.0)
    net.bout.data[V0] = 2.0
    net.bout.data[V1] = -2.0
    return net


HANDS["parity"] = _hand_parity


def run_program(task, program_path=None, n=200, seed=42):
    tc = TASKS[task]
    V = tc["vocab"]
    vg = tc["generator"](n, seed=seed)
    if program_path is None and task in HANDS:
        net = make_net(len(V), tc["seqlen"])
        with torch.no_grad():
            HANDS[task](net, tc)
    else:
        spec = importlib.util.spec_from_file_location("wsp_prog", str(program_path))
        prog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prog)
        assert hasattr(prog, "set_weights"), "def set_weights(net, task)"
        net = make_net(len(V), tc["seqlen"])
        with torch.no_grad():
            prog.set_weights(net, tc)
    correct, samples = 0, []
    for i, (inp, tgt) in enumerate(vg):
        ids = torch.tensor(tc["encode"](inp), dtype=torch.long)
        with torch.no_grad():
            logits = net.forward_logits_all(ids)
        pred = V[int(torch.argmax(logits[-1]).item())]
        ok = (pred == tgt)
        correct += int(ok)
        if i < 4:
            samples.append({"input": inp, "gold": tgt, "pred": pred, "ok": ok})
    acc = correct / n
    return round(acc, 4), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--task", default="parity")
    ap.add_argument("--program", default=None)
    ap.add_argument("-n", type=int, default=200)
    args = ap.parse_args()
    acc, samples = run_program(args.task, args.program, n=args.n)
    name = Path(args.program).stem if args.program else f"hand_{args.task}"
    print(f"ACC {acc}")
    print(f"SAMPLES {json.dumps(samples, ensure_ascii=False)}")
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["ts", "task", "program", "n", "acc"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), args.task, name, args.n, acc])


if __name__ == "__main__":
    main()

