"""Greedy exact-match evaluation of a trained aux, compatible with skill_delta/evaluate.py conventions.
Arm H, dependent mode: one target forward per generated token; the aux proposes the next token from the
    target's hidden states (logits = head(h_last + aux(states))). A drafter WITHOUT target verification:
    the accuracy gap vs the target measures what verification was buying. Train/eval mismatch to report:
    training saw states over ground-truth prefixes (teacher forcing); evaluation sees states over the aux's
    own continuations (exposure bias, standard, affects H only).
Arm H, --zero-hidden: the aux's hidden-state inputs are zeroed (h_last skip kept): how much of the accuracy
    is the internal states rather than the skip connection plus frozen head.
Arms X and C: standalone; no decoder layer ever runs (embedding table + aux + frozen head only). C crops
    each sequence to its non-pad tokens before the GRU (left padding must not enter the recurrence).
SETS: holdout | selection | control (add2) | range (3-digit x 3-digit) | confirm (fresh-seed 2-digit,
disjoint from every existing set). Hit = exact match of the text before the first newline; <= 6 new tokens;
prompts and the mul2x2/add2 sets are identical to the main experiment.
usage: eval_aux.py MODEL_DIR AUX.safetensors OUT.json --arm h|x|c [--zero-hidden] [--layers 3,11,19,23] SET...
"""
import json, sys, time
import torch
from safetensors.torch import load_file
sys.path.insert(0, "."); sys.path.insert(0, "lab/probes/closed_form_math")
sys.path.insert(0, "lab/probes/behavior_extraction")
from qwen_text import QwenText
from data import make_sets, is_hit
from range_data import make_range_set, make_confirm_set

args = sys.argv[1:]
zero = "--zero-hidden" in args
if zero:
    args.remove("--zero-hidden")
i = args.index("--arm"); arm = args[i + 1]; del args[i:i + 2]
if "--layers" in args:
    i = args.index("--layers"); LAYERS = [int(x) for x in args[i + 1].split(",")]; del args[i:i + 2]
else:
    LAYERS = [3, 11, 19, 23]
model, aux_path, out, sets = args[0], args[1], args[2], args[3:]
q = QwenText(model); tok = q.tok
state = {k[len("aux."):]: w.float() for k, w in load_file(aux_path).items()}
if arm in ("h", "x"):
    aux = torch.nn.Sequential(torch.nn.Linear(state["0.weight"].shape[1], state["0.weight"].shape[0]),
                              torch.nn.SiLU(), torch.nn.Linear(state["2.weight"].shape[1], state["2.weight"].shape[0]))
else:
    aux = torch.nn.ModuleDict({"gru": torch.nn.GRU(1024, state["out.weight"].shape[1], batch_first=True),
                               "out": torch.nn.Linear(state["out.weight"].shape[1], state["out.weight"].shape[0])})
aux.load_state_dict(state); aux.eval()
cap = {}
if arm == "h":
    for i_ in LAYERS:
        q.m.layers[i_].register_forward_hook(
            lambda m, inp, o, i_=i_: cap.__setitem__(i_, (o[0] if isinstance(o, tuple) else o).detach()))


def propose(prompts, max_new=6):
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    gen = [[] for _ in prompts]
    for _ in range(max_new):
        if arm == "h":
            h_last = q.hidden(ids, mask)[:, -1].float()
            st = torch.cat([cap[i_][:, -1].float() for i_ in LAYERS], -1)
            if zero:
                st = torch.zeros_like(st)
            z = h_last + aux(st)
        elif arm == "x":                        # standalone: embeddings + aux + head, no decoder layers
            emb = q.embed(ids).float() * mask[..., None]
            z = aux(emb.sum(1) / mask.sum(1, keepdim=True).clamp(min=1).float())
        else:                                   # arm C: GRU over each non-pad sequence, no decoder layers
            rows = []
            for r in range(len(prompts)):
                seq = ids[r][mask[r] == 1]
                out_, _ = aux["gru"](q.embed(seq[None]).float())
                rows.append(aux["out"](out_[0, -1]))
            z = torch.stack(rows)
        nxt = q.logits_last(z).argmax(-1)
        for r, t in enumerate(nxt.tolist()):
            gen[r].append(t)
        ids = torch.cat([ids, nxt[:, None]], 1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
    return [tok.decode(g, skip_special_tokens=True) for g in gen]


res = {"arm": arm, "zero_hidden": zero, "aux": aux_path}
for s in sets:
    t = time.time()
    if s == "range":
        pool = make_range_set()
    elif s == "confirm":
        pool = make_confirm_set()
    else:
        pool = make_sets("add2" if s == "control" else "mul2x2")["control" if s == "control" else s]
    gen = [g for i_ in range(0, len(pool), 5) for g in propose([p["prompt"] for p in pool[i_:i_ + 5]])]
    verdict = [is_hit(g, p["answer"]) for g, p in zip(gen, pool)]
    res[s] = {"hits": sum(verdict), "n": len(verdict), "verdict": verdict, "gen": gen, "seconds": round(time.time() - t, 1)}
    print(s, sum(verdict), "/", len(verdict), res[s]["seconds"], "s", flush=True)
    json.dump(res, open(out, "w"), indent=1)
