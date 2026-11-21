"""Tests for lab/probes/neuron_communities (phase A of its PLAN.md). Pure functions on tiny inputs, plus the two
method checks of the plan on a tiny random Qwen3.5 text model in float32: the layered gradient route against
end-to-end autograd, and the one-row ablation route against a full hooked forward. CPU only."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab.probes.neuron_communities import communities as cm  # noqa: E402
from lab.probes.neuron_communities import attribute as at  # noqa: E402


def tiny_model(seed=0):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    cfg = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, full_attention_interval=4,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
                            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8, vocab_size=64,
                            linear_conv_kernel_dim=4, max_position_embeddings=512, tie_word_embeddings=False)
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(cfg).eval().float()


class PMIGraph(unittest.TestCase):
    def test_edges_need_positive_pmi_and_min_cooccurrence(self):
        # neurons 1 and 2 always together (6 of 10 positions); neuron 3 at every position (PMI with anything = 0)
        sel = [[1, 2, 3]] * 6 + [[3, 4]] * 4
        G = cm.pmi_graph(sel, min_co=5)
        self.assertTrue(G.has_edge(1, 2))
        self.assertAlmostEqual(G[1][2]["weight"], math.log(6 * 10 / (6 * 6)), places=6)
        self.assertFalse(G.has_edge(1, 3))            # PMI = log(6*10/(6*10)) = 0, not > 0
        self.assertFalse(G.has_edge(3, 4))            # co-selected 4 < min_co
        self.assertEqual(G.nodes[3]["count"], 10)

    def test_louvain_separates_two_groups_and_is_seeded(self):
        sel = [[0, 1, 2]] * 8 + [[10, 11, 12]] * 8 + [[0, 1, 2, 10]] * 1
        G = cm.pmi_graph(sel, min_co=5)
        comms = cm.louvain(G, seed=2026, cap=128)
        self.assertEqual(sorted(map(sorted, comms)), [[0, 1, 2], [10, 11, 12]])
        self.assertEqual(comms, cm.louvain(G, seed=2026, cap=128))

    def test_louvain_cap_keeps_highest_weighted_degree(self):
        sel = [list(range(6))] * 10 + [[0, 1, 2]] * 4 + [[6]] * 10
        G = cm.pmi_graph(sel, min_co=5)
        full = cm.louvain(G, seed=2026, cap=128)
        self.assertEqual(sorted(full[0]), list(range(6)))
        deg = dict(G.subgraph(full[0]).degree(weight="weight"))
        top3 = set(sorted(full[0], key=lambda n: (-deg[n], n))[:3])
        comms = cm.louvain(G, seed=2026, cap=3)
        self.assertEqual(len(comms), 1)
        self.assertEqual(comms[0], top3)

    def test_always_on_threshold(self):
        sel = [[7, 8]] * 9 + [[8]]                      # 7 in 9/10, 8 in 10/10
        self.assertEqual(cm.always_on(sel, frac=0.9), {7, 8})
        sel = [[7, 8]] * 8 + [[8]] * 2                  # 7 in 8/10
        self.assertEqual(cm.always_on(sel, frac=0.9), {8})

    def test_candidate_is_largest_pos_minus_neg_mass(self):
        comms = [{0, 1}, {2, 3}, {4}]
        mass_pos = {0: 1.0, 1: 1.0, 2: 5.0, 3: 5.0, 4: 3.0}
        mass_neg = {0: 0.0, 1: 0.0, 2: 4.5, 3: 4.5, 4: 0.5}
        best, scores = cm.candidate(comms, mass_pos, mass_neg)
        self.assertEqual(best, {4})                      # 2.5 > 2.0 > 1.0
        self.assertEqual([round(s["score"], 3) for s in scores], [2.5, 2.0, 1.0])

    def test_merge_topk_across_layers_uses_global_ids(self):
        F = 10
        per_layer = {0: (np.array([3, 1]), np.array([0.5, -2.0])),
                     2: (np.array([7, 0]), np.array([1.5, 0.1]))}
        gids, vals = cm.merge_topk(per_layer, k=3, width=F)
        self.assertEqual(list(gids), [cm.global_id(0, 1, F), cm.global_id(2, 7, F), cm.global_id(0, 3, F)])
        self.assertEqual(list(vals), [-2.0, 1.5, 0.5])
        self.assertEqual(cm.split_id(cm.global_id(2, 7, F), F), (2, 7))

    def test_verdict_rule(self):
        a1 = dict(drop=0.4, placebo=0.05)
        self.assertEqual(cm.verdict(a1, dict(retained=0.6, random=0.2)), "community_found")
        self.assertEqual(cm.verdict(a1, dict(retained=0.4, random=0.2)), "necessary_not_sufficient")
        self.assertEqual(cm.verdict(a1, dict(retained=0.6, random=0.3)), "necessary_not_sufficient")
        self.assertEqual(cm.verdict(dict(drop=0.4, placebo=0.08), dict(retained=0.6, random=0.2)), "community_found")
        self.assertEqual(cm.verdict(dict(drop=0.4, placebo=0.09), dict(retained=0.6, random=0.2)), "nothing_sparse")  # < 5x
        self.assertEqual(cm.verdict(dict(drop=0.6, placebo=0.11), dict(retained=0.6, random=0.2)), "nothing_sparse")  # > 10 %
        self.assertEqual(cm.verdict(dict(drop=0.2, placebo=0.01), dict(retained=0.6, random=0.2)), "nothing_sparse")  # < 25 %
        self.assertEqual(cm.verdict(dict(drop=0.4, placebo=0.08), None), "necessary_not_sufficient")


class LayeredGradients(unittest.TestCase):
    def test_layered_route_matches_end_to_end_autograd(self):
        m = tiny_model()
        core, head = at.model_parts(m)
        ids = torch.randint(0, 64, (24,)).tolist()
        positions = [23, 22, 15]
        tok = 5
        ref = at.end_to_end_grads(core, head, ids, positions, tok)
        for group in (1, 2, 4):
            got = at.attribute_sequence(core, head, ids, positions, tok, group=group, keep=8, device="cpu",
                                        return_grads=True)
            for l in got["grad_in"]:                       # gradient at the input of each group's first layer
                a, b = got["grad_in"][l].double(), ref["grad_in"][l].double()
                self.assertLess(float((a - b).norm() / b.norm()), 1e-3, (group, l))
            for l in range(len(core.layers)):              # a * grad at down_proj's input, every layer
                a, b = got["attr_full"][l].double(), (ref["a"][l] * ref["a_grad"][l]).double()
                self.assertLess(float((a - b).norm() / b.norm()), 1e-3, (group, l))
        # top-k content: the kept indices are the k largest |attr| at the position itself
        l, p = 1, 0
        full = (ref["a"][l] * ref["a_grad"][l])[p, positions[p]].abs()
        idx, _ = got["same"][l][p]
        self.assertEqual(set(idx.tolist()), set(torch.topk(full, 8).indices.tolist()))
        # per-layer sum sanity number equals the sum of the same-position attributions
        self.assertAlmostEqual(float(got["layer_sum"][l, p]),
                               float((ref["a"][l] * ref["a_grad"][l])[p, positions[p]].sum()), places=4)
        # causality: no gradient after the position
        self.assertEqual(float(ref["grad_in"][0][2, 16:].abs().sum()), 0.0)


class Ablation(unittest.TestCase):
    def test_one_row_route_matches_full_forward_with_hooks(self):
        m = tiny_model(1)
        core, head = at.model_parts(m)
        ids = torch.randint(0, 64, (30,)).tolist()
        t_e, tok = 29, 7
        conds = [{}, {1: [0, 5, 9]}, {1: [0, 5, 9], 3: list(range(60))}, {2: list(range(64))}, {0: [1], 3: [2]}]
        got = at.ablate_probabilities(core, head, ids, t_e, tok, conds, device="cpu", batch=2)
        ref = [at.full_forward_prob(core, head, ids, t_e, tok, c) for c in conds]
        for g, r in zip(got, ref):
            self.assertAlmostEqual(g, r, places=5)
        self.assertNotEqual(got[0], got[2])                     # zeroing 60 of 64 neurons changes something
        self.assertEqual(got[0], at.ablate_probabilities(core, head, ids, t_e, tok, [{}], device="cpu")[0])

    def test_condition_builders(self):
        F = 6
        C = {cm.global_id(2, 1, F), cm.global_id(2, 4, F), cm.global_id(5, 0, F)}
        S0 = {cm.global_id(2, 5, F), cm.global_id(9, 3, F)}
        self.assertEqual(at.zero_community(C, F), {2: [1, 4], 5: [0]})
        self.assertEqual(at.zero_all_but(C, S0, F), {2: [0, 2, 3], 5: [1, 2, 3, 4, 5]})
        plac = at.random_same_size(C, F, n_sets=3, seed=2026)
        self.assertEqual(len(plac), 3)
        for p in plac:
            self.assertEqual({l: len(v) for l, v in p.items()}, {2: 2, 5: 1})
        self.assertEqual(plac, at.random_same_size(C, F, n_sets=3, seed=2026))
        self.assertEqual(at.per_layer(C, F), {2: [1, 4], 5: [0]})


@unittest.skipUnless((ROOT / "hf_cache" / "qwen35-08b-hf").exists(), "needs hf_cache/qwen35-08b-hf (tokenizer)")
class Positions(unittest.TestCase):
    def test_positions_from_pos_and_neg_rows(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(ROOT / "hf_cache" / "qwen35-08b-hf"))
        resp = "First we add the numbers. Then we check the sum. So the answer is 7."
        pos = {"prompt": "What is 3 + 4?", "think": True, "response": resp, "focus": [len(resp) - 20, len(resp)]}
        neg = {"prompt": "What is 3 + 4?", "think": True, "response": resp[:26], "focus": [0, 26]}
        d = at.positions_for(tok, pos, neg, k_before=3)
        p = tok.apply_chat_template([{"role": "user", "content": pos["prompt"]}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
        full = tok(p + resp, add_special_tokens=False)["input_ids"]
        self.assertEqual(d["ids"], full)
        self.assertEqual(d["pos_event"], len(full) - 1)
        self.assertEqual(d["pos_positions"], [len(full) - 1, len(full) - 2, len(full) - 3, len(full) - 4])
        enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
        last = max(k for k, (s, e) in enumerate(enc["offset_mapping"]) if e <= 26)
        self.assertEqual(d["neg_event"], d["prompt_len"] + last)
        self.assertEqual(len(d["neg_positions"]), 4)
        self.assertTrue(d["neg_event"] < d["pos_event"] - 3)
        self.assertTrue(tok.decode(full[d["prompt_len"]: d["neg_event"] + 1]).endswith("numbers."))


if __name__ == "__main__":
    unittest.main()
