import unittest

import torch

from engine import stream_score as ss


def _tiny_cfg():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=False, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    return cfg


class StreamParity(unittest.TestCase):
    def test_matches_the_library_full_forward_on_a_tiny_random_model(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        torch.manual_seed(0)
        cfg = _tiny_cfg()
        model = m.Qwen3_5ForCausalLM(cfg).float().eval()
        sd = model.state_dict()
        ids_list = [[3, 14, 15, 92, 65, 35, 89, 79, 32, 38], [7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5]]
        starts = [4, 1]
        W = sd["lm_head.weight"]

        def head():
            for s in range(0, W.shape[0], 30):  # uneven chunks on purpose
                yield s, W[s:s + 30]

        got = ss.stream_nll(cfg, ids_list, lambda ids: sd["model.embed_tokens.weight"][ids],
                            lambda L, layer: ss.fill_from_state(layer, {k[len(f"model.layers.{L}."):]: v for k, v in sd.items() if k.startswith(f"model.layers.{L}.")}),
                            sd["model.norm.weight"], head, starts)
        with torch.no_grad():
            for ids, st, g in zip(ids_list, starts, got):
                logits = model(torch.tensor(ids)[None]).logits[0, :-1]
                nll = torch.nn.functional.cross_entropy(logits, torch.tensor(ids[1:]), reduction="none")[st - 1:]
                self.assertEqual(g.shape, nll.shape)
                self.assertTrue(torch.allclose(g, nll, atol=1e-4), (g - nll).abs().max())

    def test_spill_gives_the_same_numbers_and_head_is_iterated_once(self):
        import tempfile
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        torch.manual_seed(1)
        cfg = _tiny_cfg()
        sd = m.Qwen3_5ForCausalLM(cfg).float().eval().state_dict()
        ids_list = [[3, 14, 15, 92, 65, 35, 89, 79], [7, 1, 8, 2, 8, 1, 8, 2, 8, 4], [5, 9, 0, 4, 5, 3]]
        W = sd["lm_head.weight"]
        passes = []

        def head():
            passes.append(1)
            for s in range(0, W.shape[0], 30):
                yield s, W[s:s + 30]

        run = lambda **kw: ss.stream_nll(
            cfg, ids_list, lambda ids: sd["model.embed_tokens.weight"][ids],
            lambda L, layer: ss.fill_from_state(layer, {k[len(f"model.layers.{L}."):]: v for k, v in sd.items() if k.startswith(f"model.layers.{L}.")}),
            sd["model.norm.weight"], head, [1, 1, 1], **kw)
        a = run()
        with tempfile.TemporaryDirectory() as d:
            b = run(spill_dir=d)
        self.assertEqual(len(passes), 2)  # one head pass per run, not one per prompt
        for x, y in zip(a, b):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_wrong_layer_keys_are_refused(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        cfg = _tiny_cfg()
        layer = m.Qwen3_5DecoderLayer(cfg, 0)
        with self.assertRaises(KeyError):
            ss.fill_from_state(layer, {"mlp.gate_proj.weight": torch.zeros(1)})

    def test_scoring_is_causal(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        torch.manual_seed(1)
        cfg = _tiny_cfg()
        sd = m.Qwen3_5ForCausalLM(cfg).float().state_dict()
        W = sd["lm_head.weight"]
        run = lambda ids: ss.stream_nll(cfg, [ids], lambda i: sd["model.embed_tokens.weight"][i],
                                        lambda L, layer: ss.fill_from_state(layer, {k[len(f"model.layers.{L}."):]: v for k, v in sd.items() if k.startswith(f"model.layers.{L}.")}),
                                        sd["model.norm.weight"], lambda: iter([(0, W)]), [1])[0]
        a = run([5, 6, 7, 8, 9, 10])
        b = run([5, 6, 7, 8, 9, 11])  # last token changed: NLL of earlier tokens must not move
        self.assertTrue(torch.allclose(a[:-1], b[:-1], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
