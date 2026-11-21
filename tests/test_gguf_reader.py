"""engine.gguf_reader: a local GGUF file read through the reader interface of engine.weight_diff (keys / shape / get /
get_rows), and an HF-layout view of it (names, conv1d shape, A_log, the linear-attention head order) so model_report
can compare a GGUF model against a safetensors base tensor by tensor.

Fixture: a tiny GGUF written with gguf.GGUFWriter from known tensors (BF16, F16, F32), with llama.cpp's conversion
applied by hand (ssm_a = -exp(A_log), conv1d squeezed, value heads interleaved), so every expected value is exact.
Real-file checks (skipped when the files are absent): hf_cache/h4/A0-Q6_K.gguf against hf_cache/qwen35-9b-hf.
"""
import gc
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402

import engine.gguf_reader as gr  # noqa: E402
import engine.weight_diff as wd  # noqa: E402

A0 = ROOT / "hf_cache" / "h4" / "A0-Q6_K.gguf"
HF9B = ROOT / "hf_cache" / "qwen35-9b-hf"

NV, NK, HD = 4, 2, 2                       # value heads, key heads, head dim of the fixture's linear-attention block
PERM = [(n % NK) * (NV // NK) + n // NK for n in range(NV)]   # llama.cpp: GGUF head n holds HF head PERM[n]
BLOCK = [h * HD + d for h in PERM for d in range(HD)]        # the same over rows of HD per head


def _bf16(x: torch.Tensor) -> np.ndarray:
    """A float tensor as the uint8 bytes of its bfloat16 encoding (what GGUFWriter stores for raw_dtype=BF16)."""
    return gguf.quants.quantize(x.float().contiguous().numpy(), T.BF16)


def hf_tensors():
    """The HF-layout truth for block 0 (one linear-attention layer) plus a full-attention block 1 and the MTP block."""
    g = torch.Generator().manual_seed(7)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    t = {
        "model.layers.0.input_layernorm.weight": torch.arange(4, dtype=torch.float32) + 1,
        "model.layers.0.linear_attn.A_log": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "model.layers.0.linear_attn.dt_bias": torch.tensor([-2.5, 0.25, 1.0, 3.0]).to(torch.bfloat16),
        "model.layers.0.linear_attn.in_proj_a.weight": r(NV, 4).to(torch.bfloat16),
        "model.layers.0.linear_attn.in_proj_b.weight": r(NV, 4).to(torch.bfloat16),
        "model.layers.0.linear_attn.in_proj_z.weight": r(NV * HD, 4).to(torch.bfloat16),
        "model.layers.0.linear_attn.in_proj_qkv.weight": r(2 * NK * HD + NV * HD, 4).to(torch.float16),
        "model.layers.0.linear_attn.conv1d.weight": r(2 * NK * HD + NV * HD, 1, 3),
        "model.layers.0.linear_attn.out_proj.weight": r(4, NV * HD),
        "model.layers.0.linear_attn.norm.weight": torch.ones(HD),
        "model.layers.0.mlp.down_proj.weight": r(4, 12).to(torch.float16),
        "model.layers.1.self_attn.q_proj.weight": r(8, 4).to(torch.bfloat16),
        "model.layers.1.self_attn.q_norm.weight": torch.tensor([0.25, -0.5, 0.0, 0.125]),
        "model.embed_tokens.weight": r(10, 4).to(torch.bfloat16),
        "lm_head.weight": r(10, 4).to(torch.bfloat16),
        "model.norm.weight": torch.tensor([1.25, 1.0, 0.75, 0.5]),
    }
    return t


def gguf_tensors(hf):
    """What llama.cpp's converter writes for hf: names, ssm_a = -exp(A_log), conv1d (C, 1, K) -> (C, K), value heads
    of the linear attention interleaved by key group (GGUF head n = HF head PERM[n]) on every tensor indexed by
    value head: A_log, dt_bias, in_proj_a / in_proj_b rows, in_proj_z rows, the v part of in_proj_qkv and conv1d
    rows, out_proj columns. Plus an MTP block (blk.2) that the HF view must hide."""
    L = "model.layers.0.linear_attn."
    p, b = torch.tensor(PERM), torch.tensor(BLOCK)
    kv = 2 * NK * HD
    qkv = hf[L + "in_proj_qkv.weight"]
    conv = hf[L + "conv1d.weight"].squeeze(1)
    g = {
        # Qwen3.5's RMSNorm scales by (1 + w): llama.cpp stores w + 1 for the attention / MLP / final norms
        "blk.0.attn_norm.weight": (hf["model.layers.0.input_layernorm.weight"] + 1, T.F32),
        "blk.0.ssm_a": ((-torch.exp(hf[L + "A_log"]))[p], T.F32),
        "blk.0.ssm_dt.bias": (hf[L + "dt_bias"].float()[p], T.F32),
        "blk.0.ssm_alpha.weight": (hf[L + "in_proj_a.weight"][p], T.BF16),
        "blk.0.ssm_beta.weight": (hf[L + "in_proj_b.weight"][p], T.BF16),
        "blk.0.attn_gate.weight": (hf[L + "in_proj_z.weight"][b], T.BF16),
        "blk.0.attn_qkv.weight": (torch.cat([qkv[:kv], qkv[kv:][b]]), T.F16),
        "blk.0.ssm_conv1d.weight": (torch.cat([conv[:kv], conv[kv:][b]]), T.F32),
        "blk.0.ssm_out.weight": (hf[L + "out_proj.weight"][:, b], T.F32),
        "blk.0.ssm_norm.weight": (hf[L + "norm.weight"], T.F32),
        "blk.0.ffn_down.weight": (hf["model.layers.0.mlp.down_proj.weight"], T.F16),
        "blk.1.attn_q.weight": (hf["model.layers.1.self_attn.q_proj.weight"], T.BF16),
        "token_embd.weight": (hf["model.embed_tokens.weight"], T.BF16),
        "output.weight": (hf["lm_head.weight"], T.BF16),
        "output_norm.weight": (hf["model.norm.weight"] + 1, T.F32),
        "blk.1.attn_q_norm.weight": (hf["model.layers.1.self_attn.q_norm.weight"] + 1, T.F32),
        "blk.2.nextn.eh_proj.weight": (torch.randn(4, 8), T.F32),
        "blk.2.attn_norm.weight": (torch.ones(4), T.F32),
    }
    return g


def write_fixture(path: Path, g, arch="qwen35"):
    w = gguf.GGUFWriter(str(path), arch=arch)
    w.add_block_count(3)
    for name, (x, tt) in g.items():
        if tt == T.BF16:
            w.add_tensor(name, _bf16(x), raw_dtype=T.BF16)
        else:
            w.add_tensor(name, x.contiguous().numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _expected_dtype(x, tt):
    return {T.BF16: torch.bfloat16, T.F16: torch.float16, T.F32: torch.float32}[tt]


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.gguf"
        self.hf = hf_tensors()
        self.g = gguf_tensors(self.hf)
        write_fixture(self.path, self.g)
        self.r = gr.LocalGGUF(self.path)

    def tearDown(self):
        self.r.close()
        del self.r
        gc.collect()
        self.tmp.cleanup()

    def test_keys_and_torch_order_shapes(self):
        self.assertEqual(self.r.keys(), sorted(self.g))
        for name, (x, tt) in self.g.items():
            self.assertEqual(self.r.shape(name), tuple(x.shape), name)
        # the orientation claim, checked against gguf-py itself: ReaderTensor.shape is ne order (reversed), and for
        # unquantised tensors ReaderTensor.data is already laid out in torch order
        rd = gguf.GGUFReader(str(self.path))
        tm = {t.name: t for t in rd.tensors}
        self.assertEqual(tuple(int(v) for v in tm["blk.1.attn_q.weight"].shape), (4, 8))
        self.assertEqual(tuple(tm["blk.0.ffn_down.weight"].data.shape), (4, 12))
        self.assertEqual(tuple(int(v) for v in tm["blk.0.ffn_down.weight"].shape), (12, 4))
        del rd
        gc.collect()

    def test_get_returns_file_dtype_and_exact_values(self):
        for name, (x, tt) in self.g.items():
            y = self.r.get(name)
            self.assertEqual(y.dtype, _expected_dtype(x, tt), name)
            self.assertTrue(torch.equal(y, x.to(_expected_dtype(x, tt))), name)

    def test_get_rows_reads_only_those_rows(self):
        for name in ("blk.1.attn_q.weight", "blk.0.ffn_down.weight", "blk.0.ssm_conv1d.weight", "token_embd.weight"):
            full = self.r.get(name)
            self.assertTrue(torch.equal(self.r.get_rows(name, 2, 3), full[2:5]), name)
            self.assertTrue(torch.equal(self.r.get_rows(name, 1, 100), full[1:]), name)  # clipped at the end
            self.assertEqual(tuple(self.r.get_rows(name, 100, 3).shape), (0,) + tuple(full.shape[1:]), name)
            self.assertEqual(self.r.get_rows(name, 100, 3).dtype, full.dtype, name)
        v = self.r.get("blk.0.attn_norm.weight")
        self.assertTrue(torch.equal(self.r.get_rows("blk.0.attn_norm.weight", 1, 2), v[1:3]))

    def test_close_releases_the_mapping_so_the_file_can_be_renamed(self):
        self.r.close()
        new = self.path.with_suffix(".moved.gguf")
        self.path.rename(new)       # PermissionError on Windows while the mmap is alive
        new.rename(self.path)

    def test_hf_view_renames_reshapes_unpermutes_and_hides_mtp(self):
        self.r.close()
        h = gr.open_gguf_as_hf(self.path)
        try:
            self.assertEqual(h.keys(), sorted(self.hf))
            self.assertEqual(h.hidden(), ["blk.2.attn_norm.weight", "blk.2.nextn.eh_proj.weight"])
            self.assertEqual(h.original("model.layers.1.self_attn.q_proj.weight"), "blk.1.attn_q.weight")
            self.assertEqual(h.geometry(0), {"n_v_heads": NV, "n_k_heads": NK, "head_dim": HD})
            self.assertIsNone(h.geometry(1))
            self.assertEqual(h.architecture, "qwen35")
            self.assertTrue(h.norm_offset)
            for name, x in self.hf.items():
                self.assertEqual(h.shape(name), tuple(x.shape), name)
                y = h.get(name)
                self.assertEqual(tuple(y.shape), tuple(x.shape), name)
                if name.endswith("A_log"):
                    self.assertTrue(torch.allclose(y, x, atol=1e-6), name)
                else:
                    self.assertTrue(torch.equal(y.float(), x.float()), name)
                # rows of the view == rows of the full view, whatever the stored order
                self.assertTrue(torch.equal(h.get_rows(name, 1, 2).float(), y[1:3].float()), name)
                self.assertTrue(torch.equal(h.get_rows(name, 0, 1000).float(), y.float()), name)
        finally:
            h.close()
            del h
            gc.collect()


class ReaderForCase(unittest.TestCase):
    """model_report.reader_for on a .gguf path returns the HF-named view (so `step model_report model=<file.gguf>` works)."""

    def test_reader_for_opens_a_gguf_as_the_hf_view(self):
        from engine import model_report as mr
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.gguf"
            hf = hf_tensors()
            write_fixture(path, gguf_tensors(hf))
            h = mr.reader_for(str(path))
            try:
                self.assertEqual(h.keys(), sorted(hf))
                self.assertIn("blk.2.nextn.eh_proj.weight", h.hidden())
                self.assertEqual(tuple(h.shape("model.layers.1.self_attn.q_proj.weight")),
                                 tuple(hf["model.layers.1.self_attn.q_proj.weight"].shape))
            finally:
                h.close()
                del h
                gc.collect()


class FusedFixtureCase(unittest.TestCase):
    """SenseNova's llama.cpp fork fuses two pairs of linear-attention tensors: attn_qkv_gate = [in_proj_qkv ; in_proj_z]
    (GGUF order, so the v and z blocks carry the head interleave) and ssm_beta_alpha = [in_proj_b ; in_proj_a] rows.
    The HF view must split them back and apply the same transforms as for the unfused names."""

    def test_hf_view_splits_fused_linear_attention_tensors(self):
        hf = hf_tensors()
        g = gguf_tensors(hf)
        g["blk.0.attn_qkv_gate.weight"] = (torch.cat([g.pop("blk.0.attn_qkv.weight")[0].float(),
                                                      g.pop("blk.0.attn_gate.weight")[0].float()]), T.F32)
        g["blk.0.ssm_beta_alpha.weight"] = (torch.cat([g.pop("blk.0.ssm_beta.weight")[0].float(),
                                                       g.pop("blk.0.ssm_alpha.weight")[0].float()]), T.F32)
        g["blk.0.hc_attn_fn.weight"] = (torch.randn(3, 8), T.F32)         # mHC routing: no HF counterpart
        # the fork numbers blocks by execution index (SenseNova: 0-27, then 52-55 for HF layers 28-31, nextn = 56):
        # non-MTP blocks are compacted to consecutive HF layers, so blk.5 here is model.layers.1
        for k in [k for k in g if k.startswith("blk.1.")]:
            g["blk.5." + k[len("blk.1."):]] = g.pop(k)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fused.gguf"
            write_fixture(p, g, arch="sensenova_loop")
            h = gr.open_gguf_as_hf(p)
            try:
                self.assertEqual(h.architecture, "sensenova_loop")
                self.assertTrue(h.norm_offset)        # checked on their BF16 GGUF: attn_norm = Qwen3.5's + 1
                self.assertEqual(h.keys(), sorted(hf))
                self.assertIn("blk.0.hc_attn_fn.weight", h.hidden())
                self.assertEqual(h.geometry(0), {"n_v_heads": NV, "n_k_heads": NK, "head_dim": HD})
                self.assertEqual(h.original("model.layers.0.linear_attn.in_proj_z.weight"), "blk.0.attn_qkv_gate.weight")
                self.assertEqual(h.original("model.layers.0.linear_attn.in_proj_a.weight"), "blk.0.ssm_beta_alpha.weight")
                for name, x in hf.items():
                    self.assertEqual(h.shape(name), tuple(x.shape), name)
                    y = h.get(name)
                    if name.endswith("A_log"):
                        self.assertTrue(torch.allclose(y, x, atol=1e-6), name)
                    else:
                        self.assertTrue(torch.equal(y.float(), x.float()), name)
                    self.assertTrue(torch.equal(h.get_rows(name, 1, 2).float(), y[1:3].float()), name)
            finally:
                h.close(); del h; gc.collect()


class NameMapCase(unittest.TestCase):
    def test_known_pairs(self):
        pairs = {
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
            "blk.3.attn_q.weight": "model.layers.3.self_attn.q_proj.weight",
            "blk.3.attn_k.weight": "model.layers.3.self_attn.k_proj.weight",
            "blk.3.attn_v.weight": "model.layers.3.self_attn.v_proj.weight",
            "blk.3.attn_output.weight": "model.layers.3.self_attn.o_proj.weight",
            "blk.3.attn_q_norm.weight": "model.layers.3.self_attn.q_norm.weight",
            "blk.3.attn_k_norm.weight": "model.layers.3.self_attn.k_norm.weight",
            "blk.12.ffn_gate.weight": "model.layers.12.mlp.gate_proj.weight",
            "blk.12.ffn_up.weight": "model.layers.12.mlp.up_proj.weight",
            "blk.12.ffn_down.weight": "model.layers.12.mlp.down_proj.weight",
            "blk.12.attn_norm.weight": "model.layers.12.input_layernorm.weight",
            "blk.12.post_attention_norm.weight": "model.layers.12.post_attention_layernorm.weight",
            "blk.12.ffn_norm.weight": "model.layers.12.post_attention_layernorm.weight",
            # Qwen3.5 linear attention (Gated DeltaNet), names as llama.cpp's converter writes them
            "blk.0.attn_qkv.weight": "model.layers.0.linear_attn.in_proj_qkv.weight",
            "blk.0.attn_gate.weight": "model.layers.0.linear_attn.in_proj_z.weight",
            "blk.0.ssm_alpha.weight": "model.layers.0.linear_attn.in_proj_a.weight",
            "blk.0.ssm_beta.weight": "model.layers.0.linear_attn.in_proj_b.weight",
            "blk.0.ssm_out.weight": "model.layers.0.linear_attn.out_proj.weight",
            "blk.0.ssm_conv1d.weight": "model.layers.0.linear_attn.conv1d.weight",
            "blk.0.ssm_dt.bias": "model.layers.0.linear_attn.dt_bias",
            "blk.0.ssm_a": "model.layers.0.linear_attn.A_log",
            "blk.0.ssm_norm.weight": "model.layers.0.linear_attn.norm.weight",
        }
        for g, h in pairs.items():
            self.assertEqual(gr.gguf_to_hf_name(g), h, g)
        self.assertIsNone(gr.gguf_to_hf_name("blk.0.mhc_route.weight"))
        self.assertIsNone(gr.gguf_to_hf_name("rope_freqs.weight"))

    def test_hidden_groups_know_gguf_stored_names(self):
        """model_report groups hidden tensors by wd.hidden_group: the GGUF's MTP block and SenseNova's mHC routing
        must not fall into 'other' (the SenseNova report said 'drops the base's MTP head' because of that)."""
        from engine import weight_diff as wd
        self.assertEqual(wd.hidden_group("blk.56.nextn.eh_proj.weight"), "mtp")
        self.assertEqual(wd.hidden_group("blk.4.hc_attn_fn.weight"), "mhc")
        self.assertEqual(wd.hidden_group("blk.27.hc_ffn_fn.bias"), "mhc")

    def test_mtp_block_maps_to_mtp_names(self):
        m = {32}
        self.assertEqual(gr.gguf_to_hf_name("blk.32.nextn.eh_proj.weight", mtp_blocks=m), "mtp.fc.weight")
        self.assertEqual(gr.gguf_to_hf_name("blk.32.nextn.enorm.weight", mtp_blocks=m), "mtp.pre_fc_norm_embedding.weight")
        self.assertEqual(gr.gguf_to_hf_name("blk.32.nextn.hnorm.weight", mtp_blocks=m), "mtp.pre_fc_norm_hidden.weight")
        self.assertEqual(gr.gguf_to_hf_name("blk.32.nextn.shared_head_norm.weight", mtp_blocks=m), "mtp.norm.weight")
        self.assertEqual(gr.gguf_to_hf_name("blk.32.attn_q.weight", mtp_blocks=m), "mtp.layers.0.self_attn.q_proj.weight")
        self.assertEqual(gr.gguf_to_hf_name("blk.32.ffn_down.weight", mtp_blocks=m), "mtp.layers.0.mlp.down_proj.weight")
        self.assertEqual(gr.gguf_to_hf_name("blk.32.attn_norm.weight", mtp_blocks=m), "mtp.layers.0.input_layernorm.weight")
        # without the hint a nextn tensor is still recognised from its own name
        self.assertEqual(gr.gguf_to_hf_name("blk.32.nextn.eh_proj.weight"), "mtp.fc.weight")
        self.assertEqual(gr.mtp_blocks(["blk.32.nextn.eh_proj.weight", "blk.0.attn_q.weight"]), {32})

    def test_norm_offset_depends_on_the_architecture(self):
        for arch in ("qwen35", "qwen35moe", "qwen3next", "gemma2", "gemma3"):
            self.assertTrue(gr.norms_stored_plus_one(arch), arch)
        for arch in ("llama", "qwen2", "qwen3", "phi3"):
            self.assertFalse(gr.norms_stored_plus_one(arch), arch)

    def test_llama_arch_view_leaves_norms_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "llama.gguf"
            w = gguf.GGUFWriter(str(p), arch="llama")
            w.add_block_count(1)
            w.add_tensor("blk.0.attn_norm.weight", np.array([1.5, 0.5], dtype=np.float32))
            w.add_tensor("output_norm.weight", np.array([2.0, 1.0], dtype=np.float32))
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            h = gr.open_gguf_as_hf(p)
            try:
                self.assertFalse(h.norm_offset)
                self.assertTrue(torch.equal(h.get("model.layers.0.input_layernorm.weight"), torch.tensor([1.5, 0.5])))
                self.assertTrue(torch.equal(h.get("model.norm.weight"), torch.tensor([2.0, 1.0])))
            finally:
                h.close(); del h; gc.collect()
            # an explicit override for converters whose architecture string is new (SenseNova's "sensenova_loop")
            h = gr.open_gguf_as_hf(p, norm_offset=True)
            try:
                self.assertTrue(h.norm_offset)
                self.assertTrue(torch.equal(h.get("model.layers.0.input_layernorm.weight"), torch.tensor([0.5, -0.5])))
                self.assertTrue(torch.equal(h.get_rows("model.norm.weight", 1, 1), torch.tensor([0.0])))
            finally:
                h.close(); del h; gc.collect()

    def test_head_permutation_formula(self):
        self.assertEqual(gr.value_head_order(NV, NK), PERM)
        self.assertEqual(gr.value_head_order(32, 16)[:4], [0, 2, 4, 6])
        self.assertEqual(gr.value_head_order(32, 16)[16:18], [1, 3])
        self.assertEqual(gr.value_head_order(8, 8), list(range(8)))   # one value head per key head: identity


@unittest.skipUnless(A0.exists() and HF9B.exists(), "real Qwen3.5-9B files not present")
class RealA0Case(unittest.TestCase):
    """Second method on real files: every A0 tensor maps to an HF tensor of the same shape; values agree."""

    @classmethod
    def setUpClass(cls):
        cls.g = gr.open_gguf_as_hf(A0)
        cls.h = wd.Renamed(wd.LocalSafetensors(HF9B), wd.text_model_name)

    @classmethod
    def tearDownClass(cls):
        cls.g.close()
        del cls.g
        gc.collect()

    def test_every_a0_tensor_maps_to_an_hf_tensor_of_the_same_shape(self):
        hk = set(self.h.keys())
        missing = [k for k in self.g.keys() if k not in hk]
        self.assertEqual(missing, [], f"{len(missing)} GGUF tensors without an HF partner")
        bad = [(k, self.g.shape(k), self.h.shape(k)) for k in self.g.keys() if self.g.shape(k) != self.h.shape(k)]
        self.assertEqual(bad, [])
        # the MTP block (blk.32 in A0) is hidden on both sides
        self.assertTrue(all(n.startswith("blk.32.") for n in self.g.hidden()))
        self.assertEqual(len(self.g.hidden()), 15)
        self.assertEqual(len(self.g.keys()), 442 - 15)
        self.assertEqual(self.g.geometry(0), {"n_v_heads": 32, "n_k_heads": 16, "head_dim": 128})
        self.assertEqual(self.g.architecture, "qwen35")
        self.assertTrue(self.g.norm_offset)

    def test_f32_tensors_equal_hf_values(self):
        names = ["model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
                 "model.layers.0.linear_attn.dt_bias", "model.layers.0.linear_attn.norm.weight",
                 "model.layers.0.linear_attn.A_log", "model.layers.3.self_attn.q_norm.weight", "model.norm.weight",
                 "model.layers.0.linear_attn.conv1d.weight", "model.layers.30.linear_attn.dt_bias",
                 "model.layers.29.linear_attn.A_log", "model.layers.31.self_attn.k_norm.weight"]
        worst = 0.0
        for n in names:
            a, b = self.g.get(n).float(), self.h.get(n).float()
            self.assertEqual(tuple(a.shape), tuple(b.shape), n)
            # HF keeps BF16 for most of these while the GGUF keeps F32: equal after rounding both to bf16
            err = float((a.to(torch.bfloat16).float() - b.to(torch.bfloat16).float()).abs().max())
            worst = max(worst, err)
            self.assertLessEqual(err, 1e-6, n)
        print(f"\n[real] F32 tensors vs HF after bf16 rounding: max abs error {worst:.3g} over {len(names)} tensors")

    def test_q6k_matrices_match_hf(self):
        cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))  # noqa: E731
        n = "model.layers.3.self_attn.q_proj.weight"
        a = self.g.get(n)                    # dequantised Q6_K, float32, (8192, 4096)
        b = self.h.get(n).float()
        self.assertEqual(tuple(a.shape), (8192, 4096))
        c = cos(a, b)
        rel = float((a - b).norm() / b.norm())
        print(f"[real] blk.3.attn_q Q6_K vs HF q_proj: cos {c:.6f}, rel error {rel:.4f}")
        self.assertGreater(c, 0.99)
        # get_rows on a quantised tensor returns exactly the rows of the full dequantisation
        self.assertTrue(torch.equal(self.g.get_rows(n, 1000, 7), a[1000:1007]))
        # the linear-attention tensors line up only after undoing llama.cpp's value-head interleave
        for k in ("in_proj_a", "in_proj_b", "in_proj_z", "in_proj_qkv", "out_proj"):
            name = f"model.layers.0.linear_attn.{k}.weight"
            c = cos(self.g.get(name), self.h.get(name))
            print(f"[real] {name}: cos {c:.6f}")
            self.assertGreater(c, 0.99, name)
        self.assertTrue(torch.equal(self.g.get_rows("model.layers.0.linear_attn.out_proj.weight", 10, 5),
                                    self.g.get("model.layers.0.linear_attn.out_proj.weight")[10:15]))


if __name__ == "__main__":
    unittest.main()
