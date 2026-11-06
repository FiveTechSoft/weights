"""engine.weight_diff: compare two checkpoints tensor by tensor without loading either whole.

Covered: sharded safetensors read lazily (every shard), the same reader over byte ranges (for remote files),
per-tensor statistics (relative change, cosine, spectrum of the delta, embedding rows), whole-model aggregation,
and the WSPLang `diff` op using it on a sharded model.
"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.weight_diff as wd  # noqa: E402


def write_sharded(directory, tensors, split):
    """Save `tensors` into two shards (names in `split` go to shard 2) plus an HF index file."""
    directory.mkdir(parents=True, exist_ok=True)
    s1 = {k: v for k, v in tensors.items() if k not in split}
    s2 = {k: v for k, v in tensors.items() if k in split}
    save_file(s1, str(directory / "model-00001-of-00002.safetensors"))
    save_file(s2, str(directory / "model-00002-of-00002.safetensors"))
    index = {"weight_map": {**{k: "model-00001-of-00002.safetensors" for k in s1},
                            **{k: "model-00002-of-00002.safetensors" for k in s2}}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))


class Fixture(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        g = torch.Generator().manual_seed(0)
        self.base = {
            "model.embed_tokens.weight": torch.randn(20, 8, generator=g),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, generator=g),
            "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16, generator=g),
            "model.layers.1.mlp.down_proj.weight": torch.randn(8, 16, generator=g),
            "model.norm.weight": torch.ones(8),
        }
        u, v = torch.randn(8, 1, generator=g), torch.randn(1, 16, generator=g)
        self.rank1 = 0.5 * u @ v
        self.tuned = {k: t.clone() for k, t in self.base.items()}
        self.tuned["model.layers.1.mlp.down_proj.weight"] += self.rank1   # rank-1 change, in shard 2
        self.tuned["model.embed_tokens.weight"][[3, 17]] += 1.0           # two embedding rows move
        self.base_dir, self.tuned_dir = self.dir / "base", self.dir / "tuned"
        split = {"model.layers.1.mlp.down_proj.weight", "model.norm.weight"}
        write_sharded(self.base_dir, {k: v.to(torch.bfloat16) for k, v in self.base.items()}, split)
        write_sharded(self.tuned_dir, {k: v.to(torch.bfloat16) for k, v in self.tuned.items()}, split)


class TestLocalReader(Fixture):
    def test_reads_every_shard(self):
        r = wd.LocalSafetensors(self.base_dir)
        self.assertEqual(set(r.keys()), set(self.base))
        got = r.get("model.layers.1.mlp.down_proj.weight")
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got, self.base["model.layers.1.mlp.down_proj.weight"].to(torch.bfloat16)))

    def test_single_file(self):
        f = self.dir / "one.safetensors"
        save_file({"a": torch.arange(6.0).reshape(2, 3)}, str(f))
        self.assertTrue(torch.equal(wd.LocalSafetensors(f).get("a"), torch.arange(6.0).reshape(2, 3)))


class TestRangeReader(Fixture):
    """The remote reader only needs fetch(file, start, end_inclusive) -> bytes; here it reads local files."""

    def fetch(self, name, start, end):
        self.calls.append((name, start, end))
        with open(self.base_dir / name, "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)

    def test_matches_local_reader_and_reads_only_ranges(self):
        self.calls = []
        index = json.loads((self.base_dir / "model.safetensors.index.json").read_text())
        files = sorted(set(index["weight_map"].values()))
        remote = wd.RangeSafetensors(files, self.fetch)
        local = wd.LocalSafetensors(self.base_dir)
        self.assertEqual(set(remote.keys()), set(local.keys()))
        for k in local.keys():
            self.assertTrue(torch.equal(remote.get(k), local.get(k)), k)
        size = sum((self.base_dir / f).stat().st_size for f in files)
        biggest = max(e - s + 1 for _, s, e in self.calls)
        self.assertLess(biggest, size, "the reader fetched a whole file at once")

    def test_header_length_prefix(self):
        """safetensors starts with a little-endian u64 header length; the reader must honour it."""
        self.calls = []
        f = sorted(p.name for p in self.base_dir.glob("*.safetensors"))[0]
        n = struct.unpack("<Q", self.fetch(f, 0, 7))[0]
        header = json.loads(self.fetch(f, 8, 8 + n - 1))
        self.assertIn("model.embed_tokens.weight", header)


class TestTensorStats(unittest.TestCase):
    def test_relative_change_and_cosine(self):
        a = torch.ones(4, 4)
        s = wd.tensor_delta_stats("w", a, a * 1.1)
        self.assertAlmostEqual(s["rel_delta"], 0.1, places=5)
        self.assertAlmostEqual(s["cosine"], 1.0, places=5)
        self.assertEqual(s["changed_fraction"], 1.0)

    def test_unchanged_tensor(self):
        a = torch.randn(5, 3)
        s = wd.tensor_delta_stats("w", a, a.clone())
        self.assertEqual(s["rel_delta"], 0.0)
        self.assertEqual(s["changed_fraction"], 0.0)
        self.assertEqual(s["stable_rank"], 0.0)

    def test_rank_one_delta_is_detected(self):
        g = torch.Generator().manual_seed(1)
        a = torch.randn(64, 48, generator=g)
        delta = torch.randn(64, 1, generator=g) @ torch.randn(1, 48, generator=g)
        s = wd.tensor_delta_stats("w", a, a + delta)
        self.assertGreater(s["energy_top1"], 0.99)
        self.assertAlmostEqual(s["stable_rank"], 1.0, places=2)

    def test_diffuse_delta_has_high_stable_rank(self):
        g = torch.Generator().manual_seed(2)
        a = torch.randn(64, 48, generator=g)
        s = wd.tensor_delta_stats("w", a, a + 0.1 * torch.randn(64, 48, generator=g))
        self.assertLess(s["energy_top1"], 0.2)
        self.assertGreater(s["stable_rank"], 10)

    def test_embedding_rows(self):
        a = torch.zeros(10, 4)
        b = a.clone()
        b[7] += 3.0
        b[2] += 1.0
        s = wd.tensor_delta_stats("model.embed_tokens.weight", a, b)
        self.assertEqual([r["row"] for r in s["top_rows"][:2]], [7, 2])
        self.assertEqual(s["rows_changed"], 2)

    def test_one_dimensional(self):
        s = wd.tensor_delta_stats("model.norm.weight", torch.ones(8), torch.full((8,), 1.5))
        self.assertAlmostEqual(s["rel_delta"], 0.5, places=5)
        self.assertNotIn("stable_rank", s)


class TestChunkedAndOverlap(unittest.TestCase):
    def test_chunked_equals_whole(self):
        """Big tensors are processed in row blocks; the numbers must not depend on the block size."""
        g = torch.Generator().manual_seed(3)
        a = torch.randn(300, 40, generator=g)
        b = a + 0.05 * torch.randn(300, 40, generator=g)
        b[5] += 2.0
        whole = wd.tensor_delta_stats("model.embed_tokens.weight", a, b, chunk_rows=10_000)
        parts = wd.tensor_delta_stats("model.embed_tokens.weight", a, b, chunk_rows=7)
        for key in ("rel_delta", "cosine", "changed_fraction", "energy_top1", "stable_rank", "max_abs_delta"):
            self.assertAlmostEqual(whole[key], parts[key], places=4, msg=key)
        self.assertEqual(whole["top_rows"][0]["row"], 5)
        self.assertEqual(parts["top_rows"][0]["row"], 5)

    def test_wide_matrix(self):
        """Fewer rows than columns: the spectrum comes from the row side."""
        g = torch.Generator().manual_seed(4)
        a = torch.randn(16, 200, generator=g)
        delta = torch.randn(16, 1, generator=g) @ torch.randn(1, 200, generator=g)
        s = wd.tensor_delta_stats("w", a, a + delta, chunk_rows=5)
        self.assertGreater(s["energy_top1"], 0.99)

    def test_overlap_of_two_deltas(self):
        g = torch.Generator().manual_seed(5)
        a = torch.randn(256, 64, generator=g)
        u = torch.randn(256, 4, generator=g) @ torch.randn(4, 64, generator=g)
        same = wd.delta_overlap(a, a + u, a + 2 * u, k=4)
        self.assertAlmostEqual(same["cosine"], 1.0, places=4)
        self.assertGreater(same["subspace_overlap"], 0.99)
        other = torch.randn(256, 4, generator=g) @ torch.randn(4, 64, generator=g)
        diff = wd.delta_overlap(a, a + u, a + other, k=4)
        self.assertLess(abs(diff["cosine"]), 0.2)
        self.assertLess(diff["subspace_overlap"], 0.35)  # random 4-dim subspaces of R^64: about 4/64
        opposite = wd.delta_overlap(a, a + u, a - u, k=4)
        self.assertAlmostEqual(opposite["cosine"], -1.0, places=4)
        self.assertGreater(opposite["subspace_overlap"], 0.99)


class TestDiffModels(Fixture):
    def test_finds_the_changes_and_only_them(self):
        rep = wd.diff_models(wd.LocalSafetensors(self.base_dir), wd.LocalSafetensors(self.tuned_dir))
        by = {t["tensor"]: t for t in rep["tensors"]}
        self.assertEqual(len(by), len(self.base))
        changed = {k for k, t in by.items() if t["changed_fraction"] > 0}
        self.assertEqual(changed, {"model.layers.1.mlp.down_proj.weight", "model.embed_tokens.weight"})
        self.assertGreater(by["model.layers.1.mlp.down_proj.weight"]["energy_top1"], 0.98)
        self.assertEqual(sorted(r["row"] for r in by["model.embed_tokens.weight"]["top_rows"][:2]), [3, 17])
        self.assertIn("mlp", rep["by_family"])
        self.assertIn(1, rep["by_layer"])
        self.assertEqual(rep["only_in_base"], [])
        self.assertEqual(rep["only_in_target"], [])

    def test_reports_missing_and_mismatched(self):
        extra = self.dir / "extra.safetensors"
        t = {k: v.clone() for k, v in self.base.items()}
        del t["model.norm.weight"]
        t["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(4, 4)
        t["new.tensor"] = torch.zeros(2)
        save_file(t, str(extra))
        rep = wd.diff_models(wd.LocalSafetensors(self.base_dir), wd.LocalSafetensors(extra))
        self.assertEqual(rep["only_in_base"], ["model.norm.weight"])
        self.assertEqual(rep["only_in_target"], ["new.tensor"])
        self.assertEqual(rep["shape_mismatch"], ["model.layers.0.self_attn.q_proj.weight"])

    def test_pattern_filter(self):
        rep = wd.diff_models(wd.LocalSafetensors(self.base_dir), wd.LocalSafetensors(self.tuned_dir),
                             pattern="*mlp*")
        self.assertEqual({t["tensor"] for t in rep["tensors"]},
                         {"model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight"})


class TestRenamed(Fixture):
    """Multimodal Qwen3.5 checkpoints prefix text tensors with model.language_model.; text-only ones do not."""

    def test_multimodal_and_text_only_names_meet(self):
        mm = self.dir / "mm.safetensors"
        save_file({**{k.replace("model.", "model.language_model.", 1): v.to(torch.bfloat16)
                      for k, v in self.base.items()},
                   "model.visual.blocks.0.attn.qkv.weight": torch.zeros(3, 3),
                   "mtp.fc.weight": torch.zeros(2, 2)}, str(mm))
        base = wd.Renamed(wd.LocalSafetensors(mm), wd.text_model_name)
        self.assertEqual(set(base.keys()), set(self.base))
        self.assertTrue(torch.equal(base.get("model.norm.weight"), self.base["model.norm.weight"].to(torch.bfloat16)))
        rep = wd.diff_models(base, wd.Renamed(wd.LocalSafetensors(self.tuned_dir), wd.text_model_name))
        self.assertEqual(rep["only_in_base"], [])
        self.assertEqual({t["tensor"] for t in rep["tensors"] if t["changed_fraction"] > 0},
                         {"model.layers.1.mlp.down_proj.weight", "model.embed_tokens.weight"})

    def test_text_model_name(self):
        self.assertEqual(wd.text_model_name("model.language_model.layers.3.mlp.up_proj.weight"),
                         "model.layers.3.mlp.up_proj.weight")
        self.assertEqual(wd.text_model_name("model.layers.3.mlp.up_proj.weight"), "model.layers.3.mlp.up_proj.weight")
        self.assertEqual(wd.text_model_name("lm_head.weight"), "lm_head.weight")
        self.assertIsNone(wd.text_model_name("model.visual.merger.fc.weight"))
        self.assertIsNone(wd.text_model_name("mtp.layers.0.mlp.up_proj.weight"))


class TestWSPLangDiffOp(Fixture):
    def test_diff_sees_tensors_in_every_shard(self):
        import engine.wsplang as wsplang
        code = (f"experiment d:\nsteps:\n    step diff tensor=* base={self.base_dir.as_posix()}\n")
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertFalse(errors, errors)
        res = wsplang.execute_plan_v4(wsplang.plan_run(prog, steps), prog, model_path=str(self.tuned_dir))[0]
        self.assertEqual(res["status"], "ok", res)
        tensors = res["result"]["tensors"]
        self.assertEqual(set(tensors), set(self.base))
        layer1 = tensors["model.layers.1.mlp.down_proj.weight"]
        self.assertGreater(layer1["changed_weights"], 0)
        self.assertIn("rel_delta", layer1)
        self.assertEqual(tensors["model.layers.0.mlp.down_proj.weight"]["changed_weights"], 0)


if __name__ == "__main__":
    unittest.main()
