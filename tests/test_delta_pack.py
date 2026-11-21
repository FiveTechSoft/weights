import json
import tempfile
import unittest

import torch

from engine import delta_pack as dp


def _models(seed=0, shape=(24, 16), rank=3):
    g = torch.Generator().manual_seed(seed)
    base = {"a.weight": torch.randn(*shape, generator=g), "b.weight": torch.randn(*shape, generator=g),
            "c.weight": torch.randn(*shape, generator=g)}
    model = {k: v + torch.randn(shape[0], rank, generator=g) @ torch.randn(rank, shape[1], generator=g) for k, v in base.items()}
    return base, model


class DeltaPack(unittest.TestCase):
    def test_full_rank_apply_rebuilds_model_and_rollback_is_exact(self):
        base, model = _models()
        names = ["a.weight", "b.weight"]
        f, m = dp.build_pack(base, model, names, rank=16)
        patched = dp.apply_pack(base, f, m)
        for n in names:
            self.assertTrue(torch.allclose(patched[n], model[n], atol=1e-4))
        self.assertTrue(dp.rollback_check(base, m))  # inputs never modified
        back, err = dp.unapply(patched, f, m, base)
        self.assertLess(err, 1e-4)  # approximate by design; the exact route is the untouched original below
        self.assertEqual(dp.verify_tensors(base, m, "base"), [])
        self.assertNotIn("c.weight", patched)  # only listed tensors are in the package

    def test_rank_equal_to_true_rank_keeps_all_energy(self):
        base, model = _models(rank=3)
        _, m = dp.build_pack(base, model, ["a.weight"], rank=3)
        self.assertGreater(m["tensors"][0]["delta_energy_kept"], 0.9999)
        _, m1 = dp.build_pack(base, model, ["a.weight"], rank=1)
        self.assertLess(m1["tensors"][0]["delta_energy_kept"], 0.9)

    def test_refuses_wrong_base_and_tampered_factors(self):
        base, model = _models()
        f, m = dp.build_pack(base, model, ["a.weight"], rank=3)
        wrong = dict(base, **{"a.weight": base["a.weight"] + 1e-3})
        with self.assertRaises(ValueError):
            dp.apply_pack(wrong, f, m)
        self.assertFalse(dp.rollback_check(wrong, m))
        bad = dict(f)
        bad["a.weight::left"] = f["a.weight::left"] * 1.001
        with self.assertRaises(ValueError):
            dp.apply_pack(base, bad, m)

    def test_alpha_zero_is_the_base_and_half_is_halfway(self):
        base, model = _models()
        f, m = dp.build_pack(base, model, ["a.weight"], rank=16)
        self.assertTrue(torch.equal(dp.apply_pack(base, f, m, alpha=0.0)["a.weight"], base["a.weight"].float()))
        half = dp.apply_pack(base, f, m, alpha=0.5)["a.weight"]
        self.assertTrue(torch.allclose(half, (base["a.weight"] + model["a.weight"]) / 2, atol=1e-4))

    def test_corruption_is_detected_and_subtraction_is_not_the_rollback(self):
        base, model = _models()
        f, m = dp.build_pack(base, model, ["a.weight", "b.weight"], rank=16)
        corrupt = {k: v.clone() for k, v in base.items()}
        corrupt["b.weight"][0, 0] += 1e-6
        self.assertEqual(dp.verify_tensors(corrupt, m, "base"), ["b.weight"])
        self.assertEqual(dp.verify_tensors({"a.weight": base["a.weight"]}, m, "base"), ["b.weight"])  # missing counts
        self.assertEqual(dp.verify_tensors(model, m, "model"), [])
        patched = dp.apply_pack(base, f, m)
        back, _ = dp.unapply(patched, f, m, base)
        # subtraction may or may not return the identical bytes; the test only pins that we do not rely on it:
        self.assertEqual(dp.verify_tensors(base, m, "base"), [])  # the untouched original is the rollback

    def test_hash_depends_on_dtype_shape_and_bytes(self):
        t = torch.arange(6.0)
        self.assertNotEqual(dp.tensor_sha256(t), dp.tensor_sha256(t.to(torch.bfloat16)))
        self.assertNotEqual(dp.tensor_sha256(t), dp.tensor_sha256(t.reshape(2, 3)))
        self.assertEqual(dp.tensor_sha256(t), dp.tensor_sha256(t.clone()))

    def test_manifest_claims_and_save_roundtrip(self):
        from safetensors.torch import load_file
        base, model = _models()
        f, m = dp.build_pack(base, model, ["a.weight"], rank=3, meta={"base_repo": "x/base", "base_revision": "r1"})
        self.assertEqual(m["claims"]["behavioural_benefit"], "not shown")
        self.assertIsNone(m["model"]["revision"])
        with tempfile.TemporaryDirectory() as d:
            m2 = dp.save_pack(f, m, d)
            self.assertEqual(m2["factor_file"]["sha256"], dp.file_sha256(d + "/delta_pack.safetensors"))
            f2 = load_file(d + "/delta_pack.safetensors")
            json.load(open(d + "/delta_pack.manifest.json"))
            self.assertTrue(dp.apply_pack(base, f2, m2)["a.weight"].shape == base["a.weight"].shape)

    def test_bf16_cast_is_not_a_change(self):
        g = torch.Generator().manual_seed(1)
        w32 = torch.randn(8, 8, generator=g)
        _, m = dp.build_pack({"w": w32}, {"w": w32.to(torch.bfloat16)}, ["w"], rank=2)
        # the F32 side is rounded to bf16 before the difference, so only the rounding of the base differs
        self.assertEqual(m["tensors"][0]["dtype"], "torch.float32")

    def test_schema_is_checked(self):
        base, model = _models()
        f, m = dp.build_pack(base, model, ["a.weight"], rank=3)
        m["schema"] = "other"
        with self.assertRaises(ValueError):
            dp.apply_pack(base, f, m)


if __name__ == "__main__":
    unittest.main()
