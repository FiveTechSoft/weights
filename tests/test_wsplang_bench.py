"""WSP-Lang benchmarks (docs/wsplang_0.8_bench_addendum.md, lab/probes/wsplang_bench/PLAN.md): suites shipped in
engine/bench_suites with a verified digest, bench_run over bench_runner and preserve_bench, the paired bench_compare
with the fixed verdict rule, the registry and the rendered table, the ops bench_run / bench_compare / bench_table, and
evaluate_benchmark delegating to the same path. Written before engine/wsplang_bench.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import wsplang_bench as wb  # noqa: E402
import engine.wsplang as wsplang  # noqa: E402


def item(bench, i, ok, finish="stop", tokens=1000, ok2=None):
    return {"bench": bench, "i": i, "ok": ok, "ok_independent": ok if ok2 is None else ok2, "finish": finish,
            "tokens": tokens, "reasoning_chars": 500 if finish == "stop" else 2000, "gold": "1", "pred": "1"}


SHA = "d" * 64  # the same served file in every synthetic run: comparisons are "edit vs its base" (extras differ)


def run_json(items, arm="x", preservation_items=None):
    return {"wsplang_bench_version": "0.8", "suite": "dev-2048", "suite_sha": "abc", "arm": arm,
            "served": {"model_path": "m.gguf", "sha256": SHA}, "decoding": {"max_tokens": 2048},
            "benchmarks": {}, "preservation": {}, "items": items, "preservation_items": preservation_items or [],
            "created": "2026-11-18T00:00:00"}


FAKE = {"choices": [{"message": {"content": "\\boxed{1}", "reasoning_content": "r"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 7}, "timings": {"predicted_per_second": 1.0}}
IDENT = {"model_path": "C:/x/m.gguf", "model_sha": "a" * 64, "extras": {}, "binary": {"path": "s.exe", "sha256": None}}


class SuiteTest(unittest.TestCase):
    def test_digest_is_stable_and_load_verifies(self):
        s = wb.make_suite("tiny", gsm_idx=[0, 1], math_idx=[], max_tokens=64, preservation=False)
        self.assertEqual(s["sha256"], wb.suite_digest(s))
        self.assertEqual(s["wsplang_bench_version"], "0.8")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.json"
            p.write_text(json.dumps(s), encoding="utf-8")
            loaded = wb.load_suite(p)
            self.assertEqual(loaded["items"]["gsm8k"], [0, 1])
            s2 = dict(s); s2["decoding"] = dict(s["decoding"], max_tokens=65)
            p.write_text(json.dumps(s2), encoding="utf-8")
            with self.assertRaises(ValueError):
                wb.load_suite(p)

    def test_shipped_suites_resolve_by_name_and_dev_items_are_bench_runner_defaults(self):
        dev = wb.resolve_suite("dev-2048")
        self.assertEqual(dev["items"]["gsm8k"], list(range(100)))
        self.assertEqual(len(dev["items"]["math500"]), 50)
        self.assertEqual(dev["decoding"]["max_tokens"], 2048)
        self.assertEqual(wb.resolve_suite("dev-8192")["decoding"]["max_tokens"], 8192)
        s3 = wb.resolve_suite("sealed-3")
        self.assertEqual((len(s3["items"]["gsm8k"]), len(s3["items"]["math500"])), (200, 100))
        self.assertTrue(s3["sealed_file"].endswith("sealed_confirm3.json"))
        by_path = wb.resolve_suite(ROOT / "engine" / "bench_suites" / "dev-2048.json")
        self.assertEqual(by_path["sha256"], dev["sha256"])
        with self.assertRaises(FileNotFoundError):
            wb.resolve_suite("no-such-suite")


class BenchRunTest(unittest.TestCase):
    def test_bench_run_with_fake_server_records_extra_body(self):
        s = wb.make_suite("tiny", gsm_idx=[0, 1], math_idx=[], max_tokens=64, preservation=False)
        seen = []

        def fake_post(url, body):
            seen.append(body)
            return FAKE
        with tempfile.TemporaryDirectory() as d, mock.patch.object(wb, "_post", side_effect=fake_post), \
                mock.patch.object(wb.bench_runner, "served_info", return_value=None), \
                mock.patch.object(wb, "server_identity", return_value=dict(IDENT)):
            out = Path(d) / "r.json"
            rep = wb.bench_run(s, 8099, out, "armA", extra_body={"logit_bias": [[5, 2.0]]}, log=lambda m: None,
                               work_dir=Path(d))
            self.assertTrue(out.exists())
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["logit_bias"], [[5, 2.0]])
        self.assertEqual(seen[0]["temperature"], 0)
        self.assertEqual(rep["benchmarks"]["gsm8k"]["n"], 2)
        self.assertEqual(rep["extra_body"], {"logit_bias": [[5, 2.0]]})
        self.assertEqual(rep["suite_sha"], s["sha256"])
        self.assertEqual(rep["wsplang_bench_version"], "0.8")
        self.assertEqual(rep["identity"]["model_sha"], "a" * 64)

    def test_spent_sealed_suite_is_refused_without_force(self):
        s = wb.make_suite("sealed-x", gsm_idx=[0], math_idx=[], max_tokens=64, preservation=False)
        s["sealed_use"] = "single use, spent 2026-11-17"
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(wb.SealedSpent):
                wb.bench_run(s, 8099, Path(d) / "r.json", "a", log=lambda m: None, work_dir=Path(d))

    def test_evaluate_benchmark_and_bench_run_ops_give_the_same_shape(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(wb, "_post", return_value=FAKE), \
                mock.patch.object(wb.bench_runner, "served_info", return_value=None), \
                mock.patch.object(wb, "server_identity", return_value=dict(IDENT)):
            s = wb.make_suite("tiny", gsm_idx=[0, 1], math_idx=[], max_tokens=64, preservation=False)
            sp = Path(d) / "tiny.json"; sp.write_text(json.dumps(s), encoding="utf-8")
            a = wb.bench_run(s, 8099, Path(d) / "a.json", "a", log=lambda m: None, work_dir=Path(d))
            # evaluate_benchmark's arguments (gsm8k=, math500=, max_tokens=) build an ad-hoc suite on the same path
            b = wb.bench_run_legacy(8099, "b", gsm_n=2, math_n=0, max_tokens=64, out=Path(d) / "b.json",
                                    log=lambda m: None, work_dir=Path(d))
        self.assertEqual(set(a) - {"created"}, set(b) - {"created"})
        self.assertEqual(a["benchmarks"]["gsm8k"]["n"], b["benchmarks"]["gsm8k"]["n"])
        self.assertEqual(b["suite"], "adhoc")


class CompareTest(unittest.TestCase):
    def setUp(self):
        base = [item("gsm8k", i, i < 50) for i in range(100)] + [item("math500", i, i < 10) for i in range(50)]
        arm = [item("gsm8k", i, i < 50 or 50 <= i < 62) for i in range(100)] + [item("math500", i, i < 10) for i in range(50)]
        self.base, self.arm = run_json(base, "base"), run_json(arm, "arm")

    def test_counts_and_p(self):
        c = wb.bench_compare(self.base, self.arm)
        g = c["benchmarks"]["gsm8k"]["g1"]
        self.assertEqual((g["base"], g["arm"], g["win"], g["loss"]), (50, 62, 12, 0))
        self.assertLess(g["p"], 0.001)
        self.assertEqual(c["benchmarks"]["math500"]["g1"]["win"], 0)
        self.assertEqual(c["benchmarks"]["gsm8k"]["extra"]["lost_base_ok_stop"], 0)

    def test_verdict_improves_then_blocked_by_placebo_band_and_preservation(self):
        self.assertEqual(wb.bench_compare(self.base, self.arm)["verdict"], "improves")
        placebo = run_json([item("gsm8k", i, i < 50 or 50 <= i < 60) for i in range(100)] +
                           [item("math500", i, i < 10) for i in range(50)], "placebo")
        c = wb.bench_compare(self.base, self.arm, placebo=placebo)
        self.assertEqual(c["verdict"], "same / not decidable")
        self.assertIn("placebo", " ".join(c["reasons"]))
        band = {"gsm8k": {"abs_delta": 7, "discordant": 11}, "math500": {"abs_delta": 1, "discordant": 1}}
        c = wb.bench_compare(self.base, self.arm, band=band)
        self.assertEqual(c["verdict"], "same / not decidable")
        self.assertIn("band", " ".join(c["reasons"]))
        band["gsm8k"]["abs_delta"] = 5
        self.assertEqual(wb.bench_compare(self.base, self.arm, band=band)["verdict"], "improves")
        pb = [{"bench": "humaneval", "id": f"t{i}", "ok": True, "finish": "stop", "tokens": 100} for i in range(20)]
        pa = [{"bench": "humaneval", "id": f"t{i}", "ok": i >= 8, "finish": "stop", "tokens": 100} for i in range(20)]
        c = wb.bench_compare(run_json(self.base["items"], "base", pb), run_json(self.arm["items"], "arm", pa))
        self.assertEqual(c["verdict"], "same / not decidable")
        self.assertIn("preserv", " ".join(c["reasons"]))

    def test_verdict_harms(self):
        self.assertEqual(wb.bench_compare(self.arm, self.base)["verdict"], "harms")

    def test_noise_band(self):
        a = run_json([item("gsm8k", i, i < 50) for i in range(100)])
        b = run_json([item("gsm8k", i, 5 <= i < 58) for i in range(100)])
        nb = wb.noise_band(a, b)
        self.assertEqual(nb["gsm8k"]["discordant"], 13)
        self.assertEqual(nb["gsm8k"]["abs_delta"], 3)


class RegistryTest(unittest.TestCase):
    def test_registry_add_and_render(self):
        base = run_json([item("gsm8k", i, i < 50) for i in range(100)], "base")
        base["benchmarks"] = {"gsm8k": {"n": 100, "acc": 0.5, "acc_independent": 0.5, "cut_at_max": 3, "mean_tokens": 900}}
        with tempfile.TemporaryDirectory() as d:
            rp = Path(d) / "base.json"; rp.write_text(json.dumps(base), encoding="utf-8")
            idx = Path(d) / "index.json"
            wb.registry_add(idx, rp, model="Qwen3.5-9B Q6_K")
            wb.registry_add(idx, rp, model="Qwen3.5-9B Q6_K")          # idempotent by path
            self.assertEqual(len(json.loads(idx.read_text(encoding="utf-8"))["runs"]), 1)
            cp = Path(d) / "cmp.json"
            cp.write_text(json.dumps({"base": "base", "arm": "arm", "placebo": None, "verdict": "improves",
                                      "benchmarks": {"gsm8k": {"g1": {"p": 0.001, "base": 50, "arm": 62}}}, "band": None,
                                      "reasons": []}), encoding="utf-8")
            wb.registry_add_comparison(idx, cp)
            md = Path(d) / "t.md"
            wb.render_table(idx, md)
            text = md.read_text(encoding="utf-8")
            self.assertIn(f"| Qwen3.5-9B Q6_K ({SHA[:12]}) | llm-chat-think | external | dev-2048 | base | 50 / 50 |", text)
            self.assertIn("improves", text)


class IdentityProvenanceClassTest(unittest.TestCase):
    def setUp(self):
        self.suite = wb.make_suite("tiny", gsm_idx=[0, 1], math_idx=[], max_tokens=64, preservation=False)

    def test_refuses_without_model_path_unless_unknown_model(self):
        noid = dict(IDENT, model_path=None, model_sha=None)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(wb, "_post", return_value=FAKE), \
                mock.patch.object(wb.bench_runner, "served_info", return_value=None), \
                mock.patch.object(wb, "server_identity", return_value=dict(noid)):
            with self.assertRaises(wb.IdentityError):
                wb.bench_run(self.suite, 8099, Path(d) / "r.json", "a", log=lambda m: None, work_dir=Path(d))
            rep = wb.bench_run(self.suite, 8099, Path(d) / "r2.json", "a", log=lambda m: None, work_dir=Path(d),
                               unknown_model=True)
            self.assertTrue(rep["identity"]["unknown_model"])

    def test_identity_recorded_with_declared_extras(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(wb, "_post", return_value=FAKE), \
                mock.patch.object(wb.bench_runner, "served_info", return_value=None), \
                mock.patch.object(wb, "_get", return_value={"model_path": "C:/x/m.gguf"}):
            rep = wb.bench_run(self.suite, 8099, Path(d) / "r.json", "a", log=lambda m: None, work_dir=Path(d),
                               control_vector="C:/x/cc.gguf", scale=0.35, binary="C:/x/no.exe")
        ident = rep["identity"]
        self.assertEqual(ident["model_path"], "C:/x/m.gguf")
        self.assertEqual(ident["extras"]["control_vector"]["declared_by"], "caller")
        self.assertEqual(ident["extras"]["scale"], 0.35)
        self.assertEqual(ident["suite_sha"], self.suite["sha256"])

    def test_provenance_mismatch_refused_and_match_stored(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(wb, "_post", return_value=FAKE), \
                mock.patch.object(wb.bench_runner, "served_info", return_value=None), \
                mock.patch.object(wb, "server_identity", return_value=dict(IDENT)):
            bad = Path(d) / "bad.json"
            bad.write_text(json.dumps({"wsplang_patch_version": "1.0", "id": "p1", "base_checkpoint_hash": "b" * 64,
                                       "output_sha256": "c" * 64}), encoding="utf-8")
            with self.assertRaises(wb.ProvenanceError):
                wb.bench_run(self.suite, 8099, Path(d) / "r.json", "a", log=lambda m: None, work_dir=Path(d), provenance=bad)
            good = Path(d) / "good.json"
            good.write_text(json.dumps({"wsplang_patch_version": "1.0", "id": "p1", "base_checkpoint_hash": "b" * 64,
                                        "output_sha256": "a" * 64}), encoding="utf-8")
            rep = wb.bench_run(self.suite, 8099, Path(d) / "r2.json", "a", log=lambda m: None, work_dir=Path(d), provenance=good)
        self.assertEqual(rep["provenance"]["kind"], "weightpatch")
        self.assertEqual(rep["provenance"]["base_sha"], "b" * 64)
        self.assertEqual(rep["provenance"]["patches"], ["p1"])

    def test_render_table_prints_dash_not_none_for_logits_rows(self):
        # llm-logits rows have no GSM8K / MATH-500 / cuts / token cells: the table must show "-" there, never "None".
        with tempfile.TemporaryDirectory() as d:
            idx = Path(d) / "index.json"
            idx.write_text(json.dumps({"wsplang_bench_version": wb.VERSION, "comparisons": [], "runs": [
                {"path": "x.json", "model": "M", "model_sha": "ab" * 6, "model_class": "llm-logits", "provenance": "external",
                 "suite": "dist-shift", "suite_sha": "cd" * 6, "arm": "base", "gsm8k_g1": None, "gsm8k_g2": None,
                 "math500_g1": None, "math500_g2": None, "other": {}, "cuts": None, "mean_tokens": None, "humaneval": None,
                 "compliance": None, "garbage": None, "extras": None, "extra_body": None, "date": "2026-11-18",
                 "logits": {"ppl": 8.0258, "kl_mean": 0.0, "same_top_p": 100.0, "hellaswag": 0.78, "winogrande": 0.715,
                            "mmlu": 0.41}}]}), encoding="utf-8")
            out = Path(d) / "t.md"
            wb.render_table(idx, out)
            row = [ln for ln in out.read_text(encoding="utf-8").splitlines() if "| base |" in ln][0]
        self.assertNotIn("None", row)
        self.assertIn("8.026", row)

    def test_read_provenance_compile_report_with_file_list(self):
        # The compile_control_vector report (hf_cache/cc_op_v2/report.json) lists its outputs as a plain list of paths
        # under "files"; it does not record the merged GGUF's sha. The reader must still return the base sha from
        # "inputs" and leave output_sha None (2026-11-18: the dist-shift chain died here with "'list' object has no
        # attribute 'get'").
        with tempfile.TemporaryDirectory() as d:
            rep = Path(d) / "report.json"
            rep.write_text(json.dumps({"inputs": {"base_gguf_sha256": "2f" * 32, "vector": "C:/x/cc_v2.gguf"},
                                       "vector": {"layers": {}}, "G0": 0.04,
                                       "files": ["C:/x/vector/fit.pt", "C:/x/merged-Q6_K.gguf"]}), encoding="utf-8")
            prov = wb.read_provenance(rep)
        self.assertEqual(prov["kind"], "compile_control_vector")
        self.assertEqual(prov["base_sha"], "2f" * 32)
        self.assertIsNone(prov["output_sha"])
        self.assertEqual(prov["patches"], ["compile:cc_v2.gguf"])

    def test_wsp_manifest_fingerprint_prefix_matches(self):
        self.assertTrue(wb.sha_match("2bbab016154145943ef887b6bd3c5227", "2bbab016154145943ef887b6bd3c5227" + "0" * 32))
        self.assertFalse(wb.sha_match("2bbab016", "2bbab016" + "0" * 56))

    def test_class_mismatch_refused(self):
        nano = wb.make_nanowsp_suite(tasks=("fsm4",))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(wb.ClassMismatch):
                wb.bench_run(nano, 8099, Path(d) / "r.json", "a", log=lambda m: None, work_dir=Path(d))
            with self.assertRaises(wb.ClassMismatch):
                wb.bench_run(self.suite, None, Path(d) / "r.json", "a", model="x.safetensors", log=lambda m: None, work_dir=Path(d))

    def test_think_false_plain_answer_is_graded(self):
        plain = {"choices": [{"message": {"content": "The answer is \\boxed{1}."}, "finish_reason": "stop"}],
                 "usage": {"completion_tokens": 5}, "timings": {}}
        s = wb.make_suite("tiny-nothink", gsm_idx=[0], math_idx=[], max_tokens=64, preservation=False,
                          model_class="llm-chat", think=False)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(wb, "_post", return_value=plain), \
                mock.patch.object(wb.bench_runner, "served_info", return_value=None), \
                mock.patch.object(wb, "server_identity", return_value=dict(IDENT)):
            rep = wb.bench_run(s, 8099, Path(d) / "r.json", "a", log=lambda m: None, work_dir=Path(d))
        gold = rep["items"][0]["gold"]
        self.assertEqual(rep["items"][0]["ok"], gold == "1")
        self.assertEqual(rep["items"][0]["reasoning_chars"], 0)
        self.assertEqual(rep["think_share"], 0.0)
        self.assertFalse(rep["think"])


class NanoWSPSuiteTest(unittest.TestCase):
    def test_nanowsp_suite_runs_in_process_on_a_tiny_checkpoint(self):
        from engine.nanowsp_runtime import NanoWSP
        suite = wb.make_nanowsp_suite(tasks=("fsm4",))
        task = json.loads((ROOT / "lab" / "data" / "wspx_fsm4.json").read_text(encoding="utf-8"))
        pairs = [(it["prompt"], it["target"]) for it in task["train"][:8]]
        with tempfile.TemporaryDirectory() as d:
            m = NanoWSP(seed=1)
            m.imprint_associative_batch(pairs, layer_idx=1, eta=5.0)
            ck = Path(d) / "model.safetensors"
            m.save_checkpoint(ck)
            rep = wb.bench_run(suite, out=Path(d) / "r.json", arm="nano", model=ck, log=lambda m: None, work_dir=Path(d))
            self.assertEqual(rep["model_class"], "nanowsp")
            self.assertEqual(rep["identity"]["model_sha"], wb.sha_file(ck))
            self.assertIn("fsm4/train", rep["benchmarks"])
            self.assertEqual(rep["benchmarks"]["fsm4/train"]["n"], task["n_train"])
            self.assertEqual(len(rep["items"]), task["n_train"] + task["n_holdout"])
            # the imprinted pairs are recalled by the same rule the runtime uses
            recalled = {r["i"]: r["ok"] for r in rep["items"]}
            self.assertGreaterEqual(sum(recalled[p] for p, _ in pairs), m.evaluate_recall(pairs) * len(pairs) - 1e-9)
            # a checkpoint dir with model.safetensors (the ternary layout) loads the same way
            rep2 = wb.bench_run(suite, out=Path(d) / "r2.json", arm="nano2", model=Path(d), log=lambda m: None, work_dir=Path(d))
            self.assertEqual(rep2["benchmarks"], rep["benchmarks"])


class RelationTest(unittest.TestCase):
    def test_two_models_vs_edit_against_its_base(self):
        base = run_json([item("gsm8k", i, i < 50) for i in range(100)], "base")
        base["identity"] = dict(IDENT, model_sha="b" * 64)
        other = run_json([item("gsm8k", i, i < 62) for i in range(100)], "other")
        other["identity"] = dict(IDENT, model_sha="c" * 64)
        c = wb.bench_compare(base, other)
        self.assertEqual(c["relation"], "two models")
        self.assertIn("two models", c["verdict"])
        self.assertEqual(c["benchmarks"]["gsm8k"]["g1"]["win"], 12)
        edited = dict(other, provenance={"kind": "weightpatch", "base_sha": "b" * 64, "patches": ["p1"]})
        c = wb.bench_compare(base, edited)
        self.assertEqual(c["relation"], "edit vs its base")
        self.assertEqual(c["verdict"], "improves")
        diff_suite = dict(other, suite_sha="zzz")
        with self.assertRaises(ValueError):
            wb.bench_compare(base, diff_suite)


class OpsTest(unittest.TestCase):
    def test_registered_read_only(self):
        for op in ("bench_run", "bench_compare", "bench_table"):
            self.assertIn(op, wsplang.OPS_ALLOWED)
            self.assertIn(op, wsplang.READ_ONLY_OPS)


if __name__ == "__main__":
    unittest.main()
