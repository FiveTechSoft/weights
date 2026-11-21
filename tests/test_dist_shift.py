"""engine/dist_shift.py and the `llm-logits` class of WSP-Lang benchmarks (lab/probes/dist_shift/PLAN.md): llama.cpp's
llama-perplexity as the label-free distribution-shift and general-capability component (perplexity, KL divergence to
the base's saved logits, HellaSwag / Winogrande / MMLU multiple choice). Written before the code. Fixtures are real
llama-perplexity outputs captured on the CPU with the 0.8B over 2 chunks / 8 tasks (2026-11-18)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import dist_shift as ds  # noqa: E402
from engine import wsplang_bench as wb  # noqa: E402

PPL_FIX = """0.01.668.608 I perplexity: tokenizing the input ..
0.01.672.191 I perplexity: calculating perplexity over 2 chunks, n_ctx=512, batch_size=2048, n_seq=4
0.03.644.761 I perplexity: 1.97 seconds per pass - ETA 0.00 minutes
[1]14.1066,[2]14.8565,
0.03.732.365 I Final estimate: PPL = 14.8565 +/- 1.58353
"""

KL_FIX = """0.01.319.899 I kl_divergence: computing over 2 chunks, n_ctx=512, batch_size=2048, n_seq=4

chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1      14.0349 ±    2.1238      -0.00000 ±    0.00001      -0.00000 ±    0.00000     0.001 ±  0.000 %    100.000 ±  0.000 %
   2      14.8154 ±    1.5762      -0.00000 ±  -nan(ind)      -0.00000 ±    0.00000     0.001 ±  0.000 %    100.000 ±  0.000 %

====== Perplexity statistics ======
Mean PPL(Q)                   :  14.815372 ±   1.576206
Mean PPL(base)                :  14.815402 ±   1.576210
Cor(ln(PPL(Q)), ln(PPL(base))): 100.00%
Mean ln(PPL(Q)/PPL(base))     :  -0.000002 ±  -nan(ind)
Mean PPL(Q)/PPL(base)         :   0.999998 ±  -nan(ind)
Mean PPL(Q)-PPL(base)         :  -0.000030 ±  -nan(ind)

====== KL divergence statistics ======
Mean    KLD:  -0.000000 ±   0.000001
Maximum KLD:   0.000050
99.9%   KLD:   0.000047
99.0%   KLD:   0.000037
95.0%   KLD:   0.000022
90.0%   KLD:   0.000017
Median  KLD:  -0.000000
10.0%   KLD:  -0.000017
 5.0%   KLD:  -0.000024
 1.0%   KLD:  -0.000034
 0.1%   KLD:  -0.000040
Minimum KLD:  -0.000040

====== Token probability statistics ======
Mean    Δp:  0.000 ± 0.000 %
Maximum Δp:  0.003%
99.9%   Δp:  0.003%
99.0%   Δp:  0.002%
95.0%   Δp:  0.001%
90.0%   Δp:  0.000%
75.0%   Δp:  0.000%
Median  Δp:  0.000%
25.0%   Δp: -0.000%
10.0%   Δp: -0.000%
 5.0%   Δp: -0.001%
 1.0%   Δp: -0.002%
 0.1%   Δp: -0.003%
Minimum Δp: -0.004%
RMS Δp    :  0.001 ± 0.000 %
Same top p: 100.000 ± 0.000 %
"""

HS_FIX = """0.01.442.518 I hellaswag_score : loaded 10042 tasks from prompt.
0.01.442.525 I hellaswag_score : selecting 8 randomized tasks.
0.01.448.135 I hellaswag_score : calculating hellaswag score over selected tasks.

task\tacc_norm\t95% confidence interval
1\t0.00000000%\t[0.0000%, 79.3451%]
2\t0.00000000%\t[0.0000%, 65.7620%]
3\t33.33333333%\t[6.1492%, 79.2340%]
4\t50.00000000%\t[15.0039%, 84.9961%]
5\t40.00000000%\t[11.7621%, 76.9276%]
6\t33.33333333%\t[9.6771%, 70.0007%]
7\t42.85714286%\t[15.8220%, 74.9542%]
8\t37.50000000%\t[13.6844%, 69.4258%]
"""

WG_FIX = """0.01.273.447 I winogrande_score : loaded 1267 tasks from prompt.
0.01.273.453 I winogrande_score : selecting 8 random tasks
0.01.276.087 I winogrande_score : calculating winogrande score over selected tasks.
1\t100.0000\t -3.547937   -3.925683  1  1
2\t100.0000\t -2.255579   -2.681927  1  1
3\t66.6667\t -4.044654   -4.144418  1  2
4\t75.0000\t -4.744151   -5.049946  1  1
5\t60.0000\t -3.136006   -3.712420  1  2
6\t66.6667\t -5.359867   -5.381104  1  1
7\t71.4286\t -4.528010   -4.063283  2  2
8\t75.0000\t -2.699079   -2.373311  2  2
"""

MC_FIX = """0.01.301.272 I multiple_choice_score: there are 1548 tasks in prompt
task\tacc_norm
1\t100.00000000
2\t50.00000000
3\t33.33333333
4\t50.00000000
5\t40.00000000
6\t33.33333333
7\t42.85714286
8\t37.50000000

0.06.708.938 I Final result: 37.5000 +/- 18.2981
0.06.708.948 I Random chance: 25.0000 +/- 16.3663
"""


class BuildersTest(unittest.TestCase):
    def test_argv_exact(self):
        g = ("-ngl", "99")
        self.assertEqual(ds.cmd_perplexity("pp", "m.gguf", "t.txt", 100, 512, g),
                         ["pp", "-m", "m.gguf", "-f", "t.txt", "-c", "512", "--chunks", "100", "-ngl", "99"])
        self.assertEqual(ds.cmd_save_logits("pp", "m.gguf", "t.txt", "b.bin", 16, 512, g),
                         ["pp", "-m", "m.gguf", "-f", "t.txt", "-c", "512", "--chunks", "16", "--save-all-logits", "b.bin", "-ngl", "99"])
        self.assertEqual(ds.cmd_kl("pp", "m.gguf", "b.bin", 16, 512, g),
                         ["pp", "-m", "m.gguf", "-c", "512", "--chunks", "16", "--kl-divergence", "--kl-divergence-base", "b.bin", "-ngl", "99"])
        self.assertEqual(ds.cmd_hellaswag("pp", "m.gguf", "hs.txt", 400, 1024, g),
                         ["pp", "-m", "m.gguf", "-f", "hs.txt", "--hellaswag", "--hellaswag-tasks", "400", "-c", "1024", "-ngl", "99"])
        self.assertEqual(ds.cmd_winogrande("pp", "m.gguf", "wg.csv", 400, 1024, g),
                         ["pp", "-m", "m.gguf", "-f", "wg.csv", "--winogrande", "--winogrande-tasks", "400", "-c", "1024", "-ngl", "99"])
        # the MMLU file is binary: text-mode -f truncates it on Windows (seen 2026-11-18), -bf is required
        self.assertEqual(ds.cmd_multiple_choice("pp", "m.gguf", "mmlu.bin", 400, 1024, g),
                         ["pp", "-m", "m.gguf", "-bf", "mmlu.bin", "--multiple-choice", "--multiple-choice-tasks", "400", "-c", "1024", "-ngl", "99"])
        self.assertEqual(ds.vector_args("v.gguf", 1.0), ["--control-vector-scaled", "v.gguf:1.0"])
        self.assertEqual(ds.vector_args(None, None), [])


class ParsersTest(unittest.TestCase):
    def test_perplexity(self):
        r = ds.parse_perplexity(PPL_FIX)
        self.assertAlmostEqual(r["ppl"], 14.8565)
        self.assertAlmostEqual(r["ppl_err"], 1.58353)
        self.assertEqual(r["chunks"], 2)

    def test_kl(self):
        r = ds.parse_kl(KL_FIX)
        self.assertAlmostEqual(r["kl_mean"], 0.0, places=6)
        self.assertAlmostEqual(r["kl_mean_err"], 0.000001)
        self.assertAlmostEqual(r["kl_max"], 0.00005)
        self.assertAlmostEqual(r["kl_p99"], 0.000037)
        self.assertAlmostEqual(r["kl_p99_9"], 0.000047)
        self.assertAlmostEqual(r["kl_median"], 0.0, places=6)
        self.assertAlmostEqual(r["same_top_p"], 100.0)
        self.assertAlmostEqual(r["same_top_p_err"], 0.0)
        self.assertAlmostEqual(r["ppl_q"], 14.815372)
        self.assertAlmostEqual(r["ppl_base"], 14.815402)
        self.assertAlmostEqual(r["ln_ppl_ratio_mean"], -0.000002)
        self.assertAlmostEqual(r["dp_rms"], 0.001)
        self.assertEqual(r["chunks"], 2)

    def test_task_scores(self):
        hs = ds.parse_hellaswag(HS_FIX)
        self.assertEqual((hs["tasks"], hs["pool"]), (8, 10042))
        self.assertAlmostEqual(hs["acc"], 0.375)
        self.assertAlmostEqual(hs["ci"][0], 0.136844)
        self.assertAlmostEqual(hs["ci"][1], 0.694258)
        wg = ds.parse_winogrande(WG_FIX)
        self.assertEqual((wg["tasks"], wg["pool"]), (8, 1267))
        self.assertAlmostEqual(wg["acc"], 0.75)
        mc = ds.parse_multiple_choice(MC_FIX)
        self.assertEqual((mc["tasks"], mc["pool"]), (8, 1548))
        self.assertAlmostEqual(mc["acc"], 0.375)
        self.assertAlmostEqual(mc["acc_err"], 0.182981)
        self.assertAlmostEqual(mc["chance"], 0.25)

    def test_parsers_refuse_output_without_a_result(self):
        with self.assertRaises(ValueError):
            ds.parse_perplexity("llama_model_loader: loaded\n")
        with self.assertRaises(ValueError):
            ds.parse_multiple_choice("multiple_choice_score: failed to read task positions from prompt\n")


def _fake_runner(calls, saves):
    """Stands in for subprocess: returns the fixture matching the flags; creates the logits file on --save-all-logits."""
    def run(argv):
        calls.append(list(argv))
        if "--save-all-logits" in argv:
            Path(argv[argv.index("--save-all-logits") + 1]).write_bytes(b"logits")
            saves.append(1)
            return PPL_FIX
        if "--kl-divergence" in argv:
            return KL_FIX
        if "--hellaswag" in argv:
            return HS_FIX
        if "--winogrande" in argv:
            return WG_FIX
        if "--multiple-choice" in argv:
            return MC_FIX
        return PPL_FIX
    return run


class SuiteAndRunTest(unittest.TestCase):
    def _files(self, d):
        d = Path(d)
        for name in ("base.gguf", "edit.gguf", "pp.exe"):
            (d / name).write_bytes(name.encode() * 10)
        (d / "wiki.txt").write_text("some text " * 100, encoding="utf-8")
        (d / "hs.txt").write_text("hs\n", encoding="utf-8")
        (d / "wg.csv").write_text("wg\n", encoding="utf-8")
        (d / "mmlu.bin").write_bytes(b"\x01\x00\x00\x00")
        return d

    def _suite(self, d):
        return ds.make_logits_suite(text=str(d / "wiki.txt"), hellaswag=str(d / "hs.txt"), winogrande=str(d / "wg.csv"),
                                    mmlu=str(d / "mmlu.bin"), ppl_chunks=100, kl_chunks=16, ctx=512, tasks=400, task_ctx=1024,
                                    thresholds={"kl_mean": None, "same_top_p": None}, root=d)

    def test_suite_class_digest_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            d = self._files(d)
            s = self._suite(d)
            self.assertEqual(s["model_class"], "llm-logits")
            self.assertIn("llm-logits", wb.MODEL_CLASSES)
            p = d / "dist.json"
            p.write_text(json.dumps(s), encoding="utf-8")
            loaded = wb.load_suite(p)
            self.assertEqual(loaded["sha256"], s["sha256"])
            s2 = dict(s)
            s2["components"] = json.loads(json.dumps(s["components"]))
            s2["components"]["perplexity"]["chunks"] = 7
            p.write_text(json.dumps(s2), encoding="utf-8")
            with self.assertRaises(ValueError):
                wb.load_suite(p)

    def test_bench_run_logits_class(self):
        with tempfile.TemporaryDirectory() as d:
            d = self._files(d)
            s = self._suite(d)
            calls, saves = [], []
            logits_dir = d / "logits"
            run = wb.bench_run(s, model=str(d / "edit.gguf"), base=str(d / "base.gguf"), arm="edit", out=str(d / "r.json"),
                               binary=str(d / "pp.exe"), runner=_fake_runner(calls, saves), logits_dir=logits_dir, log=lambda m: None)
            self.assertEqual(run["model_class"], "llm-logits")
            self.assertEqual(run["identity"]["model_sha"], wb.sha_file(d / "edit.gguf"))
            self.assertEqual(run["identity"]["binary"]["sha256"], wb.sha_file(d / "pp.exe"))
            self.assertEqual(run["identity"]["base_sha"], wb.sha_file(d / "base.gguf"))
            self.assertEqual(set(run["benchmarks"]), {"perplexity", "kl", "hellaswag", "winogrande", "mmlu"})
            self.assertAlmostEqual(run["benchmarks"]["perplexity"]["ppl"], 14.8565)
            self.assertAlmostEqual(run["benchmarks"]["kl"]["same_top_p"], 100.0)
            self.assertAlmostEqual(run["benchmarks"]["hellaswag"]["acc"], 0.375)
            self.assertEqual(run["items"], [])
            self.assertEqual(saves, [1])                                   # base logits written once ...
            base_logits = logits_dir / f"{wb.sha_file(d / 'base.gguf')[:16]}.bin"
            self.assertTrue(base_logits.exists())
            kl_call = [c for c in calls if "--kl-divergence" in c][0]
            self.assertEqual(kl_call[kl_call.index("--kl-divergence-base") + 1], str(base_logits))
            self.assertIn("data_sha", run["identity"])
            run2 = wb.bench_run(s, model=str(d / "edit.gguf"), base=str(d / "base.gguf"), arm="edit2", out=str(d / "r2.json"),
                                binary=str(d / "pp.exe"), runner=_fake_runner(calls, saves), logits_dir=logits_dir, log=lambda m: None)
            self.assertEqual(saves, [1])                                   # ... and reused
            self.assertEqual(run2["arm"], "edit2")
            saved = json.loads((d / "r.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["benchmarks"]["mmlu"]["acc"], 0.375)

    def test_bench_run_logits_refuses_server_or_missing_files(self):
        with tempfile.TemporaryDirectory() as d:
            d = self._files(d)
            s = self._suite(d)
            with self.assertRaises(wb.ClassMismatch):
                wb.bench_run(s, port=8095, arm="x", out=str(d / "x.json"))
            with self.assertRaises(FileNotFoundError):
                wb.bench_run(s, model=str(d / "missing.gguf"), arm="x", out=str(d / "x.json"), binary=str(d / "pp.exe"),
                             runner=_fake_runner([], []), logits_dir=d / "l")
            with self.assertRaises(wb.ClassMismatch):
                wb.bench_run(wb.resolve_suite("dev-2048"), model=str(d / "edit.gguf"), arm="x", out=str(d / "x.json"))

    def test_control_vector_goes_into_every_command_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            d = self._files(d)
            (d / "vec.gguf").write_bytes(b"vec")
            s = self._suite(d)
            calls = []
            run = wb.bench_run(s, model=str(d / "base.gguf"), base=str(d / "base.gguf"), arm="vec", out=str(d / "v.json"),
                               binary=str(d / "pp.exe"), runner=_fake_runner(calls, []), logits_dir=d / "l",
                               control_vector=str(d / "vec.gguf"), scale=0.5, log=lambda m: None)
            scored = [c for c in calls if "--save-all-logits" not in c]
            self.assertTrue(all("--control-vector-scaled" in c for c in scored))
            self.assertTrue(all("--control-vector-scaled" not in c for c in calls if "--save-all-logits" in c))
            self.assertEqual(run["identity"]["extras"]["control_vector"]["sha256"], wb.sha_file(d / "vec.gguf"))
            self.assertEqual(run["identity"]["extras"]["scale"], 0.5)


class CompareTest(unittest.TestCase):
    def _run(self, arm, ppl, kl_mean, top, hs, wg, mc, thresholds=None, model_sha="aa" * 32, base_sha="bb" * 32):
        return {"model_class": "llm-logits", "suite": "dist-shift", "suite_sha": "s" * 64, "arm": arm,
                "identity": {"model_sha": model_sha, "base_sha": base_sha},
                "thresholds": thresholds or {"kl_mean": None, "same_top_p": None},
                "benchmarks": {"perplexity": {"ppl": ppl, "ppl_err": 0.1, "chunks": 100},
                               "kl": {"kl_mean": kl_mean, "kl_mean_err": 0.001, "kl_p99": kl_mean * 10, "kl_max": kl_mean * 20,
                                      "same_top_p": top, "same_top_p_err": 0.3},
                               "hellaswag": {"acc": hs, "ci": (hs - 0.05, hs + 0.05), "tasks": 400},
                               "winogrande": {"acc": wg, "tasks": 400},
                               "mmlu": {"acc": mc, "acc_err": 0.02, "tasks": 400}}, "items": []}

    def test_not_calibrated_when_thresholds_null(self):
        base = self._run("base", 8.0, 0.0, 100.0, 0.70, 0.70, 0.60, model_sha="bb" * 32)
        arm = self._run("edit", 8.4, 0.05, 92.0, 0.69, 0.71, 0.58)
        c = wb.bench_compare(base, arm)
        self.assertEqual(c["relation"], "edit vs its base")
        self.assertAlmostEqual(c["logits"]["ppl_ratio"], 1.05)
        self.assertAlmostEqual(c["logits"]["kl_mean"], 0.05)
        self.assertAlmostEqual(c["logits"]["same_top_p"], 92.0)
        self.assertAlmostEqual(c["logits"]["delta"]["hellaswag"], -0.01)
        self.assertAlmostEqual(c["logits"]["delta"]["mmlu"], -0.02)
        self.assertEqual(c["verdict"], "not calibrated yet")

    def test_calibrated_cells(self):
        thr = {"kl_mean": 0.02, "same_top_p": 95.0}
        base = self._run("base", 8.0, 0.0, 100.0, 0.70, 0.70, 0.60, thr, model_sha="bb" * 32)
        keep = self._run("edit", 8.1, 0.01, 97.0, 0.70, 0.70, 0.60, thr)
        shift = self._run("edit2", 8.4, 0.05, 92.0, 0.69, 0.71, 0.58, thr)
        self.assertEqual(wb.bench_compare(base, keep)["verdict"], "preserves distribution")
        c = wb.bench_compare(base, shift)
        self.assertEqual(c["verdict"], "distribution shifted")
        self.assertTrue(any("kl_mean" in r for r in c["reasons"]))

    def test_two_models_gets_statistics_only(self):
        base = self._run("qwen", 8.0, 0.0, 100.0, 0.70, 0.70, 0.60, model_sha="bb" * 32)
        other = self._run("llama", 9.0, 0.0, 100.0, 0.72, 0.68, 0.55, model_sha="cc" * 32, base_sha="cc" * 32)
        c = wb.bench_compare(base, other)
        self.assertEqual(c["relation"], "two models")
        self.assertIn("two models", c["verdict"])
        self.assertAlmostEqual(c["logits"]["ppl_ratio"], 1.125)

    def test_registry_row_carries_logits_columns(self):
        with tempfile.TemporaryDirectory() as d:
            run = self._run("edit", 8.4, 0.05, 92.0, 0.69, 0.71, 0.58)
            run.update({"wsplang_bench_version": "0.8", "created": "2026-11-18T00:00:00", "preservation": {}})
            p = Path(d) / "r.json"
            p.write_text(json.dumps(run), encoding="utf-8")
            idx = wb.registry_add(Path(d) / "index.json", p, model="M")
            row = idx["runs"][0]
            self.assertEqual(row["model_class"], "llm-logits")
            self.assertAlmostEqual(row["logits"]["ppl"], 8.4)
            self.assertAlmostEqual(row["logits"]["kl_mean"], 0.05)
            wb.render_table(Path(d) / "index.json", Path(d) / "t.md")
            md = (Path(d) / "t.md").read_text(encoding="utf-8")
            self.assertIn("ppl", md.splitlines()[[i for i, l in enumerate(md.splitlines()) if l.startswith("| model (sha)")][0]])
            self.assertIn("8.4", md)


if __name__ == "__main__":
    unittest.main()
