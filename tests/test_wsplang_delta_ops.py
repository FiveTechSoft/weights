"""WSPLang ops for studying improved models: delta_stats, delta_signature, delta_compare, evaluate_benchmark.

None of them writes weights; the files they write never replace an input model.
"""
import http.server
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.wsplang as wsplang  # noqa: E402


def run(steps, model_path=None):
    body = "\n".join(f"    step {s}" for s in steps)
    prog = wsplang.parse_args_lines(f"experiment t:\nsteps:\n{body}\n")
    v, errors = wsplang.validate(prog)
    assert not errors, errors
    return wsplang.execute_plan_v4(wsplang.plan_run(prog, v), prog, model_path=model_path)


class Models(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        g = torch.Generator().manual_seed(0)
        base = {f"model.layers.{i}.mlp.down_proj.weight": torch.randn(16, 48, generator=g) for i in range(3)}
        base["model.norm.weight"] = torch.ones(16)
        shared = {i: torch.linalg.qr(torch.randn(16, 2, generator=g))[0] for i in range(3)}

        def tuned(scale):
            t = {k: v.clone() for k, v in base.items()}
            for i in range(3):
                k = f"model.layers.{i}.mlp.down_proj.weight"
                t[k] += shared[i] @ torch.randn(2, 48, generator=g) * scale + 0.01 * torch.randn(16, 48, generator=g)
            return t

        self.paths = {}
        for name, tensors in (("base", base), ("a", tuned(1.0)), ("b", tuned(0.5))):
            p = self.dir / name
            p.mkdir()
            save_file(tensors, str(p / "model.safetensors"))
            self.paths[name] = p.as_posix()
        rnd = {k: v + (0.1 * torch.randn(v.shape, generator=g) if v.dim() == 2 else 0) for k, v in base.items()}
        p = self.dir / "rnd"
        p.mkdir()
        save_file(rnd, str(p / "model.safetensors"))
        self.paths["rnd"] = p.as_posix()


class TestDeltaStats(Models):
    def test_stats(self):
        out = self.dir / "stats_a.json"
        r = run([f"delta_stats base={self.paths['base']} model={self.paths['a']} out={out.as_posix()}"])[0]
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(r["result"]["overall"]["tensors"], 4)
        self.assertIn("mlp", r["result"]["by_family"])
        self.assertEqual(r["result"]["by_family"]["norm"]["rel_delta"], 0.0)
        self.assertTrue(out.exists())
        self.assertEqual(json.loads(out.read_text())["overall"]["tensors"], 4)

    def test_out_cannot_be_an_input(self):
        target = Path(self.paths["a"]) / "model.safetensors"
        before = target.read_bytes()
        r = run([f"delta_stats base={self.paths['base']} model={self.paths['a']} out={target.as_posix()}"])[0]
        self.assertEqual(r["status"], "error")
        self.assertEqual(target.read_bytes(), before)


class TestSignatureAndCompare(Models):
    def test_shared_directions_found_and_random_not(self):
        sig = {n: (self.dir / f"sig_{n}.pt").as_posix() for n in ("a", "b", "rnd")}
        steps = [f"delta_signature base={self.paths['base']} model={self.paths[n]} out={sig[n]} k=2 sketch=256"
                 for n in ("a", "b", "rnd")]
        steps += [f"delta_compare a={sig['a']} b={sig['b']}", f"delta_compare a={sig['a']} b={sig['rnd']}"]
        res = run(steps)
        for r in res:
            self.assertEqual(r["status"], "ok", r)
        loaded = torch.load(sig["a"])
        self.assertEqual(loaded["model.layers.0.mlp.down_proj.weight"]["basis"].shape, (16, 2))
        same = res[3]["result"]["by_type"]["mlp.down_proj.weight"]
        diff = res[4]["result"]["by_type"]["mlp.down_proj.weight"]
        self.assertTrue(same["H3"], same)
        self.assertFalse(diff["H3"], diff)
        self.assertGreater(same["same"], 0.9)


class TestCLIFeasibilityGate(unittest.TestCase):
    """The CLI rejects ops it does not list as implemented; unit tests that call execute_plan_v4 skip that gate."""

    def test_new_ops_pass_the_gate(self):
        code = ("experiment g:\nsteps:\n"
                "    step delta_compare a=x.pt b=y.pt\n"
                "    step delta_stats base=a model=b\n"
                "    step delta_signature base=a model=b out=s.pt\n"
                "    step evaluate_benchmark endpoint=8095 name=t\n"
                "    step subspace_transfer tensors=*mlp* source=s basis=random k=4 out=o\n")
        prog = wsplang.parse_args_lines(code)
        steps, errors = wsplang.validate(prog)
        self.assertFalse(errors, errors)
        res = wsplang.check_feasibility(prog, steps, model_arg=str(ROOT / "hf_cache" / "qwen35-08b-hf"))
        errs = res[0] if isinstance(res, tuple) else res.get("errors", [])
        self.assertFalse([e for e in errs if "NO implementada" in e], errs)


class FakeChat(http.server.BaseHTTPRequestHandler):
    """Answers every question with \\boxed{18}; reports 7 completion tokens."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"message": {"content": "so \\boxed{18}"}, "finish_reason": "stop"}],
                           "usage": {"completion_tokens": 7}, "timings": {"predicted_per_second": 50.0}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestEvaluateBenchmark(unittest.TestCase):
    def test_scores_against_a_server(self):
        srv = http.server.HTTPServer(("127.0.0.1", 0), FakeChat)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        out = Path(d.name) / "bench.json"
        r = run([f"evaluate_benchmark endpoint={srv.server_port} name=fake gsm8k=2 math500=0 "
                 f"out={out.as_posix()}"])[0]
        self.assertEqual(r["status"], "ok", r)
        s = r["result"]["summary"]["gsm8k"]
        # The first GSM8K test answer is 18 (Janet's ducks); the second is 3.
        self.assertEqual((s["n"], s["acc"], s["acc_independent"]), (2, 0.5, 0.5))
        self.assertEqual(s["mean_tokens"], 7)
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
