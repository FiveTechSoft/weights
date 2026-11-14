import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

CC_DIR = Path(__file__).resolve().parent.parent / "lab" / "probes" / "cc"
sys.path.insert(0, str(CC_DIR))


def load_cc(**env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        import cc
        return importlib.reload(cc)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def screen(path, name, ok, finished, n=64):
    items = [{"bench": "math500", "i": i, "ok": i < ok, "finish": "stop" if i < finished else "length"}
             for i in range(n)]
    (path / f"{name}.json").write_text(json.dumps({"items": items}))


class Selector(unittest.TestCase):
    def test_defaults_are_the_9b_study(self):
        cc = load_cc()
        self.assertEqual(cc.RUNS.name, "cc")
        self.assertEqual(cc.BASE_HF.name, "qwen35-9b-hf")

    def test_env_switches_model_and_runs(self):
        with tempfile.TemporaryDirectory() as d:
            cc = load_cc(CC_RUNS=d, CC_HF="hf_cache/qwen35-08b-hf")
            self.assertEqual(cc.RUNS, Path(d))
            self.assertEqual(cc.BASE_HF.name, "qwen35-08b-hf")
        load_cc()


class Choose(unittest.TestCase):
    def run_choose(self, base, scales):
        with tempfile.TemporaryDirectory() as d:
            cc = load_cc(CC_RUNS=d)
            p = Path(d)
            screen(p, "screen_base", *base)
            for s, (ok, fin) in scales.items():
                screen(p, f"screen_s{s}", ok, fin)
            res = cc.choose()
        load_cc()
        return res

    def test_most_correct_wins(self):
        res = self.run_choose((8, 10), {0.25: (10, 12), 0.5: (17, 20), 1.0: (35, 40), 2.0: (2, 3)})
        self.assertEqual(res["scale"], 1.0)
        self.assertEqual(res["measure"], "correct")

    def test_floor_uses_finished(self):
        res = self.run_choose((1, 3), {0.25: (2, 5), 0.5: (0, 20), 1.0: (2, 20), 2.0: (1, 9)})
        self.assertEqual(res["measure"], "finished")
        self.assertEqual(res["scale"], 0.5)                    # tie 20/20 -> smaller scale

    def test_floor_nothing_beats_base(self):
        res = self.run_choose((0, 30), {0.25: (0, 10), 0.5: (1, 20), 1.0: (2, 30), 2.0: (0, 1)})
        self.assertIsNone(res["scale"])


if __name__ == "__main__":
    unittest.main()
