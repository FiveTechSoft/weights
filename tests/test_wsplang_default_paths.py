"""Default model paths keep their historical values and can be overridden by environment variables."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = "import sys; sys.path.insert(0, %r); import engine.wsplang as w; print(w.DEFAULT_MIMO_GGUF); print(w.DEFAULT_QWEN_BASE_GGUF)" % str(ROOT)


def paths(env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("WSP_MODELO_GGUF", "WSP_BASE_GGUF")}
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout.split("\n")[:2]


class TestDefaultPaths(unittest.TestCase):
    def test_historical_defaults_are_unchanged(self):
        mimo, base = paths({})
        self.assertTrue(mimo.endswith("mimo-v2.6-distill-qwen-9b-Q6_K.gguf") and "Anto" in mimo)
        self.assertTrue(base.endswith("Qwen3.5_9B_Base.Q6_K.gguf") and "qwen35base" in base)

    def test_environment_overrides(self):
        mimo, base = paths({"WSP_MODELO_GGUF": "/models/a.gguf", "WSP_BASE_GGUF": "/models/b.gguf"})
        self.assertEqual((mimo, base), ("/models/a.gguf", "/models/b.gguf"))


if __name__ == "__main__":
    unittest.main()
