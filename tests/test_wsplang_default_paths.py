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
    def test_defaults_are_repo_local_paths(self):
        mimo, base = paths({})
        self.assertTrue(mimo.endswith("MiMo-V2.6-Distill-Qwen-9B-Q6_K.gguf"))
        self.assertTrue(base.endswith("Qwen3.5_9B_Base.Q6_K.gguf"))
        for path in (mimo, base):
            self.assertNotIn("C:", path)
            self.assertNotIn("Anto", path)
            self.assertTrue(Path(path).is_absolute())
            self.assertTrue(str(ROOT) in str(path))

    def test_environment_overrides(self):
        mimo, base = paths({"WSP_MODELO_GGUF": "/models/a.gguf", "WSP_BASE_GGUF": "/models/b.gguf"})
        # the runtime normalizes env paths to host separators: compare OS-agnostically
        self.assertEqual(str(Path(mimo)), str(Path("/models/a.gguf")))
        self.assertEqual(str(Path(base)), str(Path("/models/b.gguf")))


if __name__ == "__main__":
    unittest.main()
