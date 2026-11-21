"""programs/nanowsp_architecture.wspl runs end to end once its key file exists, and never overwrites the input."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from engine.nanowsp_runtime import NanoWSP  # noqa: E402
from engine.wsplang import execute_plan_v4, parse_args_lines, plan_run, validate  # noqa: E402
from make_nanowsp_key import make_key  # noqa: E402


class TestNanoWSPArchitectureProgram(unittest.TestCase):
    def test_all_five_steps_pass_and_the_input_model_is_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model_dir = d / "m"
            model_dir.mkdir()
            NanoWSP().save_checkpoint(model_dir / "model.safetensors")
            before = hashlib.sha256((model_dir / "model.safetensors").read_bytes()).hexdigest()
            key = d / "key.npz"
            np.savez(key, u=make_key())
            code = (ROOT / "programs" / "nanowsp_architecture.wspl").read_text(encoding="utf-8")
            code = code.replace("data/nanowsp_key.npz", key.as_posix())
            code = code.replace("out=models/nanowsp/model_unlearned.safetensors", f"out={(d / 'unlearned.safetensors').as_posix()}")
            prog = parse_args_lines(code)
            steps, errors = validate(prog)
            self.assertEqual(errors, [])
            res = execute_plan_v4(plan_run(prog, steps), prog, model_path=str(model_dir), dry_run=False)
            self.assertEqual([r["status"] for r in res], ["ok"] * 5, res)
            self.assertTrue((d / "unlearned.safetensors").exists())
            self.assertEqual(hashlib.sha256((model_dir / "model.safetensors").read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
