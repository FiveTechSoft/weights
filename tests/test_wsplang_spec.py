"""docs/wsplang_specification.md must match the engine: same version, every implemented op listed in the op table with
its kind, and no step documented as a WSP-Lang op that the runtime rejects."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.wsplang as w  # noqa: E402

SPEC = (ROOT / "docs" / "wsplang_specification.md").read_text(encoding="utf-8")


class TestSpec(unittest.TestCase):
    def test_version_matches_engine(self):
        m = re.search(r"\*\*Version:\*\*\s*([0-9.]+)", SPEC)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), w.__version__)

    def test_every_implemented_op_is_in_the_table_with_its_kind(self):
        table = SPEC.split("<!-- ops:begin -->")[1].split("<!-- ops:end -->")[0]
        rows = {r.split("|")[1].strip().strip("`"): r.split("|")[2].strip() for r in table.splitlines()
                if r.startswith("| `")}
        self.assertEqual(set(rows), set(w.IMPLEMENTED_OPS))
        for op, kind in rows.items():
            want = "stub" if op in w.STUB_OPS else "read / report" if op in w.READ_ONLY_OPS else "writes weights"
            self.assertEqual(kind, want, op)

    def test_no_step_documented_that_the_runtime_rejects(self):
        steps = set(re.findall(r"^\s*step ([a-z_\-]+)", SPEC, re.M))
        self.assertFalse(steps - w.IMPLEMENTED_OPS, sorted(steps - w.IMPLEMENTED_OPS))


if __name__ == "__main__":
    unittest.main()
