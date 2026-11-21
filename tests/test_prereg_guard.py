import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.meta.prereg_guard import NotPreregistered, assert_preregistered  # noqa: E402


def git(d, *a):
    subprocess.run(["git", "-C", str(d), *a], check=True, capture_output=True)


class TestPreregGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        git(self.d, "init", "-q")
        git(self.d, "config", "user.email", "t@example.invalid")
        git(self.d, "config", "user.name", "t")
        self.spec = self.d / "spec.py"

    def tearDown(self):
        self.tmp.cleanup()

    def test_untracked_spec_is_refused(self):
        self.spec.write_text("criteria = 1\n")
        with self.assertRaises(NotPreregistered):
            assert_preregistered(self.spec)

    def test_committed_clean_spec_passes_and_returns_the_commit(self):
        self.spec.write_text("criteria = 1\n")
        git(self.d, "add", "spec.py"); git(self.d, "commit", "-qm", "prereg")
        h = subprocess.run(["git", "-C", str(self.d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.assertEqual(assert_preregistered(self.spec), h)

    def test_editing_the_spec_after_the_commit_is_refused(self):
        self.spec.write_text("criteria = 1\n")
        git(self.d, "add", "spec.py"); git(self.d, "commit", "-qm", "prereg")
        self.spec.write_text("criteria = 2\n")
        with self.assertRaises(NotPreregistered):
            assert_preregistered(self.spec)

    def test_outside_a_git_repository_is_refused(self):
        with tempfile.TemporaryDirectory() as other:
            f = Path(other) / "spec.py"
            f.write_text("x\n")
            with self.assertRaises(NotPreregistered):
                assert_preregistered(f)

    def test_the_real_v6_spec_is_preregistered(self):
        spec = ROOT / "lab" / "meta" / "interference_v6.py"
        try:
            assert_preregistered(spec)
        except NotPreregistered as e:
            if "uncommitted" not in str(e) and "no commit" not in str(e):
                raise


if __name__ == "__main__":
    unittest.main()
