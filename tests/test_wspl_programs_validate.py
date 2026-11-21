"""Static contract: every committed .wspl program parses and passes validate(). It says nothing about running them
(most need model files that are not in the repo)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.wsplang import parse_args_lines, validate  # noqa: E402

FILES = sorted(list((ROOT / "programs").glob("*.wspl")) + list((ROOT / "lab").rglob("*.wspl")))


class TestCommittedPrograms(unittest.TestCase):
    def test_there_are_programs_to_check(self):
        self.assertGreater(len(FILES), 20)

    def test_every_program_parses_and_validates(self):
        for f in FILES:
            with self.subTest(program=str(f.relative_to(ROOT))):
                prog = parse_args_lines(f.read_text(encoding="utf-8"))
                steps, errors = validate(prog)
                self.assertEqual(errors, [])
                self.assertTrue(steps)

    def test_validation_catches_an_unknown_step(self):
        prog = parse_args_lines("experiment t:\nmodel: nanowsp3\nsteps:\n    step not_a_real_op x=1\n")
        _, errors = validate(prog)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
