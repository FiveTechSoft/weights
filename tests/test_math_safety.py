"""Regression checks only, not a benchmark or evidence of model improvement."""
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mathrig', ROOT / 'rig/math500.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MathSafety(unittest.TestCase):
    def test_distinct_expressions_not_equal(self):
        for pred, gold in [
            (r'\frac{1}{2+3}', r'\frac{1}{2}+3'),
            (r'\sqrt[3]{8}', r'\sqrt{8}'),
            ('(1,234)', '(1234)'),
            ('[1,234]', '[1234]'),
            (r'\frac{1}{\frac{2}{3}}', '1/2/3'),
        ]:
            with self.subTest(pred=pred, gold=gold):
                self.assertFalse(m.equivalent(pred, gold))

    def test_nested_fraction_preserves_structure(self):
        self.assertEqual(m.norm(r'\frac{1}{\frac{2}{3}}'), '1/(2/3)')

    def test_explicit_square_root(self):
        self.assertTrue(m.equivalent(r'\sqrt[2]{8}', r'\sqrt{8}'))

    def test_final_line(self):
        self.assertEqual(m.extract_pred('Calculation 17.\nRespuesta: p - q'), 'p - q')
        self.assertEqual(m.extract_pred(r'work 3 | Respuesta: $3\sqrt{13}$'), r'3\sqrt{13}')

    def test_legacy_runner_uses_shared_parser_and_real_hash(self):
        spec = importlib.util.spec_from_file_location('legacy', ROOT / 'blog_prompts/eval_math.py')
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        self.assertEqual(legacy.ROOT, ROOT)
        self.assertEqual(legacy.extract('Respuesta: p - q', 'math500'), 'p - q')
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'model-fixture'); f.flush()
            self.assertEqual(legacy.hash_model_file(f.name),
                             hashlib.sha256(b'model-fixture').hexdigest())

    def test_legacy_tail_replay(self):
        # These are previously published tails, not new generations. No gold is
        # sent to a model. Empty/truncated answers remain failures.
        rows = json.loads((ROOT / 'rig_runs/r0-orig-math.json').read_text())
        rows = [r for r in rows if 'i' in r]
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(r['ok'] for r in rows), 4)
        self.assertEqual(sum(m.equivalent(m.extract_pred(r['tail']), r['gold'])
                             for r in rows), 10)


if __name__ == '__main__':
    unittest.main()
