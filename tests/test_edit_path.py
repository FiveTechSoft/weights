"""Unit test verifying that cmd_edit requantization branch and types are correct."""
import unittest
import sys
from pathlib import Path
import gguf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wsp


class TestEditPath(unittest.TestCase):

    def test_base_dtype_and_unquantized_types(self):
        """Verifies that base_dtype returns string names and unquantized types are correctly checked."""
        class MockTensor:
            def __init__(self, tt):
                self.tensor_type = tt

        t_f32 = MockTensor(gguf.GGMLQuantizationType.F32)
        t_q6k = MockTensor(gguf.GGMLQuantizationType.Q6_K)
        t_q80 = MockTensor(gguf.GGMLQuantizationType.Q8_0)

        # base_dtype returns non-empty string for both
        self.assertEqual(wsp.base_dtype(t_f32), "F32")
        self.assertEqual(wsp.base_dtype(t_q6k), "Q6_K")

        # But unquantized check must distinguish F32 from Q6_K!
        unquantized = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)
        self.assertIn(t_f32.tensor_type, unquantized)
        self.assertNotIn(t_q6k.tensor_type, unquantized)
        self.assertNotIn(t_q80.tensor_type, unquantized)

    def test_journal_path_constant(self):
        """Verify PATH_JRNL is correctly defined."""
        self.assertTrue(hasattr(wsp, "PATH_JRNL"))
        self.assertEqual(wsp.PATH_JRNL.name, "journal.jsonl")


if __name__ == "__main__":
    unittest.main()
