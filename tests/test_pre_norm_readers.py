"""Every reader of transformers' hidden_states must take the last row through contrast_vector.hidden_states_pre_norm
(results 45.2: hidden_states[-1] is recorded after the final norm; the raw row made the last increment the norm's
rescaling). Audit of 2026-11-13: act_map capture and close_probe/cvec_extract read the raw tuple."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READERS = ["lab/probes/act_map/act_map.py", "lab/probes/close_probe/cvec_extract.py", "lab/probes/cc_neurons/decompose.py"]


class PreNormReaders(unittest.TestCase):
    def test_readers_use_the_shared_helper_and_not_the_raw_tuple(self):
        for rel in READERS:
            src = (ROOT / rel).read_text(encoding="utf-8")
            with self.subTest(rel):
                self.assertIn("hidden_states_pre_norm(", src)
                self.assertIsNone(re.search(r"\)\.hidden_states\b", src), "raw .hidden_states read left in " + rel)


if __name__ == "__main__":
    unittest.main()
