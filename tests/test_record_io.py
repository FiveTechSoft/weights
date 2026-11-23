"""Tests for the ExperimentRecord IO layer and the id generator."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine.record as record


def _valid_data(**overrides):
    base = {
        "experiment_id": "e00123456789abcd",
        "model_id": "nanowsp3",
        "objective": "medir delta de energia",
        "baseline": {"capacidad_edicion": 0.9},
        "edicion": {"program": "test.wspl", "operaciones": ["poke"]},
        "ejecucion": {"estado": "done"},
        "resultados": {"capacidad_edicion": 0.9, "capacidad_mejro": 0.7},
    }
    base.update(overrides)
    return base


class TestIdGeneration(unittest.TestCase):

    def test_next_id_passes_schema_validation(self):
        for _ in range(50):
            self.assertEqual(record.validate(_valid_data(experiment_id=record.ExperimentRecord.next_id())), [])

    def test_next_id_is_unique(self):
        ids = {record.ExperimentRecord.next_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)


class TestNormalizar(unittest.TestCase):

    def test_base_and_resultados_numbers_are_normalized(self):
        r = record.ExperimentRecord(
            "e00123456789abcd", "m", "obj",
            {"capacidad_edicion": 1, "texto": "x"},
            {"program": "p"},
            {"estado": "done"},
            {"capacidad_mejro": 2, "texto": "y"},
        )
        self.assertEqual(r.baseline["capacidad_edicion"], 1.0)
        self.assertIsNone(r.baseline["texto"])
        self.assertEqual(r.resultados["capacidad_mejro"], 2.0)
        self.assertIsNone(r.resultados["texto"])


class TestSaveLoad(unittest.TestCase):

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rec.json"
            rec = record.ExperimentRecord(
                experiment_id="e00123456789abcd", model_id="m", objective="obj",
                baseline={"capacidad_edicion": 0.5},
                edicion={"program": "p", "operaciones": ["poke"]},
                ejecucion={"estado": "done"},
                resultados={"capacidad_mejro": 0.5},
                conclusion="SUPPORTED",
            )
            rec.save(str(path))
            loaded = record.ExperimentRecord.load(str(path))
            self.assertEqual(loaded.experiment_id, "e00123456789abcd")
            self.assertEqual(loaded.baseline["capacidad_edicion"], 0.5)
            self.assertEqual(loaded.conclusion, "SUPPORTED")

    def test_save_raises_on_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rec.json"
            rec = record.ExperimentRecord(
                experiment_id="INVALID-ID", model_id="m", objective="obj",
                baseline={}, edicion={}, ejecucion={}, resultados={},
            )
            with self.assertRaises(ValueError):
                rec.save(str(path))

    def test_load_raises_on_invalid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"experiment_id": "bad-id"}))
            with self.assertRaises(ValueError):
                record.ExperimentRecord.load(str(path))

    def test_save_writes_veredicto_only_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rec.json"
            rec = record.ExperimentRecord(
                experiment_id="e00123456789abcd", model_id="m", objective="obj",
                baseline={}, edicion={}, ejecucion={}, resultados={},
                veredicto="keep",
            )
            rec.save(str(path))
            self.assertIn("veredicto", json.loads(path.read_text()))

            path2 = Path(tmp) / "rec2.json"
            rec2 = record.ExperimentRecord(
                experiment_id="e00123456789abcd", model_id="m", objective="obj",
                baseline={}, edicion={}, ejecucion={}, resultados={},
            )
            rec2.save(str(path2))
            self.assertNotIn("veredicto", json.loads(path2.read_text()))


class TestJsonlHistory(unittest.TestCase):

    def test_append_and_from_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            rec = record.ExperimentRecord(
                experiment_id="e00123456789abcd", model_id="m", objective="obj",
                baseline={}, edicion={}, ejecucion={}, resultados={},
            )
            rec.append(str(path))
            rec2 = record.ExperimentRecord(
                experiment_id="e00123456789abcde", model_id="m", objective="obj2",
                baseline={}, edicion={}, ejecucion={}, resultados={},
            )
            rec2.append(str(path))
            registros = record.ExperimentRecord.from_jsonl(str(path))
            self.assertEqual(len(registros), 2)
            self.assertEqual(registros[0].objective, "obj")
            self.assertEqual(registros[1].objective, "obj2")

    def test_from_jsonl_missing_file_returns_empty(self):
        self.assertEqual(record.ExperimentRecord.from_jsonl("/nonexistent/x.jsonl"), [])

    def test_from_jsonl_stops_on_malformed_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            path.write_text('{"experiment_id": "e00123456789abcd", "model_id": "m", "objective": "o"}\n'
                              'not json at all\n'
                              '{"experiment_id": "e00123456789abcde", "model_id": "m", "objective": "o2"}\n')
            registros = record.ExperimentRecord.from_jsonl(str(path))
            self.assertEqual(len(registros), 1)


class TestIndexByObjective(unittest.TestCase):

    def test_groups_preserving_order(self):
        r = lambda obj: record.ExperimentRecord(
            "e00123456789abcd", "m", obj, {}, {}, {}, {})
        grupos = record.index_by_objective([r("a"), r("b"), r("a")])
        self.assertEqual(list(grupos.keys()), ["a", "b"])
        self.assertEqual(len(grupos["a"]), 2)


if __name__ == "__main__":
    unittest.main()
