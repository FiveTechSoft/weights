"""Tests for challenge.py's filesystem-only surface (catalogo / estado / reto /
expediente / _verdict_stats / _save_exp) using a temporary challenges tree.

The judges (j_*) and sellar/verdicto flows read a real model or spawn llama.cpp
and are intentionally not covered here.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import challenge


def _fake_catalog(rids):
    return {"reglas_globales": ["no self-grading"], "retos": [{"id": r, "problema": "p-" + r} for r in rids]}


class TestChallengePure(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.RETOS = tmp
        self.EXP = tmp / "records"
        self.SELLADOS = tmp / "sellados"
        self.HIST = tmp / "historial.jsonl"
        self.EXP.mkdir()
        self.SELLADOS.mkdir()
        self.saved = {k: getattr(challenge, k) for k in
                      ("RETOS", "CATALOGO", "EXP", "SELLADOS", "HIST")}
        challenge.RETOS = self.RETOS
        challenge.CATALOGO = self.RETOS / "wsp40.json"
        challenge.EXP = self.EXP
        challenge.SELLADOS = self.SELLADOS
        challenge.HIST = self.HIST
        (self.RETOS / "wsp40.json").write_text(
            json.dumps(_fake_catalog(["01", "02", "03"])), encoding="utf-8")

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(challenge, k, v)
        self.tmp.cleanup()

    def test_catalogo_reads_json(self):
        cat = challenge.catalogo()
        self.assertEqual(len(cat["retos"]), 3)
        self.assertEqual(cat["reglas_globales"], ["no self-grading"])

    def test_estado_counts_approved_records(self):
        (self.EXP / "01.json").write_text(json.dumps({"estado": "aprobado"}))
        (self.EXP / "02.json").write_text(json.dumps({"estado": "en_progreso"}))
        out = challenge.estado()
        self.assertEqual(out["aprobados"], ["01"])
        self.assertEqual(out["n_aprobados"], 1)
        self.assertEqual(out["n_total"], 3)

    def test_estado_reports_verdicto_counts_from_history(self):
        self.HIST.write_text(
            json.dumps({"event": "verdicto", "vered": "keep"}) + "\n"
            + json.dumps({"event": "verdicto", "vered": "discard"}) + "\n"
            + json.dumps({"event": "verdicto", "vered": "keep"}) + "\n",
            encoding="utf-8")
        out = challenge.estado()
        self.assertEqual(out["verdictos"], {"keep": 2, "discard": 1})

    def test_reto_missing_id_returns_error(self):
        self.assertEqual(challenge.reto("99"), {"error": "reto no existe"})

    def test_reto_finds_pending_reto_excluding_approved(self):
        (self.EXP / "01.json").write_text(json.dumps({"estado": "aprobado"}))
        out = challenge.reto()
        self.assertEqual(out["reto"]["id"], "02")

    def test_reto_with_existing_record_attaches_expediente_summary(self):
        (self.EXP / "02.json").write_text(json.dumps({
            "estado": "aprobado", "score": 9.5, "intentos": 1,
            "hipotesis": "h", "intervencion": "i", "resultado": "ok",
            "principio": "p", "extra": "ignored"}))
        out = challenge.reto("02")
        exp = out["reto"]["expediente"]
        self.assertEqual(exp["estado"], "aprobado")
        self.assertNotIn("extra", exp)

    def test_expediente_missing_is_none(self):
        self.assertEqual(challenge.expediente("42"), {"expediente": None})

    def test_expediente_existing_round_trip(self):
        (self.EXP / "03.json").write_text(json.dumps({"estado": "en_progreso"}))
        out = challenge.expediente("03")
        self.assertEqual(out["expediente"]["estado"], "en_progreso")


if __name__ == "__main__":
    unittest.main()
