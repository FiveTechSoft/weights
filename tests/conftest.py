"""Shared pytest fixtures.

The NanoWSP-3 checkpoint (data/nanowsp3_weights.safetensors) is not versioned
(*.safetensors is git-ignored). Tests that need it used to fail on a clean clone.
This session fixture builds it with the same closed-form imprinting the demo script
uses (about one second on CPU) and removes it afterwards, but only when this run
created it, so a checkpoint the user already has is never touched.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "data" / "nanowsp3_weights.safetensors"


def _load_demo_module():
    path = ROOT / "scripts" / "demo_nanowsp3_capabilities.py"
    spec = importlib.util.spec_from_file_location("demo_nanowsp3_capabilities", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session", autouse=True)
def nanowsp3_checkpoint():
    if CHECKPOINT.exists():
        yield CHECKPOINT
        return
    sys.path.insert(0, str(ROOT))
    from engine.nanowsp3_runtime import NanoWSP3

    demo = _load_demo_module()
    model = NanoWSP3()
    model.imprint_thematic_corpus(
        thematic_domains=demo.THEMATIC_DOMAINS, target_layer=3, eta=3.5, reg_lambda=0.01
    )
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(CHECKPOINT)
    try:
        yield CHECKPOINT
    finally:
        CHECKPOINT.unlink(missing_ok=True)
