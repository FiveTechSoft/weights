"""Create data/nanowsp_key.npz, the unit key vector programs/nanowsp_architecture.wspl expects.

Same deterministic recipe as tests/test_nanowsp.py: numpy seed 42, a 64-dimensional standard normal vector
(the d_model of NanoWSP), normalised to unit length, stored under the name `u`. It is an arbitrary direction used to
exercise the Weyl, Davis-Kahan and delta_unlearn steps, not a learned concept.

    python scripts/make_nanowsp_key.py [--out data/nanowsp_key.npz]
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def make_key(dim: int = 64, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "nanowsp_key.npz"))
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, u=make_key())
    print(out)


if __name__ == "__main__":
    main()
