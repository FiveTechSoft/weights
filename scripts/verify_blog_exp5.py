"""
verify_blog_exp5.py
Mechanically verifies that the WSP-Lang program `programas/blog_exp5_write_mlp.wspl`
correctly synthesizes an MLP from scratch implementing:
    y(x) = 1.5 * x0 - 2.5 * x1 + 0.7
"""

import re
import numpy as np
from pathlib import Path

def test_blog_exp5():
    wspl_path = Path("programas/blog_exp5_write_mlp.wspl")
    text = wspl_path.read_text(encoding="utf-8")
    
    # Initialize empty weights
    W1 = np.zeros((2, 2), dtype=np.float32)
    b1 = np.zeros(2, dtype=np.float32)
    W2 = np.zeros((2, 1), dtype=np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    
    # Parse poke statements from WSP-Lang program
    pokes = re.findall(r"step\s+poke\s+([A-Za-z0-9_]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)", text)
    print(f"Extracted {len(pokes)} poke operations from {wspl_path}:")
    for tensor, r, c, val in pokes:
        r, c, val = int(r), int(c), float(val)
        print(f"  poke {tensor}[{r}, {c}] = {val}")
        if tensor == "W1":
            W1[r, c] = val
        elif tensor == "b1":
            b1[r] = val
        elif tensor == "W2":
            W2[r, c] = val
        elif tensor == "b2":
            b2[r] = val
            
    # Forward evaluation function
    def forward(x):
        # h = ReLU(x @ W1 + b1)
        h = np.maximum(x @ W1 + b1, 0.0)
        # y = h @ W2 + b2
        return float((h @ W2 + b2)[0])
        
    # Evaluate across 100 random test vectors in R^2 (including negative values)
    rng = np.random.default_rng(42)
    test_points = rng.uniform(-10.0, 10.0, size=(100, 2))
    errors = []
    
    for x in test_points:
        y_pred = forward(x)
        y_true = 1.5 * x[0] - 2.5 * x[1] + 0.7
        err = abs(y_pred - y_true)
        errors.append(err)
        
    max_err = max(errors)
    mean_err = np.mean(errors)
    print("\n=== MECHANICAL VERIFICATION RESULTS ===")
    print(f"Test vectors: 100 random samples in [-10, +10]^2")
    print(f"Max absolute error:  {max_err:.8e}")
    print(f"Mean absolute error: {mean_err:.8e}")
    
    assert max_err < 1e-5, f"Error exceeded tolerance: {max_err}"
    print("[PASS] Experiment 5 verified: WSP-Lang synthesized MLP achieves EXACT mathematical equivalence!")
    return True

if __name__ == "__main__":
    test_blog_exp5()
