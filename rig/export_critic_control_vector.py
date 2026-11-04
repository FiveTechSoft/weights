"""export_critic_control_vector.py — Export llama.cpp GGUF Control Vector for MiMo-9B Self-Critique.
"""
import time
from pathlib import Path
import numpy as np
import gguf

ROOT = Path(__file__).resolve().parent.parent
VECTOR_NPZ = ROOT / "data" / "u_critic_L16.npz"
OUT_GGUF = ROOT / "data" / "u_critic_control_vector.gguf"

def export_gguf():
    print("=" * 70)
    print("EXPORTING MIMO-9B SELF-CRITIQUE CONTROL VECTOR TO GGUF")
    print("=" * 70)
    
    data = np.load(VECTOR_NPZ)
    u_critic = data["u_critic"].astype(np.float32)
    dim = len(u_critic)
    print(f"Loaded u_critic: dimension {dim}, norm = {np.linalg.norm(u_critic):.6f}")

    writer = gguf.GGUFWriter(str(OUT_GGUF), "controlvector")
    
    # MiMo-9B has 32 layers (0..31)
    # Critique subspace peaks at Layer 16, with Gaussian/cosine envelope across L13..L19
    peak_layer = 16
    bandwidth = 2.5
    
    print("\nLayer Activation Envelope:")
    for layer_idx in range(32):
        # Gaussian envelope centered at peak_layer
        weight = float(np.exp(-0.5 * ((layer_idx - peak_layer) / bandwidth) ** 2))
        if weight < 0.05:
            vec = np.zeros(dim, dtype=np.float32)
        else:
            vec = (u_critic * weight).astype(np.float32)
            print(f"  Layer {layer_idx:>2}: scale = {weight:.4f}")
        
        writer.add_tensor(f"direction.{layer_idx}", vec)
        
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_bytes = OUT_GGUF.stat().st_size
    print(f"\n[SUCCESS] Control vector exported to: {OUT_GGUF.relative_to(ROOT)}")
    print(f"File Size: {size_bytes:,} bytes ({size_bytes / 1024:.1f} KB)")

if __name__ == "__main__":
    export_gguf()
