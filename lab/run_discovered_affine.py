"""Verify the actual checkpoint produced by the WSPLang discovery command."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors import safe_open
from engine.nanogpt_runtime import GPTConfig, NanoGPT


def main():
    torch.set_num_threads(2)
    data = Path("programs/register/discover_affine.json")
    path = Path("models/primitives/discovered_affine.safetensors")
    observed = json.loads(data.read_text())
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata()
    assert meta["provenance"] == "closed_form_symbolic_discovery"
    assert meta["method"] == "discover_register"
    assert meta["backward_calls"] == "0"
    assert meta["examples_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert meta["examples_count"] == str(len(observed))
    assert json.loads(meta["law"]) == [3, 13]
    cfg = json.loads(meta["architecture"])
    rt = NanoGPT(GPTConfig(vocab_size=cfg["vocab"], n_embd=cfg["d"],
        n_head=cfg["heads"], n_layer=cfg["layers"], block_size=cfg["block"],
        d_ff=cfg["ff"], bias=False))
    rt.load_checkpoint(path)
    rt.model.eval()
    excluded = {(x, y) for x, y, _ in observed}
    pairs = [(x,y) for x in range(16) for y in range(16) if (x,y) not in excluded]
    correct = 0
    with torch.no_grad():
        for start in range(0, len(pairs), 32):
            batch = pairs[start:start+32]
            ids = torch.tensor([[x,y,16] for x,y in batch])
            logits, _ = rt.forward(ids)
            assert torch.isfinite(logits).all()
            correct += sum(p == (3*x-3*y)%16 for p,(x,y) in
                           zip(logits[:,-1].argmax(-1).tolist(),batch))
    report = {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "examples_sha256": meta["examples_sha256"], "law": [3,13],
              "heldout_correct": correct, "heldout_total": len(pairs),
              "passed": correct == len(pairs) == 252,
              "parameters": rt.count_parameters(), "provenance": meta["provenance"]}
    out = Path("lab/runs/nanowsp_discovered_affine.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    assert report["passed"], report


if __name__ == "__main__":
    main()
