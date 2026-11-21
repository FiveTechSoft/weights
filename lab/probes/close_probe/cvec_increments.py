"""Turn a cumulative control vector (direction.N = mean h_improved - h_base after layer N) into per-layer increments.
llama.cpp adds direction.N at the output of every layer N and the residual stream carries it on, so a cumulative
difference would be summed ~N times. Increments: inc.1 = dir.1 (layer 0 cannot be steered), inc.N = dir.N - dir.N-1;
their running sum after layer N equals dir.N.
usage: python lab/probes/close_probe/cvec_increments.py <in.gguf> <out.gguf>
"""
import sys

import gguf
import numpy as np


def increments(dirs):
    ks = sorted(dirs)
    return {k: dirs[k] - (dirs[ks[i - 1]] if i else 0) for i, k in enumerate(ks)}


def main():
    r = gguf.GGUFReader(sys.argv[1])
    dirs = {int(t.name.split(".")[1]): np.array(t.data, dtype=np.float32) for t in r.tensors}
    inc = increments(dirs)
    w = gguf.GGUFWriter(sys.argv[2], "controlvector")
    w.add_string("controlvector.model_hint", "qwen35")
    w.add_uint32("controlvector.layer_count", len(inc))
    for k in sorted(inc):
        w.add_tensor(f"direction.{k}", inc[k].astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print({k: round(float(np.linalg.norm(v)), 2) for k, v in inc.items() if k in (1, 8, 16, 24, 31)})


if __name__ == "__main__":
    main()
