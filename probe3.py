import gguf, numpy as np
r = gguf.GGUFReader(r"C:\Users\Anto\llamacpp\mimo-v2.6-distill-qwen-9b-Q6_K.gguf")
for nm in ("blk.5.ssm_beta.weight","blk.5.ssm_alpha.weight","blk.22.ssm_beta.weight","blk.17.ffn_down.weight","output.weight"):
    w = [x for x in r.tensors if x.name == nm][0]
    elems = int(np.prod(w.shape))
    print(nm, "shape(list)", list(w.shape), "elems", elems, "data.shape", w.data.shape, "dtype", w.data.dtype, "n_bytes", w.n_bytes, "mmap_off", w.data_offset, w.tensor_type.name)
