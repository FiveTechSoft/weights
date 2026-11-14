import sys, numpy as np; sys.path.insert(0, r"C:\weights\lab\runs\loopfix\editor"); import wsp
nm = sys.argv[1]; specs = eval(sys.argv[2])  # list of (L,h,ndirs,scale)
dirsrc = sys.argv[3] if len(sys.argv) > 3 else "dirs"
steps = []
for L, h, n, c in specs:
    E = np.load(rf"C:\weights\lab\runs\loopfix\editor\{dirsrc}_L{L}H{h}.npz")["E"]
    steps += wsp.head_dirs(L, h, E, [c] * n, nm)
m = wsp.run(nm, steps); print(m, wsp.count(m)[0])
