import sys; sys.path.insert(0, r"C:\weights\lab\runs\loopfix\editor"); import wsp
nm = sys.argv[1]; specs = eval(sys.argv[2])  # list of (L,h,part,scale)
steps = sum((wsp.head_rows(L, h, p, s, nm) for L, h, p, s in specs), [])
m = wsp.run(nm, steps); print(m, wsp.count(m)[0])
