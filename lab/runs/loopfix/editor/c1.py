import sys; sys.path.insert(0, r"C:\weights\lab\runs\loopfix\editor"); import wsp
for nm, sc in [("v53_050", 0.5)]:
    m = wsp.run(nm, wsp.head_rows(5, 3, 2, sc, nm)); print(m, wsp.count(m))
