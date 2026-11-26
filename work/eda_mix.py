exec(open("eda_theta.py").read().split("rows = []")[0])
from scipy.optimize import nnls
for sp in ["hyena","leopard"]:
    mt, me = (tr.species == sp).to_numpy(), (te.species == sp).to_numpy()
    Xs = l2n(np.vstack([Xtr[mt], Xte[me]])); Xc = l2n(Xs - Xs.mean(0)); P, T = Xc[:mt.sum()], Xc[mt.sum():]
    S = P @ P.T; np.fill_diagonal(S, -9); y = tr.identity.to_numpy()[mt]
    t1 = np.sort((T @ P.T).max(1)); grid = np.quantile(t1, np.linspace(0.01,0.99,99))
    comps, names = [], []
    for theta in [0.5,0.6,0.7,0.8,0.9,0.95]:
        kn=[]; nw=[]
        for seed in range(3):
            f, un = split(S, y, seed, theta, 0.3)
            for k in range(4):
                q,g=np.flatnonzero(f==k),np.flatnonzero(f!=k); top=S[np.ix_(q,g)].max(1); kn.append(top[~un[q]]); nw.append(top[un[q]])
        comps.append(np.concatenate(kn)); names.append(f"known@{theta}")
        if theta==0.8: newc=np.concatenate(nw)
    comps.append(newc); names.append("new")
    A = np.stack([np.searchsorted(np.sort(c), grid, side="right")/len(c) for c in comps],1)
    b = np.searchsorted(t1, grid, side="right")/len(t1)
    A2=np.vstack([A, 50*np.ones(len(comps))]); b2=np.r_[b,50]
    w,_ = nnls(A2,b2); print(sp, dict(zip(names, np.round(w,3))), "sum", w.sum().round(3), "maxCDFerr %.3f"%np.abs(A@w-b).max())
    for pin in [0.25,0.3,0.35,0.4]:
        # constrain new share: fit only known mix
        A3=np.vstack([A[:,:-1], 50*np.ones(len(comps)-1)]); b3=np.r_[b-pin*A[:,-1], 50*(1-pin)]
        w3,_=nnls(A3,b3); print("   pi_new=%.2f"%pin, dict(zip(names[:-1],np.round(w3,3))), "maxCDFerr %.3f"%np.abs(A[:,:-1]@w3+pin*A[:,-1]-b).max())
