exec(open("eda_theta.py").read().split("rows = []")[0])
qs=[.05,.1,.25,.5,.75,.9,.95,.99]
for sp in ["hyena","leopard"]:
    mt, me = (tr.species == sp).to_numpy(), (te.species == sp).to_numpy()
    Xs = l2n(np.vstack([Xtr[mt], Xte[me]])); Xc = l2n(Xs - Xs.mean(0)); P, T = Xc[:mt.sum()], Xc[mt.sum():]
    S = P @ P.T; np.fill_diagonal(S, -9); y = tr.identity.to_numpy()[mt]
    t1 = (T @ P.T).max(1); TT=T@T.T; np.fill_diagonal(TT,-9)
    print(sp, "TEST      top1", np.round(np.quantile(t1,qs),3), "frac>0.8 %.3f"%(t1>0.8).mean())
    # test q2q at CV-like pool size: subsample test pool to ~ fold size
    nf = int(mt.sum()*0.25); sub = np.random.default_rng(0).choice(len(T), min(nf,len(T)), replace=False)
    print(sp, "TEST q2q full pool med %.3f | subsampled pool (%d) med %.3f" % (np.median(TT.max(1)), len(sub), np.median(TT[np.ix_(sub,sub)].max(1))))
    for theta in [0.7,0.8,0.9]:
        f, un = split(S, y, 0, theta, 0.3); v=[];vk=[];vn=[]
        for k in range(4):
            q,g=np.flatnonzero(f==k),np.flatnonzero(f!=k); top=S[np.ix_(q,g)].max(1); v.append(top); vk.append(top[~un[q]]); vn.append(top[un[q]])
        v=np.concatenate(v); vk=np.concatenate(vk); vn=np.concatenate(vn)
        print(sp, f"CV th{theta} all ", np.round(np.quantile(v,qs),3), "frac>0.8 %.3f"%(v>0.8).mean())
        print(sp, f"CV th{theta} known", np.round(np.quantile(vk,qs),3))
        print(sp, f"CV th{theta} new  ", np.round(np.quantile(vn,qs),3))
    # viewpoint distribution
    print(pd.concat([tr[mt].viewpoint.fillna('unk').value_counts(normalize=True).rename('train'), te[me].viewpoint.fillna('unk').value_counts(normalize=True).rename('test')],axis=1).round(3).T.to_string())
    ic = tr[mt].identity.value_counts(); print(sp, "identity sizes quantiles", np.quantile(ic,[.1,.5,.9,.99]), "max", ic.max())
