"""EDA: how does the burst threshold theta (difficulty of known validation queries) change the CV distribution of
top-1 similarity, and how well does each (theta, pi) match the TEST distribution? Features only, no SIFT."""
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp
D = "C:/Users/anupd/OneDrive/Desktop/ml/reid_student_data_main/"
tr, te, ids = pd.read_csv(D + "train.csv"), pd.read_csv(D + "test.csv"), pd.read_csv(D + "miewid_ids.csv")
F = np.load(D + "miewid_features.npy").astype(np.float32); pos = pd.Series(np.arange(len(ids)), index=ids.id)
Xtr, Xte = F[pos.loc[tr.id].to_numpy()], F[pos.loc[te.id].to_numpy()]
l2n = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)

def split(S, y, seed, theta, pi, k=4, cap=120):
    n = len(y); rng = np.random.default_rng(seed); lab = np.zeros(n, int); off = 0
    for i in np.unique(y):
        idx = np.flatnonzero(y == i)
        c = np.zeros(1, int) if len(idx) == 1 else connected_components(csr_matrix(S[np.ix_(idx, idx)] > theta), directed=False)[1]
        lab[idx] = c + off; off += c.max() + 1
    fold = np.full(n, -1)
    for i in np.unique(y):
        idx = np.flatnonzero(y == i); gs = np.unique(lab[idx]); rng.shuffle(gs); o = rng.integers(k)
        if len(gs) > 1:
            for j, g in enumerate(gs): fold[idx[lab[idx] == g]] = (j + o) % k
    idu, cnt = np.unique(y, return_counts=True); tot, new = 0, []
    for o in rng.permutation(len(idu)):
        if tot >= pi * n: break
        if cnt[o] > cap: continue
        new.append(idu[o]); tot += cnt[o]
    for i, kk in zip(new, rng.permutation(len(new)) % k): fold[y == i] = kk
    return fold, np.isin(y, new)

rows = []
for sp in ["hyena", "leopard"]:
    mt, me = (tr.species == sp).to_numpy(), (te.species == sp).to_numpy()
    Xs = l2n(np.vstack([Xtr[mt], Xte[me]])); Xc = l2n(Xs - Xs.mean(0)); P, T = Xc[:mt.sum()], Xc[mt.sum():]
    S = P @ P.T; np.fill_diagonal(S, -9); y = tr.identity.to_numpy()[mt]
    t1 = (T @ P.T).max(1)
    TT = T @ T.T; np.fill_diagonal(TT, -9); tq = TT.max(1)
    print(f"{sp}: test top1-to-train quantiles {np.round(np.quantile(t1, [.1,.25,.5,.75,.9]),3)} | test top1-to-test {np.round(np.quantile(tq,[.1,.5,.9]),3)} | share test whose NN is a test image {np.mean(tq > t1):.3f}")
    for theta in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for pi in [0.2, 0.25, 0.3, 0.35]:
            ks, kk, nk, q2q, nnt = [], [], [], [], []
            for seed in range(2):
                f, un = split(S, y, seed, theta, pi)
                v_all = []
                for k in range(4):
                    q, g = np.flatnonzero(f == k), np.flatnonzero(f != k)
                    top = S[np.ix_(q, g)].max(1); v_all.append(top)
                    kk.append(np.median(top[~un[q]])); nk.append((~un[q]).sum())
                    Sq = S[np.ix_(q, q)]; qq = Sq.max(1); q2q.append(np.median(qq)); nnt.append(np.mean(qq > top))
                v = np.concatenate(v_all); ks.append(ks_2samp(v, t1).statistic)
            rows.append(dict(sp=sp, theta=theta, pi=pi, KS=np.mean(ks), known_top1_med=np.mean(kk), n_known=np.sum(nk) / 2, q2q_med=np.mean(q2q), nn_is_query=np.mean(nnt)))
R = pd.DataFrame(rows)
print(R.round(3).to_string())
