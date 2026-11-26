"""Small experiment: does RootSIFT / other matching settings separate same-vs-different identity pairs better than the current SIFT setup?
Pairs: for random train queries, their top-40 MiewID neighbours in train (same-identity pairs above cos 0.8 excluded, like the CV folds)."""
import numpy as np, pandas as pd, cv2, time, sys
from concurrent.futures import ThreadPoolExecutor
D = "C:/Users/anupd/OneDrive/Desktop/ml/"
tr = pd.read_csv(D + "reid_student_data_main/train.csv"); ids = pd.read_csv(D + "reid_student_data_main/miewid_ids.csv")
F = np.load(D + "reid_student_data_main/miewid_features.npy").astype(np.float32)
pos = pd.Series(np.arange(len(ids)), index=ids.id); X = F[pos.loc[tr.id].to_numpy()]
X /= np.linalg.norm(X, axis=1, keepdims=True)
rng = np.random.default_rng(0)
NQ, M = int(sys.argv[1]) if len(sys.argv) > 1 else 300, 40
pairs = []
for sp in ["hyena", "leopard"]:
    rows = np.flatnonzero(tr.species == sp); Xs = X[rows]; Xs = Xs - Xs.mean(0); Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    S = Xs @ Xs.T; np.fill_diagonal(S, -9); y = tr.identity.to_numpy()[rows]
    same = y[:, None] == y[None, :]; S[same & (S > 0.8)] = -9
    for q in rng.choice(len(rows), NQ, replace=False):
        for g in np.argsort(-S[q])[:M]: pairs.append((rows[q], rows[g], int(same[q, g]), sp, q))
P = pd.DataFrame(pairs, columns=["q", "g", "y", "sp", "qq"]); print(len(P), "pairs, positives", P.y.mean())
need = np.unique(np.r_[P.q, P.g]); tid = tr.id.to_numpy()
SIFT = {i: (np.load(f"{D}sift_features_train.npz/{tid[i]}_xy.npy").reshape(-1, 2).astype(np.float32), np.load(f"{D}sift_features_train.npz/{tid[i]}_desc.npy").reshape(-1, 128).astype(np.float32)) for i in need}
def root(d): d = d / np.maximum(d.sum(1, keepdims=True), 1e-9); return np.sqrt(d).astype(np.float32)
ROOT = {i: (x, root(d)) for i, (x, d) in SIFT.items()}

def stats(src, ratio, method, thr):
    bf = cv2.BFMatcher(cv2.NORM_L2); out = []
    for q, g in zip(P.q, P.g):
        (xq, dq), (xg, dg) = src[q], src[g]
        if len(xq) < 4 or len(xg) < 4: out.append((0, 0)); continue
        knn = bf.knnMatch(dq, dg, k=2); good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
        if not good: out.append((0, 0)); continue
        rev = bf.match(dg, dq); rt = np.array([r.trainIdx for r in rev]); good = [m for m in good if rt[m.trainIdx] == m.queryIdx]
        if len(good) < 4: out.append((len(good), 0)); continue
        pq = np.float32([xq[m.queryIdx] for m in good]); pg = np.float32([xg[m.trainIdx] for m in good]); cv2.setRNGSeed(0)
        if method == "aff": _, mk = cv2.estimateAffinePartial2D(pq, pg, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=2000)
        elif method == "full": _, mk = cv2.estimateAffine2D(pq, pg, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=2000)
        else: _, mk = cv2.findHomography(pq, pg, cv2.USAC_MAGSAC, thr)
        out.append((len(good), 0 if mk is None else int(mk.sum())))
    return np.array(out)

from sklearn.metrics import roc_auc_score
def evaluate(st, name):
    P["inl"] = st[:, 1]; res = {"cfg": name}
    for sp in ["hyena", "leopard"]:
        m = (P.sp == sp).to_numpy(); res[f"auc_{sp}"] = roc_auc_score(P.y[m], P.inl[m] + 1e-3 * st[m, 0])
        # top-1 precision: among queries that have a positive among candidates, is the max-inlier candidate a positive?
        g = P[m].groupby("qq"); ok = []
        for _, d in g:
            if d.y.max() == 0: continue
            ok.append(d.y.to_numpy()[np.argmax(d.inl.to_numpy() + 1e-3 * np.arange(len(d))[::-1])])
        res[f"top1_{sp}"] = np.mean(ok)
    print(res, flush=True); return res

cfgs = [("sift r.78 aff5", SIFT, 0.78, "aff", 5.0), ("root r.78 aff5", ROOT, 0.78, "aff", 5.0), ("root r.85 aff5", ROOT, 0.85, "aff", 5.0),
        ("root r.80 aff8", ROOT, 0.80, "aff", 8.0), ("root r.80 full8", ROOT, 0.80, "full", 8.0), ("root r.80 H magsac5", ROOT, 0.80, "H", 5.0)]
cv2.setNumThreads(1)
with ThreadPoolExecutor(6) as ex:
    futs = [(c[0], ex.submit(stats, *c[1:])) for c in cfgs]
    rows = [evaluate(f.result(), n) for n, f in futs]
print(pd.DataFrame(rows).round(4).to_string())
