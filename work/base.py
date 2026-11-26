import matplotlib; matplotlib.use("Agg")

def display(x): print(x)

import os, sys, json, time, glob, hashlib, warnings, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 0
np.random.seed(SEED)
T_START = time.time()
def log(*a): print(f"[{time.time() - T_START:6.0f}s]", *a, flush=True)

def _env_int(name, default): return int(os.environ.get(name, default))

def _resolve_data_dir():
    for c in [os.environ.get("DATA_DIR"), "/home/urshita/Documents/college/sem5/ml/project/projectwild3/reid_student_data_main", "reid_student_data_main", "."]:
        if c and os.path.isfile(os.path.join(c, "train.csv")): return c
    raise FileNotFoundError("train.csv not found - set the DATA_DIR environment variable to the folder that contains it.")
DATA_DIR   = _resolve_data_dir()
PARENT_DIR = os.environ.get("PARENT_DIR", os.path.dirname(os.path.abspath(DATA_DIR)))
CACHE_DIR, FIG_DIR = os.environ.get("CACHE_DIR", "cache"), "figures"
OUT_DIR    = os.environ.get("OUT_DIR", ".")
for d in (CACHE_DIR, FIG_DIR, OUT_DIR): os.makedirs(d, exist_ok=True)

NEW, SPECIES = "new_individual", ["hyena", "leopard"]
SP_ID = {s: i for i, s in enumerate(SPECIES)}

# ---------- pipeline settings ----------
N_FOLDS        = _env_int("N_FOLDS", 4)          # 4 folds: val/gallery ratio 1:3, closest to test/train = 0.39
# ---- test-like validation design (see section 2) ----
CV_SEEDS       = tuple(int(x) for x in os.environ.get("CV_SEEDS", "0,1,2").split(","))   # independent partitions; models are trained on all of them
RUN_STRESS     = os.environ.get("RUN_STRESS", "1") == "1"       # evaluation-only 'hard' scheme (never used for training)
THETA, CAP_NEW = 0.80, 120
PI_NEW = float(os.environ.get("PI_NEW", "0.30"))   # v6: 0.40 -> 0.30. Feature-only mixture fits put the test new-share at ~0.28-0.30 (see eda_calibration.py)                        # burst threshold (cosine), target share of images in new identities, max size of a 'new' identity
NEW_QUERY_FRAC = PI_NEW                                         # used by the stress folds
PRIOR_SHIFT    = os.environ.get("PRIOR_SHIFT", "auto")          # off | auto (conservative, within the estimated band) | force (point estimate)
PCA_DIM        = _env_int("PCA_DIM", 512)
RERANK_K1, RERANK_K2, RERANK_LAM = 20, 6, 0.3
CAND_K, TOP_M_SIFT = _env_int("CAND_K", 300), _env_int("TOP_M_SIFT", 150)
SIFT_RATIO, RANSAC_THRESH, RANSAC_ITERS = 0.78, 5.0, 2000
KN             = 5
NN_STRONG      = 8
TOP_ID         = _env_int("TOP_ID", 30)
TOP_L          = _env_int("TOP_L", 10)
CAND_LDA       = _env_int("CAND_LDA", 60)
ALPHAS         = [1.0, 0.6, 0.35, 0.2, 0.1, 0.05, 0.02, 0.01]
FAST           = os.environ.get("FAST", "0") == "1"
MODEL_FOLDS    = 5
TOP_M_EFF      = TOP_M_SIFT + CAND_LDA
SUBMIT_VERSION = "v4"
assert CAND_K >= TOP_M_EFF

# V2 hand-tuned parameters from the previous notebook (full precision file is used if present)
BEST_PARAMS = dict(w_rerank=0.2812, w_sift=0.1906, b_same=-0.0062, b_adj=0.0094, b_opp=0.0125, agg="top2", floor=3)
_cfgp = next((p for p in [os.path.join(d, "best_config_v2.json") for d in [".", DATA_DIR, PARENT_DIR]] if os.path.isfile(p)), None)
if _cfgp:
    BEST_PARAMS = json.load(open(_cfgp))["params"]; print("Loaded V2 params from", _cfgp)
assert SUBMIT_VERSION in ("v1", "v2", "v3", "v4")
print("DATA_DIR:", DATA_DIR, "| folds:", N_FOLDS, "| new-share target:", NEW_QUERY_FRAC, "| FAST:", FAST)

def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, name), dpi=150); plt.close("all")

def load_data(d=DATA_DIR):
    tr = pd.read_csv(f"{d}/train.csv"); te = pd.read_csv(f"{d}/test.csv")
    ids = pd.read_csv(f"{d}/miewid_ids.csv")
    col = "id" if "id" in ids.columns else ids.columns[0]
    F = np.load(f"{d}/miewid_features.npy").astype(np.float32)
    pos = pd.Series(np.arange(len(ids)), index=ids[col].to_numpy())
    Xtr, Xte = F[pos.loc[tr.id.to_numpy()].to_numpy()], F[pos.loc[te.id.to_numpy()].to_numpy()]   # KeyError if ids misalign
    for df in (tr, te): df["viewpoint"] = df["viewpoint"].fillna("unk").astype(str)
    return tr, te, Xtr, Xte

tr, te, Xtr, Xte = load_data()
print(tr.shape, te.shape, Xtr.shape, Xte.shape)
CLASSES, code_tr = np.unique(tr.identity.astype(str).to_numpy(), return_inverse=True)   # one global code per identity
code_tr = code_tr.astype(np.int32)
print(tr.groupby("species").identity.agg(["size", "nunique"]))
print("identities with a single training image:", int((tr.identity.value_counts() == 1).sum()))

NEG = -1e6

def ap5(pred, truth):
    """Average precision at 5 for one query (labels are strings)."""
    for k, p in enumerate(pred[:5]):
        if p == truth: return 1.0 / (k + 1)
    return 0.0

def map5(preds, truths):
    return float(np.mean([ap5(p, t) for p, t in zip(preds, truths)]))

assert abs(map5([["B", "C", "A"], ["x", "y", "N"], ["N"]], ["A", "N", "A"]) - (1/3 + 1/3 + 0) / 3) < 1e-12   # worked example sanity

def l2n(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp

ROWS_SP = {sp: np.flatnonzero((tr.species == sp).to_numpy()) for sp in SPECIES}
ID_SP   = {sp: tr.identity.astype(str).to_numpy(dtype=object)[ROWS_SP[sp]] for sp in SPECIES}

def _train_sim(sp):
    """Centred, L2-normalised MiewID cosine between train images (centring uses features only, no labels). Used only to BUILD/AUDIT the folds."""
    m_tr, m_te = (tr.species == sp).to_numpy(), (te.species == sp).to_numpy()
    Xs = l2n(np.vstack([Xtr[m_tr], Xte[m_te]])); Xc = l2n(Xs - Xs.mean(0))
    P = Xc[:int(m_tr.sum())]; S = P @ P.T; np.fill_diagonal(S, -9); return S

def burst_clusters(S, y, theta):
    """Connected components of {cos > theta} inside each identity = pseudo-encounters/bursts."""
    lab = np.zeros(len(y), int); off = 0
    for i in np.unique(y):
        idx = np.flatnonzero(y == i)
        if len(idx) == 1: c = np.zeros(1, int)
        else: _, c = connected_components(csr_matrix(S[np.ix_(idx, idx)] > theta), directed=False)
        lab[idx] = c + off; off += c.max() + 1
    return lab

def testlike_split(S, y, seed, k=N_FOLDS, theta=THETA, pi=PI_NEW, cap=CAP_NEW):
    """-> fold (n,) with -1 = never validated, and the set of identities held out entirely (the 'new' ones)."""
    n = len(y); rng = np.random.default_rng(seed)
    groups = burst_clusters(S, y, theta); fold = np.full(n, -1)
    for i in np.unique(y):                                   # known identities: spread whole burst clusters over folds
        idx = np.flatnonzero(y == i); gs = np.unique(groups[idx]); rng.shuffle(gs); off = rng.integers(k)
        if len(gs) > 1:
            for j, gk in enumerate(gs): fold[idx[groups[idx] == gk]] = (j + off) % k
    idu, cnt = np.unique(y, return_counts=True); tot, newids = 0, []
    for o in rng.permutation(len(idu)):                      # new identities: the whole identity goes into ONE fold
        if tot >= pi * n: break
        if cnt[o] > cap: continue
        newids.append(idu[o]); tot += cnt[o]
    for i, kk in zip(newids, rng.permutation(len(newids)) % k): fold[y == i] = kk
    return fold, set(newids)

_SIMS = {sp: _train_sim(sp) for sp in SPECIES}
FOLDS = {}                                                   # scheme -> (fold per train row [-1 = never a query], held-out-identity flag per train row)
for seed in CV_SEEDS:
    fold = np.full(len(tr), -1); unseen = np.zeros(len(tr), bool)
    for sp in SPECIES:
        f_sp, newids = testlike_split(_SIMS[sp], ID_SP[sp], seed)
        fold[ROWS_SP[sp]] = f_sp; unseen[ROWS_SP[sp]] = np.isin(ID_SP[sp], list(newids))
    FOLDS[f"s{seed}"] = (fold, unseen)
TRS = tuple(FOLDS); PRIMARY = TRS[0]                         # TRS = schemes used for training AND validation

def audit_folds(name):
    """Hard leakage checks: disjoint, held-out identities absent from the gallery, known queries present in it, no burst/duplicate leakage."""
    fold, unseen = FOLDS[name]
    for sp in SPECIES:
        rows = ROWS_SP[sp]; f = fold[rows]; u = unseen[rows]; y = ID_SP[sp]; S = _SIMS[sp]
        for k in range(N_FOLDS):
            v = np.flatnonzero(f == k); g = np.flatnonzero(f != k)
            if len(v) == 0: continue
            assert not (set(v.tolist()) & set(g.tolist()))
            gl = set(y[g].tolist()); isnew = np.array([yy not in gl for yy in y[v]])
            assert (isnew == u[v]).all(), f"{name}/{sp}/fold {k}: held-out flag inconsistent with gallery membership"
            for i in set(y[v][isnew].tolist()): assert (f[y == i] == k).all(), "a new identity is split over folds"
            kn = v[~isnew]
            if len(kn):
                ms = np.where(y[kn][:, None] == y[g][None, :], S[np.ix_(kn, g)], -9).max(1)
                assert (ms > -9).all(), "known query without its identity in the gallery"
                assert ms.max() <= THETA + 1e-6, f"burst leakage: same-identity similarity {ms.max():.3f} > {THETA}"
            assert (S[np.ix_(v, g)] > 0.9999).sum() == 0, "exact duplicate vectors across val/gallery"
for name in TRS: audit_folds(name)
log(f"validation folds {TRS}: leakage audit PASSED (disjoint; held-out identities absent from gallery; no same-identity pair above {THETA})")

# reproducibility check against the standalone split notebook (splits_v1.csv), if present
for p in [os.environ.get("SPLITS_CSV"), "splits_v1.csv", os.path.join(OUT_DIR, "splits_v1.csv"), os.path.join(DATA_DIR, "splits_v1.csv")]:
    if p and os.path.isfile(p):
        ref = pd.read_csv(p); ok = (ref.id.to_numpy() == tr.id.to_numpy()).all() and N_FOLDS == 4 and (THETA, PI_NEW, CAP_NEW) == (0.8, 0.4, 120)
        for s_ in CV_SEEDS:
            if ok and f"fold_s{s_}" in ref: ok &= bool((ref[f"fold_s{s_}"].to_numpy() == FOLDS[f"s{s_}"][0]).all() and (ref[f"heldout_identity_s{s_}"].to_numpy() == FOLDS[f"s{s_}"][1]).all())
        log(f"folds identical to {p}: {ok}"); break

for name in TRS:
    f_, u_ = FOLDS[name]; q_ = f_ >= 0
    log(f"CV scheme {name}: queries/fold {np.bincount(f_[q_], minlength=N_FOLDS).tolist()} | new share " +
        ", ".join(f"{sp} {u_[ROWS_SP[sp]][f_[ROWS_SP[sp]] >= 0].mean():.1%}" for sp in SPECIES) + f" | never-validated images {int((~q_).sum())}")
del _SIMS

def l2n(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

def pca_embed(Xtr_sp, Xte_sp, dim=None):
    dim = dim or PCA_DIM
    X = l2n(np.vstack([Xtr_sp, Xte_sp]).astype(np.float32))
    X = l2n(X - X.mean(0, keepdims=True))
    dim = min(dim, X.shape[0] - 1, X.shape[1])
    X = l2n(PCA(dim, whiten=True, random_state=SEED).fit_transform(X).astype(np.float32))   # unsupervised: no labels used
    return X[:len(Xtr_sp)], X[len(Xtr_sp):]

PCA_DIM = min(PCA_DIM, min(int((tr.species == s_).sum() + (te.species == s_).sum()) for s_ in SPECIES) - 1)
SUB, TE_SUB, EMB, RAW = {}, {}, {}, {}
for sp in SPECIES:
    m_tr, m_te = (tr.species == sp).to_numpy(), (te.species == sp).to_numpy()
    SUB[sp], TE_SUB[sp] = np.flatnonzero(m_tr), np.flatnonzero(m_te)
    EMB[sp] = pca_embed(Xtr[m_tr], Xte[m_te])
    RAW[sp] = (l2n(Xtr[m_tr]), l2n(Xte[m_te]))
    print(f"{sp}: train={m_tr.sum()} test={m_te.sum()} identities={len(np.unique(code_tr[m_tr]))}")

E_TR = np.zeros((len(tr), EMB[SPECIES[0]][0].shape[1]), np.float32)
for sp in SPECIES: E_TR[SUB[sp]] = EMB[sp][0]

def make_stress_folds(tr, k=N_FOLDS, new_frac=NEW_QUERY_FRAC, seed=SEED):
    """ORIGINAL 'hard' scheme, kept verbatim as a stress test (NOT a validation estimate)."""
    rng = np.random.default_rng(seed)
    groups = tr.groupby("identity").indices
    sizes = pd.Series({i: len(r) for i, r in groups.items()})
    unseen = set(sizes.index[sizes == 1]); n_unseen = len(unseen)
    for i in rng.permutation(sizes.index[sizes >= 2].to_numpy()):
        if n_unseen >= new_frac * len(tr): break
        unseen.add(i); n_unseen += sizes[i]
    fold = np.zeros(len(tr), int); is_unseen = np.zeros(len(tr), bool)
    for ident, rows in groups.items():
        if ident in unseen:
            fold[rows] = rng.integers(k); is_unseen[rows] = True; continue
        if len(rows) >= 3:
            Es = E_TR[rows]; Dm = np.clip(1.0 - Es @ Es.T, 0, None); np.fill_diagonal(Dm, 0); Dm = (Dm + Dm.T) / 2
            lab = fcluster(linkage(squareform(Dm, checks=False), "average"), t=int(np.clip(round(len(rows) / 4), 2, len(rows))), criterion="maxclust") - 1
            nc = int(lab.max()) + 1
            if nc >= 2:
                order = rng.permutation(nc); start = rng.integers(k); cf = np.empty(nc, int); cf[order] = (start + np.arange(nc)) % k
                fold[rows] = cf[lab]; continue
        rows = rng.permutation(rows); start = rng.integers(k)
        fold[rows] = (start + np.arange(len(rows))) % k
    return fold, is_unseen

EVS = ()                                                   # evaluation-only schemes
if RUN_STRESS:
    FOLDS["stress"] = make_stress_folds(tr, seed=SEED + 7); EVS = ("stress",)
ALLS = TRS + EVS
for k_, (f_, u_) in FOLDS.items():
    q_ = f_ >= 0
    log(f"scheme {k_:7s} ({'train+val' if k_ in TRS else 'EVAL ONLY'}): queries/fold {np.bincount(f_[q_], minlength=N_FOLDS).tolist()} | new share {u_[q_].mean():.1%}")

# identity-grouped folds for model fitting (all images/queries of an identity always fall in the same model fold)
MODEL_FOLDS = 5
_rg = np.random.default_rng(SEED + 99)
_gmap = dict(zip(CLASSES, _rg.integers(MODEL_FOLDS, size=len(CLASSES))))
MF_TR = np.array([_gmap[c] for c in tr.identity.astype(str).to_numpy()], dtype=np.int8)

def within_scatter(X, y):
    u, inv = np.unique(y, return_inverse=True)
    M = sparse.csr_matrix((np.ones(len(y), np.float32), (inv, np.arange(len(y)))), shape=(len(u), len(y)))
    means = (M @ X) / np.asarray(M.sum(1)).ravel()[:, None]
    Xc = (X - means[inv]).astype(np.float64)
    return Xc.T @ Xc / max(len(y) - len(u), 1)

def wccn_matrix(Sw, alpha):
    d = Sw.shape[0]; S = (1 - alpha) * Sw + alpha * np.trace(Sw) / d * np.eye(d)
    ev, V = np.linalg.eigh(S); ev = np.maximum(ev, 1e-8 * ev.max())
    return ((V * ev ** -0.5) @ V.T).astype(np.float32)

def fit_proj(Pg, yg, alpha, Sw=None):
    Sw = within_scatter(Pg, yg) if Sw is None else Sw
    return Pg.mean(0), wccn_matrix(Sw, alpha)
def apply_proj(X, mu, W): return l2n((X - mu) @ W)

def select_alpha(sp):
    """Closed-set top-1 of the supervised projection vs shrinkage alpha, on the TEST-LIKE folds (all training seeds).
    Also logs a cross-fit optimism: pick alpha on the other folds of the same seed, score on the held-out fold."""
    rows = SUB[sp]; P_tr = EMB[sp][0]; y = code_tr[rows]; cells = []
    for sch in TRS:
        f = FOLDS[sch][0][rows]; un = FOLDS[sch][1][rows]
        for k in range(N_FOLDS):
            g, q = f != k, (f == k) & ~un
            if q.sum() == 0: continue
            Sw = within_scatter(P_tr[g], y[g]); accs = []
            for a in ALPHAS:
                mu, W = fit_proj(P_tr[g], y[g], a, Sw)
                pred = y[g][np.argmax(apply_proj(P_tr[q], mu, W) @ apply_proj(P_tr[g], mu, W).T, 1)]
                accs.append(float((pred == y[q]).mean()))
            cells.append((sch, k, accs))
    A = np.array([c[2] for c in cells]); m = dict(zip(ALPHAS, A.mean(0))); best = max(m, key=m.get); bi = ALPHAS.index(best)
    cf = []
    for j, (s_, k_, _) in enumerate(cells):
        sel = [i for i, (s2, k2, _) in enumerate(cells) if s2 == s_ and k2 != k_]
        cf.append(A[j, int(np.argmax(A[sel].mean(0)))])
    log(f"  {sp}: best alpha {best} (top-1 {m[best]:.4f}; no projection {m[1.0]:.4f}); cross-fit selection optimism {A[:, bi].mean() - np.mean(cf):+.4f}")
    return best, m

ALPHA, ALPHA_TAB = {}, {}
for sp in SPECIES: ALPHA[sp], ALPHA_TAB[sp] = select_alpha(sp)

def k_reciprocal_rerank(Q, G, k1=RERANK_K1, k2=RERANK_K2, lam=RERANK_LAM, chunk=1024):
    """Q (nq,d), G (ng,d): L2-normalised. Returns (S_rerank (nq,ng), S_cos (nq,ng))."""
    Q = np.ascontiguousarray(Q, dtype=np.float32); G = np.ascontiguousarray(G, dtype=np.float32)
    nq, ng = len(Q), len(G)
    X = np.vstack([Q, G]); N = nq + ng
    S_cos = Q @ G.T
    kk = min(k1, N - 1)
    D = np.empty((N, N), np.float32)                      # row-max-normalised squared euclidean distance
    nn = np.empty((N, kk + 1), np.int32)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        d = 2.0 - 2.0 * (X[a:b] @ X.T)
        np.maximum(d, 0, out=d)
        d /= np.maximum(d.max(1, keepdims=True), 1e-12)
        D[a:b] = d
        part = np.argpartition(d, kk, axis=1)[:, :kk + 1]
        o = np.argsort(np.take_along_axis(d, part, 1), axis=1)
        nn[a:b] = np.take_along_axis(part, o, 1)
    ar = np.arange(N)
    for i in np.flatnonzero(nn[:, 0] != ar):             # exact duplicates can displace self from position 0
        w = np.flatnonzero(nn[i] == i)
        if len(w): nn[i, [0, w[0]]] = nn[i, [w[0], 0]]
        else:      nn[i] = np.r_[i, nn[i, :-1]]
    rec = (nn[nn] == ar[:, None, None]).any(2)
    h = min(int(round(k1 / 2)), kk)
    nnh = nn[:, :h + 1]
    rech = (nnh[nnh] == ar[:, None, None]).any(2)
    Rfull = [nn[i][rec[i]] for i in range(N)]
    Rhalf = [set(nnh[i][rech[i]].tolist()) for i in range(N)]
    rows, cols, vals = [], [], []
    for i in range(N):
        R = Rfull[i]; Rset = set(R.tolist()); add = set()
        for c in R.tolist():
            rc = Rhalf[c]
            if rc and sum(1 for x in rc if x in Rset) > (2.0 / 3.0) * len(rc):
                add |= rc
        idx = np.fromiter(Rset | add, dtype=np.int64)
        w = np.exp(-D[i, idx])
        rows.append(np.full(len(idx), i, np.int64)); cols.append(idx); vals.append((w / w.sum()).astype(np.float32))
    V = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    if k2 > 1:                                            # local query expansion
        k2e = min(k2, kk + 1)
        A = sparse.csr_matrix((np.full(N * k2e, 1.0 / k2e, np.float32), (np.repeat(ar, k2e), nn[:, :k2e].ravel())), shape=(N, N))
        V = (A @ V).tocsr()
    Vc = V.tocsc()
    rp, ri, rd = V.indptr, V.indices, V.data
    cp, ci, cd = Vc.indptr, Vc.indices, Vc.data
    J = np.empty((nq, ng), np.float32)                    # jaccard distance, queries x gallery
    for i in range(nq):
        cs, vs = ri[rp[i]:rp[i + 1]], rd[rp[i]:rp[i + 1]]
        st, en = cp[cs].astype(np.int64), cp[cs + 1].astype(np.int64)
        cnt = en - st; tot = int(cnt.sum())
        gather = np.arange(tot) + np.repeat(st - (np.cumsum(cnt) - cnt), cnt)
        tm = np.bincount(ci[gather], weights=np.minimum(cd[gather], np.repeat(vs, cnt)), minlength=N)
        J[i] = (1.0 - tm / (2.0 - tm))[nq:]
    S_rerank = 1.0 - ((1.0 - lam) * J + lam * D[:nq, nq:])
    return S_rerank.astype(np.float32), S_cos

BK_SAME, BK_ADJ, BK_OPP, BK_MISSING, BK_OTHER = 0, 1, 2, 3, 4
_ADJ, _OPP = set(), set()
for a, bs in {"left": ["frontleft", "backleft"], "right": ["frontright", "backright"],
              "front": ["frontleft", "frontright"], "back": ["backleft", "backright"]}.items():
    for b in bs: _ADJ |= {(a, b), (b, a)}
for a, b in [("left", "right"), ("front", "back"), ("up", "down")]: _OPP |= {(a, b), (b, a)}

def vp_bucket(vq, vg):
    if vq == "unk" or vg == "unk": return BK_MISSING
    if vq == vg:                   return BK_SAME
    if (vq, vg) in _ADJ:           return BK_ADJ
    if (vq, vg) in _OPP:           return BK_OPP
    return BK_OTHER

VP_VOCAB = sorted(set(tr.viewpoint) | set(te.viewpoint) | {"unk"})
VP_IDX = {v: i for i, v in enumerate(VP_VOCAB)}
VP_TABLE = np.array([[vp_bucket(a, b) for b in VP_VOCAB] for a in VP_VOCAB], dtype=np.int8)
vp_id_tr = tr.viewpoint.map(VP_IDX).to_numpy().astype(np.int16)
vp_id_te = te.viewpoint.map(VP_IDX).to_numpy().astype(np.int16)
print("viewpoints:", VP_VOCAB)

# ---------- SIFT source (.npz, or an unzipped folder of <id>_xy.npy / <id>_desc.npy) ----------
def locate_sift(name):
    roots = [os.environ.get("SIFT_DIR"), DATA_DIR, PARENT_DIR, os.getcwd()] + sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))
    for r in roots:
        if not r: continue
        for nm in (name, name[:-4] if name.endswith(".npz") else name):
            p = os.path.join(r, nm)
            if os.path.exists(p): return p
    raise FileNotFoundError(f"{name} not found in {roots} - set SIFT_DIR to the folder that contains it.")

class SiftSource:
    def __init__(self, path):
        self.zs, self.dir, self.map = [], None, None
        if os.path.isfile(path): self.zs = [np.load(path)]
        else:
            npzs = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
            if npzs: self.zs = [np.load(p) for p in npzs]
            else:    self.dir = path
        print("SIFT source:", path, "| mode:", "npz" if self.zs else "folder of .npy")
    def get(self, key):
        for z in self.zs:
            try: return z[key]
            except KeyError: continue
        if self.dir is not None:
            p = os.path.join(self.dir, key + ".npy")
            if os.path.exists(p): return np.load(p)
            if self.map is None:
                self.map = {os.path.splitext(os.path.basename(q))[0]: q for q in glob.glob(os.path.join(self.dir, "**", "*.npy"), recursive=True)}
            return np.load(self.map[key])
        raise KeyError(key)

_TR_IDS, _TE_IDS = set(tr.id.to_numpy()), set(te.id.to_numpy())
def load_sift_for(ids_needed):
    srcs = [(SiftSource(locate_sift("sift_features_train.npz")), _TR_IDS), (SiftSource(locate_sift("sift_features_test.npz")), _TE_IDS)]
    out, t0 = {}, time.time()
    for n, i in enumerate(ids_needed):
        for src, idset in srcs:
            if i in idset:
                out[i] = (src.get(f"{i}_xy").reshape(-1, 2).astype(np.float32), src.get(f"{i}_desc").reshape(-1, 128)); break
        if (n + 1) % 2000 == 0: print(f"  loaded {n+1}/{len(ids_needed)} SIFT arrays [{time.time()-t0:.0f}s]")
    print(f"SIFT loaded: {len(out)} images [{time.time()-t0:.0f}s]")
    return out

_tls = threading.local()
def _bf():
    if not hasattr(_tls, "bf"): _tls.bf = cv2.BFMatcher(cv2.NORM_L2)
    return _tls.bf

def _pair_stats(bf, xq, dq32, xg, dg):
    """(ratio_matches, ransac_inliers) for one image pair."""
    if len(xq) < 4 or len(xg) < 4: return 0, 0
    dg32 = dg.astype(np.float32)
    knn = bf.knnMatch(dq32, dg32, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < SIFT_RATIO * m[1].distance]      # Lowe ratio test
    if not good: return 0, 0
    rev = bf.match(dg32, dq32)                                                                      # mutual NN check
    rt = np.fromiter((r.trainIdx for r in rev), dtype=np.int64, count=len(rev))
    good = [m for m in good if rt[m.trainIdx] == m.queryIdx]
    if len(good) < 4: return len(good), 0
    pq = np.float32([xq[m.queryIdx] for m in good]); pg = np.float32([xg[m.trainIdx] for m in good])
    cv2.setRNGSeed(0)
    _, mask = cv2.estimateAffinePartial2D(pq, pg, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH, maxIters=RANSAC_ITERS)
    return len(good), (0 if mask is None else int(mask.sum()))

def run_sift(SIFT, q_list, g_lists, label, chunk=20):
    """q_list[i] is matched against every id in g_lists[i]. Returns int32 (total_pairs, 4): n_kp_q, n_kp_g, ratio_matches, ransac_inliers."""
    cv2.setNumThreads(1)
    def work(items):
        bf, out = _bf(), []
        for qid, gids in items:
            xq, dq = SIFT[qid]; dq32 = dq.astype(np.float32)
            for gid in gids:
                xg, dg = SIFT[gid]
                nm, ni = _pair_stats(bf, xq, dq32, xg, dg)
                out.append((len(xq), len(xg), nm, ni))
        return np.array(out, dtype=np.int32).reshape(-1, 4)
    tasks = [list(zip(q_list[a:a + chunk], g_lists[a:a + chunk])) for a in range(0, len(q_list), chunk)]
    workers = max(1, os.cpu_count() or 1)
    print(f"[{label}] {sum(len(g) for g in g_lists):,} pairs on {workers} thread(s)")
    t0, res = time.time(), []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, r in enumerate(ex.map(work, tasks)):
            res.append(r)
            if (n + 1) % max(1, len(tasks) // 10) == 0: print(f"  {n+1}/{len(tasks)} chunks [{time.time()-t0:.0f}s]")
    return np.concatenate(res)

# ---------- pair cache: int64 key = query_index * N + gallery_index over the concatenated [train, test] id list ----------
ALL_IDS = np.concatenate([tr.id.to_numpy(dtype=object), te.id.to_numpy(dtype=object)]); NID = len(ALL_IDS)
_IDX = pd.Index(ALL_IDS); assert _IDX.is_unique, "image ids must be unique across train and test"
def pair_key(q, g):
    qi, gi = _IDX.get_indexer(q), _IDX.get_indexer(g)
    assert (qi >= 0).all() and (gi >= 0).all(), "unknown image id"
    return qi.astype(np.int64) * NID + gi

class PairCache:
    def __init__(self, path):
        self.path = path
        self.sig = dict(ratio=SIFT_RATIO, ransac=RANSAC_THRESH, iters=RANSAC_ITERS, ids=hashlib.md5("|".join(ALL_IDS).encode()).hexdigest()[:12])
        self.keys, self.stats = np.empty(0, np.int64), np.empty((0, 4), np.int32)
        src = path if os.path.exists(path) else next(iter(sorted(glob.glob("/kaggle/input/*/sift_pair_cache.npz") + glob.glob("/kaggle/input/*/*/sift_pair_cache.npz"))), None)
        if src:
            z = np.load(src)
            if json.loads(str(z["sig"])) == self.sig: self.keys, self.stats = z["keys"], z["stats"]; print(f"pair cache: {len(self.keys):,} pairs loaded")
            else: print("pair cache signature differs (params or ids changed) -> ignored")
        else: print("no pair cache found -> every SIFT pair will be computed (slow on the first run)")
    def lookup(self, keys):
        if len(self.keys) == 0: return np.full(len(keys), -1, np.int64)
        i = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return np.where(self.keys[i] == keys, i, -1)
    def add(self, keys, stats):
        allk = np.concatenate([self.keys, keys]); alls = np.concatenate([self.stats, stats.astype(np.int32)])
        u, first = np.unique(allk, return_index=True)
        self.keys, self.stats = u, alls[first]
    def save(self):
        np.savez_compressed(self.path, keys=self.keys, stats=self.stats, sig=np.array(json.dumps(self.sig)))

PAIRS = PairCache(os.path.join(CACHE_DIR, "sift_pair_cache.npz"))

def seed_from_legacy():
    """Import pairs from the old notebook's cache/sift_<kind>_top*.parquet|csv.gz if its SIFT parameters match ours."""
    mp = os.path.join(CACHE_DIR, "sift_meta.json")
    if not os.path.exists(mp): return
    meta, added = json.load(open(mp)), 0
    for kind, s in meta.items():
        if not (isinstance(s, dict) and s.get("ratio") == SIFT_RATIO and s.get("ransac") == RANSAC_THRESH and s.get("iters") == RANSAC_ITERS): continue
        for p in sorted(glob.glob(os.path.join(CACHE_DIR, f"sift_{kind}_top*.parquet")) + glob.glob(os.path.join(CACHE_DIR, f"sift_{kind}_top*.csv.gz"))):
            try: df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            except Exception as e: print("  could not read", p, type(e).__name__); continue
            qi, gi = _IDX.get_indexer(df.query_id.to_numpy(dtype=object)), _IDX.get_indexer(df.gallery_id.to_numpy(dtype=object)); ok = (qi >= 0) & (gi >= 0)
            PAIRS.add(qi[ok].astype(np.int64) * NID + gi[ok], df[["n_kp_q", "n_kp_g", "ratio_matches", "ransac_inliers"]].to_numpy(np.int32)[ok]); added += int(ok.sum()); break
    if added: print(f"imported {added:,} pairs from the legacy cache"); PAIRS.save()
seed_from_legacy()

def sift_stats(qids, gid_mat, label):
    """qids (nq,), gid_mat (nq, M) -> int32 (nq*M, 4). Only pairs missing from the cache are computed."""
    nq, M = gid_mat.shape
    q_rep, g_flat = np.repeat(np.asarray(qids, dtype=object), M), gid_mat.ravel()
    keys = pair_key(q_rep, g_flat); pos = PAIRS.lookup(keys); miss = np.flatnonzero(pos < 0)
    print(f"[{label}] pairs: {len(keys):,} | cached: {len(keys) - len(miss):,} | to compute: {len(miss):,}")
    if len(miss):
        grp = pd.DataFrame({"q": q_rep[miss], "g": g_flat[miss]}).groupby("q", sort=False)["g"].agg(list)
        q_list, g_lists = list(grp.index), list(grp.values)
        SIFT = load_sift_for(list(dict.fromkeys(q_list + [g for gl in g_lists for g in gl])))
        st = run_sift(SIFT, q_list, g_lists, label)
        PAIRS.add(pair_key(np.repeat(np.array(q_list, dtype=object), [len(l) for l in g_lists]), np.array([g for gl in g_lists for g in gl], dtype=object)), st)
        PAIRS.save(); pos = PAIRS.lookup(keys); assert (pos >= 0).all()
    return PAIRS.stats[pos]

def topk_sorted(S, K):
    part = np.argpartition(-S, K - 1, axis=1)[:, :K]
    return np.take_along_axis(part, np.argsort(-np.take_along_axis(S, part, 1), axis=1), 1)

def centroid_sims(Q, G, gcode, gc):
    u, inv = np.unique(gcode, return_inverse=True)
    M = sparse.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))), shape=(len(u), len(inv)))
    C = l2n(np.asarray(M @ G, dtype=np.float32)); S = Q @ C.T
    return np.take_along_axis(S, np.searchsorted(u, gc), 1)

def merge_candidates(c_sr, c_l, M, ML, K):
    out = np.empty((len(c_sr), K), c_sr.dtype)
    for i in range(len(c_sr)):
        head = c_sr[i, :M]; seen = set(head.tolist())
        extra = [x for x in c_l[i].tolist() if x not in seen][:ML]; seen.update(extra)
        tail = [x for x in c_sr[i, M:].tolist() if x not in seen]
        out[i] = np.concatenate([head, np.asarray(extra, c_sr.dtype), np.asarray(tail, c_sr.dtype)])[:K]
    return out

N_INJ_ID, N_INJ_PER = 12, 2     # identity-level injection: best N_INJ_PER images of the N_INJ_ID closest identities (learned space)
def learned_candidates(Slf, Lq, Lg, gcode, K):
    u, inv = np.unique(gcode, return_inverse=True)
    Mx = sparse.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))), shape=(len(u), len(inv)))
    C = l2n(np.asarray(Mx @ Lg, dtype=np.float32))
    top_id = topk_sorted(Lq @ C.T, min(N_INJ_ID, len(u)))
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    base = topk_sorted(Slf, K); out = np.empty_like(base)
    for i in range(len(base)):
        inj = []
        for j in top_id[i]:
            m = members[j]; inj.extend(m[np.argsort(-Slf[i, m])[:N_INJ_PER]].tolist())
        seen = set(inj); out[i] = (inj + [x for x in base[i].tolist() if x not in seen])[:K]
    return out

def build_pool(sp, q_pos, q_ids, vq, Pq, Rq, g_rows, Pg, Rg, Lq, Lg):
    """q_pos: row positions of the queries in their own frame; g_rows: gallery row positions in `tr`."""
    K = CAND_K; assert len(g_rows) >= K, f"gallery has {len(g_rows)} images; lower CAND_K"
    nq = len(q_pos)
    Sr, Sc = k_reciprocal_rerank(Pq, Pg)
    cand = topk_sorted(Sr, K); Slf = Lq @ Lg.T
    if CAND_LDA > 0: cand = merge_candidates(cand, learned_candidates(Slf, Lq, Lg, code_tr[g_rows], K), TOP_M_SIFT, CAND_LDA, K)
    gcode = code_tr[g_rows]; gcnt = np.bincount(gcode, minlength=len(CLASSES)); vg = vp_id_tr[g_rows]
    Rs = Rq @ Rg.T; rc = topk_sorted(Rs, K)                       # V1: plain cosine on RAW descriptors
    qq = Pq @ Pq.T; np.fill_diagonal(qq, -1.0)
    gc = gcode[cand].astype(np.int32)
    Sl = np.take_along_axis(Slf, cand, 1)
    kn = min(KN, nq - 1); nn = np.zeros((nq, KN), np.int32); nn_sim = np.full((nq, KN), -1.0, np.float32)
    if kn > 0:
        t = topk_sorted(qq, kn); nn[:, :kn] = t; nn_sim[:, :kn] = np.take_along_axis(qq, t, 1)
    return dict(Sr=np.take_along_axis(Sr, cand, 1), Sc=np.take_along_axis(Sc, cand, 1), Sl=Sl,
                cen=centroid_sims(Pq, Pg, gcode, gc), cenl=centroid_sims(Lq, Lg, gcode, gc),
                bk=VP_TABLE[vq[:, None], vg[cand]], gc=gc, gn=gcnt[gc].astype(np.float32), cand=g_rows[cand].astype(np.int32),
                q_rows=np.asarray(q_pos), qid=np.asarray(q_ids, dtype=object), sp=np.full(nq, SP_ID[sp], np.int8), q_vp=vq,
                q2q1=qq.max(1), q2q3=np.sort(qq, axis=1)[:, -3:].mean(1), nn=nn, nn_sim=nn_sim,
                r_sc=np.take_along_axis(Rs, rc, 1), r_gc=gcode[rc].astype(np.int32))

def concat_pools(parts):
    offs = np.cumsum([0] + [len(p["qid"]) for p in parts[:-1]]); out = {}
    for k in parts[0]:
        out[k] = np.concatenate([p[k] + o for p, o in zip(parts, offs)]) if k == "nn" else np.concatenate([p[k] for p in parts])
    return out

tr_ids = tr.id.to_numpy(dtype=object); te_ids = te.id.to_numpy(dtype=object)
POOL, TCODE = {}, {}
for scheme in ALLS:
    fold_s, unseen_s = FOLDS[scheme]; parts = []
    for sp in SPECIES:
        rows = SUB[sp]; P_tr, _ = EMB[sp]; R_tr, _ = RAW[sp]; f = fold_s[rows]
        for k in range(N_FOLDS):
            q_loc, g_loc = np.flatnonzero(f == k), np.flatnonzero(f != k)
            if len(q_loc) == 0: continue
            mu, W = fit_proj(P_tr[g_loc], code_tr[rows[g_loc]], ALPHA[sp])
            pool = build_pool(sp, rows[q_loc], tr_ids[rows[q_loc]], vp_id_tr[rows[q_loc]], P_tr[q_loc], R_tr[q_loc], rows[g_loc], P_tr[g_loc], R_tr[g_loc],
                              apply_proj(P_tr[q_loc], mu, W), apply_proj(P_tr[g_loc], mu, W))
            pool["fold"] = np.full(len(q_loc), k, np.int8); parts.append(pool)
            log(f"  CV[{scheme}] {sp} fold {k}: queries={len(q_loc)} gallery={len(g_loc)}")
    POOL[scheme] = concat_pools(parts)
    TCODE[scheme] = np.where(unseen_s[POOL[scheme]["q_rows"]], -1, code_tr[POOL[scheme]["q_rows"]])      # -1 = truth is new_individual
    log(f"CV[{scheme}] queries: {len(TCODE[scheme])} | new share {100 * (TCODE[scheme] < 0).mean():.1f}%")

# ---------- test pool (gallery = full train) ----------
parts = []
for sp in SPECIES:
    P_tr, P_te = EMB[sp]; R_tr, R_te = RAW[sp]; qr = TE_SUB[sp]
    mu, W = fit_proj(P_tr, code_tr[SUB[sp]], ALPHA[sp])
    parts.append(build_pool(sp, qr, te_ids[qr], vp_id_te[qr], P_te, R_te, SUB[sp], P_tr, R_tr, apply_proj(P_te, mu, W), apply_proj(P_tr, mu, W)))
    log(f"  TEST {sp}: queries={len(qr)} gallery={len(SUB[sp])}")
POOL["test"] = concat_pools(parts)
print("queries per CV fold vs test:", {s: len(TCODE[s]) // N_FOLDS for s in ALLS}, "vs", len(te))

def attach_sift(P, label):
    nq, K = P["Sr"].shape; M = min(TOP_M_EFF, K)
    st = sift_stats(P["qid"], tr_ids[P["cand"][:, :M]], label).reshape(nq, M, 4)
    for j, name in enumerate(["kq", "kg", "nm", "inl"]):
        a = np.zeros((nq, K), np.float32); a[:, :M] = st[:, :, j]; P[name] = a
    st = sift_stats(P["qid"], P["qid"][P["nn"]], label + "-q2q").reshape(nq, KN, 4)
    inl = st[:, :, 3].astype(np.float32); inl[P["nn_sim"] < -0.5] = 0; P["nn_inl"] = inl

for name in ("test",) + ALLS: attach_sift(POOL[name], name)
for sch in ALLS:
    P, tcode = POOL[sch], TCODE[sch]
    same = (P["gc"][:, :TOP_M_EFF] == tcode[:, None])
    print(f"[{sch}] share of top-M pairs with 0 inliers: %.1f%%" % (100 * (P["inl"][:, :TOP_M_EFF] == 0).mean()))
    print(f"[{sch}] mean inliers  same-identity: %.1f | different-identity: %.1f" % (P["inl"][:, :TOP_M_EFF][same].mean(), P["inl"][:, :TOP_M_EFF][~same].mean()))

def _q(v): return np.round(np.quantile(v, [.1, .5, .9]), 3) if len(v) else np.array([np.nan] * 3)

def shift_report():
    lines = ["Shift report: quantiles (10/50/90 %). KS = distance of the top-1 cosine to the test distribution."]
    for i, sp in enumerate(SPECIES):
        t_top1 = POOL["test"]["Sc"].max(1)[POOL["test"]["sp"] == i]
        for name in ("test",) + ALLS:
            P = POOL[name]; m = P["sp"] == i
            groups = [("all", m)] if name == "test" else [("all", m), ("known", m & (TCODE[name] >= 0)), ("new", m & (TCODE[name] < 0))]
            for lab, mm in groups:
                lines.append(f"{sp:8s} {name:6s} {lab:5s} n={int(mm.sum()):5d} | top1 cos {_q(P['Sc'].max(1)[mm])} | q2q1 {_q(P['q2q1'][mm])} | q2q3 {_q(P['q2q3'][mm])} | top1 learned {_q(P['Sl'].max(1)[mm])}")
            if name != "test":
                lines.append(f"{sp:8s} {name:6s} KS(top-1 cosine, all queries vs test) = {ks_2samp(P['Sc'].max(1)[m], t_top1).statistic:.3f}   | mean pool size per fold {int(m.sum() / N_FOLDS)} vs test {len(t_top1)}")
        lines.append("")
    return "\n".join(lines)
SHIFT_TXT = shift_report(); print(SHIFT_TXT)

# ---- estimate the test share of new individuals (per species) ----
def estimate_new_share(i):
    t = np.sort(POOL["test"]["Sc"].max(1)[POOL["test"]["sp"] == i])
    kn = np.sort(np.concatenate([POOL[s]["Sc"].max(1)[(POOL[s]["sp"] == i) & (TCODE[s] >= 0)] for s in TRS]))
    nw = np.sort(np.concatenate([POOL[s]["Sc"].max(1)[(POOL[s]["sp"] == i) & (TCODE[s] < 0)] for s in TRS]))
    grid = np.unique(np.quantile(t, np.linspace(0, 1, 301)))
    F = lambda a: np.searchsorted(a, grid, side="right") / len(a)
    Ft, Fk, Fn = F(t), F(kn), F(nw)
    pis = np.linspace(0, 1, 101); ks = np.array([np.abs(Ft - ((1 - p) * Fk + p * Fn)).max() for p in pis])
    band = pis[ks <= ks.min() + 0.01]
    return float(pis[ks.argmin()]), float(band.min()), float(band.max()), float(ks.min())

PI_CV = {sp: float(np.mean([(TCODE[s][POOL[s]["sp"] == i] < 0).mean() for s in TRS])) for i, sp in enumerate(SPECIES)}
PI_FIT_MAX_KS = 0.12      # if even the best mixture is further than this from the test distribution, the estimate is not trusted
def decide_pi_test():
    out = {}
    for i, sp in enumerate(SPECIES):
        hat, lo, hi, kmin = estimate_new_share(i); fit_ok = kmin <= PI_FIT_MAX_KS
        auto = float(np.clip(PI_CV[sp], lo, hi)) if fit_ok else PI_CV[sp]      # auto: move only if the CV prior lies outside the plausible band, only to its edge, and only if the mixture fits
        use = {"off": PI_CV[sp], "force": hat}.get(PRIOR_SHIFT, auto)
        out[sp] = dict(cv=PI_CV[sp], hat=hat, lo=lo, hi=hi, ks=kmin, fit_ok=fit_ok, use=use)
        if not fit_ok: log(f"  WARNING {sp}: mixture fit to the test distribution is poor (KS {kmin:.3f} > {PI_FIT_MAX_KS}); the new-share estimate is not reliable" + ("" if PRIOR_SHIFT != "force" else " (applied anyway: PRIOR_SHIFT=force)"))
    return out
PIS = decide_pi_test()
print("new-individual share: CV prior vs estimate from the test similarity distribution (PRIOR_SHIFT=%s)" % PRIOR_SHIFT)
print(pd.DataFrame(PIS).T.round(3).to_string())
open(os.path.join(OUT_DIR, "shift_diagnostics.txt"), "w").write(SHIFT_TXT + "\n\n" + pd.DataFrame(PIS).T.round(3).to_string())

def prior_adjust(sc, pnew, sp_arr):
    """Prior-shift correction: P(new) odds and identity probabilities are rescaled from the CV prior to the test prior."""
    sc2, p2 = sc.copy(), pnew.copy()
    for i, sp in enumerate(SPECIES):
        a, b = PIS[sp]["cv"], PIS[sp]["use"]
        if abs(a - b) < 1e-9: continue
        m = sp_arr == i
        sc2[m] = np.where(sc[m] > NEG / 2, np.clip(sc[m] * (1 - b) / (1 - a), 0, 1), sc[m])
        odds = pnew[m] / np.maximum(1 - pnew[m], 1e-9) * (b / (1 - b)) / (a / (1 - a)); p2[m] = odds / (1 + odds)
    return sc2, p2

def agg_topk(sc, gc, agg="max", topn=5):
    nq, K = sc.shape
    k = 1 if agg == "max" else int(agg[3:])
    o1 = np.argsort(-sc, axis=1)
    gc1 = np.take_along_axis(gc, o1, 1)
    o2 = np.argsort(gc1, axis=1, kind="stable")
    order = np.take_along_axis(o1, o2, 1)
    A, V = np.take_along_axis(gc, order, 1), np.take_along_axis(sc, order, 1)
    acc, cnt = V.astype(np.float32).copy(), np.ones((nq, K), np.float32)
    for j in range(1, k):
        same = A[:, j:] == A[:, :K - j]
        acc[:, :K - j] += np.where(same, V[:, j:], 0); cnt[:, :K - j] += same
    first = np.ones((nq, K), bool); first[:, 1:] = A[:, 1:] != A[:, :-1]
    gs = np.where(first, acc / cnt, NEG).astype(np.float32)
    kk = min(topn, K)
    part = np.argpartition(-gs, kk - 1, axis=1)[:, :kk]
    sel = np.take_along_axis(part, np.argsort(-np.take_along_axis(gs, part, 1), axis=1), 1)
    ids, scs = np.take_along_axis(A, sel, 1), np.take_along_axis(gs, sel, 1)
    return np.where(scs > NEG / 2, ids, -2).astype(np.int32), scs

def blend(P, p):
    btab = np.array([p["b_same"], p["b_adj"], p["b_opp"], 0.0, 0.0], np.float32)   # same, adjacent, opposite, missing, other
    lsift = np.log1p(np.maximum(P["inl"] - p["floor"], 0))
    return p["w_rerank"] * P["Sr"] + (1 - p["w_rerank"]) * P["Sc"] + p["w_sift"] * lsift + btab[P["bk"]]

def rank_of_truth(top, tcode):
    hit = (top == tcode[:, None]) & (tcode[:, None] >= 0)
    return np.where(hit.any(1), hit.argmax(1), 99)

def ap_given_pos(r, is_new, pos):
    """r: 0-based index of the true identity in the identity list (99 = absent); pos: index where NEW is inserted."""
    rank = np.where(r >= pos, r + 2, r + 1)
    return np.where(is_new, 1.0 / (pos + 1), np.where(rank <= 5, 1.0 / rank, 0.0))

def new_pos(sc, tau_q): return np.minimum((sc >= np.asarray(tau_q)[:, None]).sum(1), 4)

def per_query_ap(top, sc, tau_q, tcode):
    r = rank_of_truth(top, tcode); pos = new_pos(sc, tau_q)
    return ap_given_pos(r, tcode < 0, pos), r, pos

def tune_tau(top, sc, tcode, mask, n_grid=150):
    r, isnew, s = rank_of_truth(top[mask], tcode[mask]), tcode[mask] < 0, sc[mask]
    grid = np.unique(np.quantile(s[s > NEG / 2], np.linspace(0, 1, n_grid)))
    grid = np.r_[grid[0] - 1e-6, grid, grid[-1] + 1e-6]
    aps = np.array([ap_given_pos(r, isnew, np.minimum((s >= t).sum(1), 4)).mean() for t in grid])
    best = np.flatnonzero(aps >= aps.max() - 1e-9)
    return float(grid[best[len(best) // 2]])

def make_lists(top, sc, tau_q):
    pos, out = new_pos(sc, tau_q), []
    for row, p in zip(top, pos):
        names = [CLASSES[c] for c in row if c >= 0]
        names.insert(int(p), NEW); out.append(" ".join(names[:5]))
    return out

def crossfit_tau(top, sc, tcode, sp_arr, fold_arr):
    """tau per query: tuned on the OTHER folds of the same scheme and species, applied to the held-out fold. Also returns tau tuned on everything (used at test time)."""
    taus = np.array([tune_tau(top, sc, tcode, sp_arr == i) for i in range(len(SPECIES))])
    tau_q = taus[sp_arr].astype(np.float32).copy()
    for i in range(len(SPECIES)):
        for k in np.unique(fold_arr):
            m = (sp_arr == i) & (fold_arr == k); o = (sp_arr == i) & (fold_arr != k)
            if m.any() and o.sum() > 20: tau_q[m] = tune_tau(top, sc, tcode, o)
    return tau_q, taus

RESULTS = []
def evaluate(name, scheme, top, sc, tau_q=None, verbose=True):
    """tau_q None -> cross-fitted per-species tau (V1/V2/F). Otherwise per-query threshold (learned P(new))."""
    P, tcode = POOL[scheme], TCODE[scheme]; sp_arr = P["sp"].astype(int)
    taus, ap_in = None, np.nan
    if tau_q is None:
        tau_q, taus = crossfit_tau(top, sc, tcode, sp_arr, P["fold"].astype(int))
        ap_in = per_query_ap(top, sc, taus[sp_arr], tcode)[0].mean()           # in-sample tau (the old, optimistic way) - logged only
    ap, r, pos = per_query_ap(top, sc, tau_q, tcode); new = tcode < 0
    row = {"version": name, "scheme": scheme, "MAP@5": ap.mean(), "MAP@5 (in-sample tau)": ap_in}
    for i, sp in enumerate(SPECIES): row[sp] = ap[sp_arr == i].mean()
    row.update({"known_queries": ap[~new].mean(), "new_queries": ap[new].mean(), "new@1_rate": float((pos == 0).mean())})
    RESULTS.append(row)
    if verbose: log(f"  {name:38s} [{scheme:6s}] MAP@5 {row['MAP@5']:.4f}" + ("" if np.isnan(ap_in) else f" (in-sample tau {ap_in:.4f})") + f" | hyena {row['hyena']:.4f} leopard {row['leopard']:.4f} | known {row['known_queries']:.4f} new {row['new_queries']:.4f}")
    return dict(ap=ap, r=r, pos=pos, taus=taus, tau_q=tau_q)

no_sift = dict(BEST_PARAMS, w_sift=0.0, b_same=0.0, b_adj=0.0, b_opp=0.0)
RES_A, RES_B, RES_C, RES_D, RES_E = {}, {}, {}, {}, {}
for sch in ALLS:
    P = POOL[sch]
    topA, scA = agg_topk(P["r_sc"], P["r_gc"], "max")
    RES_A[sch] = evaluate("A  raw cosine  [V1 baseline]", sch, topA, scA)
    sB = blend(P, dict(no_sift, w_rerank=0.0)); topB, scB = agg_topk(sB, P["gc"], "max")
    RES_B[sch] = evaluate("B  + centre / PCA-whiten", sch, topB, scB)
    sC = blend(P, dict(no_sift)); topC, scC = agg_topk(sC, P["gc"], "max")
    RES_C[sch] = evaluate("C  + k-reciprocal re-ranking", sch, topC, scC)
    topD, scD = agg_topk(sC, P["gc"], BEST_PARAMS["agg"])
    RES_D[sch] = evaluate("D  + top-2 identity aggregation", sch, topD, scD)
    sE = blend(P, BEST_PARAMS); topE, scE = agg_topk(sE, P["gc"], BEST_PARAMS["agg"])
    RES_E[sch] = evaluate("E  + SIFT/RANSAC + viewpoint  [V2]", sch, topE, scE)
    print(f"[{sch}] per-species tau  V1:", dict(zip(SPECIES, RES_A[sch]["taus"].round(4))), "| V2:", dict(zip(SPECIES, RES_E[sch]["taus"].round(4))))

def identity_table(P, tcode=None, params=BEST_PARAMS, M=TOP_M_EFF):
    """One row per (query, candidate identity) with aggregated evidence."""
    nq = len(P["Sr"]); M = min(M, P["Sr"].shape[1])
    v2 = blend(P, params)[:, :M]
    df = pd.DataFrame({"qi": np.repeat(np.arange(nq), M), "code": P["gc"][:, :M].ravel(), "sr": P["Sr"][:, :M].ravel(),
                       "sc": P["Sc"][:, :M].ravel(), "sl": P["Sl"][:, :M].ravel(), "cen": P["cen"][:, :M].ravel(), "cenl": P["cenl"][:, :M].ravel(),
                       "inl": P["inl"][:, :M].ravel(), "nm": P["nm"][:, :M].ravel(), "v2": v2.ravel(),
                       "rk": np.tile(np.arange(M), nq), "gn": P["gn"][:, :M].ravel(), "bk": P["bk"][:, :M].ravel().astype(np.int16), "one": 1,
                       "kq": P["kq"][:, :M].ravel(), "kg": P["kg"][:, :M].ravel()})
    df["frac"] = df.inl / np.maximum(np.minimum(df.kq, df.kg), 1); df["ratio"] = df.inl / np.maximum(df.nm, 1); df["strong"] = (df.inl >= 10).astype(np.int16)
    df = df.sort_values(["qi", "code", "v2"], ascending=[True, True, False], kind="stable")     # best image of each identity first
    T = df.groupby(["qi", "code"], sort=False).agg(sr_max=("sr", "max"), sc_max=("sc", "max"), sl_max=("sl", "max"), cen=("cen", "first"), cenl=("cenl", "first"),
            v2_max=("v2", "max"), inl_max=("inl", "max"), inl_sum=("inl", "sum"), n_strong=("strong", "sum"), ratio_max=("ratio", "max"),
            n_img=("one", "sum"), best_rk=("rk", "min"), gn=("gn", "first"), bk=("bk", "first"), frac_max=("frac", "max"))
    top2 = df.groupby(["qi", "code"], sort=False).head(2).groupby(["qi", "code"], sort=False)
    T["v2_top2"], T["sr_top2"], T["sc_top2"], T["sl_top2"] = top2["v2"].mean(), top2["sr"].mean(), top2["sc"].mean(), top2["sl"].mean()
    T["inl_top2"] = top2["inl"].mean()
    T = T.reset_index()
    gq = T.groupby("qi")
    for c, nm in [("v2_top2", "q_v2_max"), ("sr_top2", "q_sr_max"), ("inl_max", "q_inl_max"), ("sc_max", "q_sc_max"), ("sl_max", "q_sl_max"), ("cenl", "q_cenl_max"), ("cen", "q_cen_max"), ("frac_max", "q_frac_max")]:
        T[nm] = gq[c].transform("max"); T[c + "_rel"] = T[c] - T[nm]
    T["v2_rank"] = gq["v2_top2"].rank(ascending=False, method="first"); T["sl_rank"] = gq["sl_max"].rank(ascending=False, method="first")
    T["cenl_rank"] = gq["cenl"].rank(ascending=False, method="first")
    T["log_gn"] = np.log1p(T.gn); T["has_strong"] = (T.n_strong > 0).astype(np.int16)
    qi = T.qi.to_numpy()
    T["sp"], T["q_vp"], T["kq0"] = P["sp"][qi], P["q_vp"][qi], P["kq"][qi, 0]
    T = T[(T.v2_rank <= TOP_ID) | (T.sl_rank <= TOP_L) | (T.cenl_rank <= 3)].reset_index(drop=True)
    if tcode is not None: T["y"] = (T.code.to_numpy() == tcode[T.qi.to_numpy()]).astype(np.int8)
    return T

ID_FEATS_V3 = ["sr_max", "sr_top2", "sc_max", "sc_top2", "v2_max", "v2_top2", "inl_max", "inl_sum", "n_strong", "ratio_max", "n_img", "best_rk",
               "log_gn", "bk", "q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "v2_top2_rel", "sr_top2_rel", "inl_max_rel", "sc_max_rel",
               "v2_rank", "sp", "q_vp", "kq0"]
ID_FEATS = ["sr_max", "sr_top2", "sc_max", "sc_top2", "sl_max", "sl_top2", "cen", "cenl", "v2_max", "v2_top2", "inl_max", "inl_sum", "n_strong", "ratio_max", "n_img", "best_rk",
            "log_gn", "bk", "q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "q_sl_max", "q_cenl_max", "q_cen_max", "v2_top2_rel", "sr_top2_rel", "inl_max_rel", "sc_max_rel",
            "sl_max_rel", "cenl_rel", "cen_rel", "v2_rank", "sl_rank", "cenl_rank", "sp", "q_vp", "kq0",
            "frac_max", "inl_top2", "q_frac_max", "frac_max_rel"]
NN_Q = ["nn_sim1", "nn_sim_mean", "nn_inl_max", "nn_n_strong"]
ID_FEATS2 = ID_FEATS + ["p_s1", "p_s1_rel", "p_s1_rank", "nb_pmax", "nb_pstrong", "nb_wp"] + NN_Q

def top_from_table(T, nq, col="p", n=5):
    S = T[["qi", "code", col]].sort_values(["qi", col], ascending=[True, False], kind="stable")
    r = S.groupby("qi").cumcount().to_numpy(); keep = r < n
    top, sc = np.full((nq, n), -2, np.int32), np.full((nq, n), NEG, np.float32)
    top[S.qi.to_numpy()[keep], r[keep]] = S.code.to_numpy()[keep]; sc[S.qi.to_numpy()[keep], r[keep]] = S[col].to_numpy()[keep]
    return top, sc

NEW_FEATS_V3 = ["p1", "p2", "p3", "p_sum", "p_gap", "n_id", "n_strong_ids", "q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "sp", "q_vp", "kq0"]
NEW_FEATS = ["p1", "p2", "p3", "p_sum", "p_gap", "n_id", "n_strong_ids", "q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "q_sl_max", "q_cenl_max", "sp", "q_vp", "kq0"]
NEW_FEATS2 = NEW_FEATS + ["q2q1", "q2q3", "s1_p1", "s1_pnew"] + NN_Q

def query_table(P, T, col="p"):
    nq = len(P["qid"]); _, s3 = top_from_table(T, nq, col, 3); s3 = np.where(s3 > NEG / 2, s3, 0.0)
    g = T.groupby("qi"); ix = np.arange(nq)
    Q = pd.DataFrame({"p1": s3[:, 0], "p2": s3[:, 1], "p3": s3[:, 2], "p_sum": g[col].sum().reindex(ix, fill_value=0).to_numpy(),
                      "n_id": g[col].size().reindex(ix, fill_value=0).to_numpy(), "n_strong_ids": g["has_strong"].sum().reindex(ix, fill_value=0).to_numpy()})
    Q["p_gap"] = Q.p1 - Q.p2
    for c in ["q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "q_sl_max", "q_cenl_max"]: Q[c] = g[c].first().reindex(ix).to_numpy()
    Q["sp"], Q["q_vp"], Q["kq0"], Q["q2q1"], Q["q2q3"] = P["sp"], P["q_vp"], P["kq"][:, 0], P["q2q1"], P["q2q3"]
    Q["nn_sim1"], Q["nn_sim_mean"] = P["nn_sim"][:, 0], P["nn_sim"].mean(1)
    Q["nn_inl_max"], Q["nn_n_strong"] = P["nn_inl"].max(1), (P["nn_inl"] >= NN_STRONG).sum(1)
    return Q

def add_neighbour_feats(P, T, col="p"):
    nq = len(P["qid"])
    D = pd.DataFrame({"qi": np.repeat(np.arange(nq), KN), "j": P["nn"].ravel(), "sim": P["nn_sim"].ravel(), "inl": P["nn_inl"].ravel()}); D = D[D.sim > -0.5]
    D["sc"] = D.sim.clip(lower=0); ssum = D.groupby("qi").sc.sum()
    M = D.merge(T[["qi", "code", col]].rename(columns={"qi": "j", col: "pj"}), on="j")
    M["pj_strong"] = M.pj * (M.inl >= NN_STRONG); M["wp"] = M.sc * M.pj
    A = M.groupby(["qi", "code"]).agg(nb_pmax=("pj", "max"), nb_pstrong=("pj_strong", "max"), nb_wp=("wp", "sum")).reset_index()
    A["nb_wp"] = A.nb_wp / np.maximum(A.qi.map(ssum).fillna(0).to_numpy(), 1e-6)
    n0 = len(T); T = T.merge(A, on=["qi", "code"], how="left"); assert len(T) == n0
    for c in ["nb_pmax", "nb_pstrong", "nb_wp"]: T[c] = T[c].fillna(0.0)
    T["p_s1"] = T[col]; T["p_s1_rel"] = T.p_s1 - T.groupby("qi").p_s1.transform("max"); T["p_s1_rank"] = T.groupby("qi").p_s1.rank(ascending=False, method="first")
    qi = T.qi.to_numpy(); T["nn_sim1"], T["nn_sim_mean"] = P["nn_sim"][qi, 0], P["nn_sim"].mean(1)[qi]
    T["nn_inl_max"], T["nn_n_strong"] = P["nn_inl"].max(1)[qi], (P["nn_inl"] >= NN_STRONG).sum(1)[qi]
    return T

CAT_FEATS = {"sp", "q_vp", "bk"}                     # categorical, not ordinal
def cat_idx(feats): return [i for i, f in enumerate(feats) if f in CAT_FEATS]

def fit_hgb(X, y, feats):
    ci = cat_idx(feats)
    return HistGradientBoostingClassifier(learning_rate=0.08, max_iter=200, max_leaf_nodes=24, min_samples_leaf=40, l2_regularization=1.0,
                                          categorical_features=ci or None, early_stopping=False, random_state=SEED).fit(X, y)

try:
    import lightgbm as lgb
except Exception:
    lgb = None

class TreeEnsemble:
    CFGS = [dict(learning_rate=0.08, max_iter=200, max_leaf_nodes=24, min_samples_leaf=40, l2_regularization=1.0),
            dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=12, min_samples_leaf=80, l2_regularization=3.0),
            dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=48, min_samples_leaf=30, l2_regularization=1.0)]
    def __init__(self, feats): self.cat = cat_idx(feats)
    def fit(self, X, y):
        cfgs = self.CFGS[:1] if FAST else self.CFGS
        self.models = [HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, categorical_features=self.cat or None, **c).fit(X, y) for c in cfgs]
        if lgb is not None and not FAST:
            for s in range(2):
                m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.7, reg_lambda=2.0, random_state=SEED + s, verbose=-1, n_jobs=-1)
                self.models.append(m.fit(X, y, categorical_feature=self.cat) if self.cat else m.fit(X, y))
        return self
    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.c_[1 - p, p]
def fit_model(X, y, feats): return TreeEnsemble(feats).fit(X, y)

def cv_predict(X, y, folds, feats, Xe=None, fe=None):
    """Out-of-fold predictions for the training rows (X) and, with the SAME fold models, predictions for evaluation-only rows (Xe,fe)
    whose model fold is fe. A row is only ever predicted by a model that never saw its identity."""
    oof = np.zeros(len(X), np.float32); pe = None if Xe is None else np.zeros(len(Xe), np.float32)
    for k in np.unique(folds):
        m = folds == k; mdl = fit_model(X[~m], y[~m], feats)
        oof[m] = mdl.predict_proba(X[m])[:, 1]
        if Xe is not None:
            me = fe == k
            if me.any(): pe[me] = mdl.predict_proba(Xe[me])[:, 1]
        log(f"    model fold {k} done")
    return oof, pe

def pooled(frames, cols): return np.vstack([f[cols].to_numpy(np.float32) for f in frames])
def scatter_back(vals, frames, col):
    o = 0
    for f in frames: f[col] = vals[o:o + len(f)]; o += len(f)

TB = {s: identity_table(POOL[s], TCODE[s]) for s in ALLS}
HAS = {}
for s in ALLS:
    HAS[s] = TB[s].groupby("qi").y.max().reindex(np.arange(len(TCODE[s])), fill_value=0).to_numpy().astype(bool)
    log(f"identity table [{s}]: {len(TB[s]):,} rows | positives: {int(TB[s].y.sum()):,} | retrieval recall ceiling for known queries: {100 * HAS[s][TCODE[s] >= 0].mean():.1f}%")

def pooled_opt(frames, cols): return pooled(frames, cols) if frames else None
def tb_mf(s): return MF_TR[POOL[s]["q_rows"]][TB[s].qi.to_numpy()]      # model fold of each identity-table row (= fold of the query's identity)
def q_mf(s):  return MF_TR[POOL[s]["q_rows"]]
def cvp(X, y, f, feats, Xe, fe, frames_tr, frames_ev, col):
    oof, pe = cv_predict(X, y, f, feats, Xe, fe)
    scatter_back(oof, frames_tr, col)
    if frames_ev: scatter_back(pe, frames_ev, col)

for s in ALLS:
    log(f"[{s}] candidate-identity recall ceiling: top-{TOP_ID} v2 / union {TB[s].groupby('qi').size().mean():.1f} identities per query | known queries with truth in the table: {HAS[s][TCODE[s] >= 0].mean():.3f}")

TBL_TR, TBL_EV = [TB[s] for s in TRS], [TB[s] for s in EVS]
y1  = np.concatenate([TB[s].y.to_numpy() for s in TRS]); mf1 = np.concatenate([tb_mf(s) for s in TRS]); mf1e = np.concatenate([tb_mf(s) for s in EVS]) if EVS else None
yq  = np.concatenate([(TCODE[s] < 0).astype(np.int8) for s in TRS]); mfq = np.concatenate([q_mf(s) for s in TRS]); mfqe = np.concatenate([q_mf(s) for s in EVS]) if EVS else None

log("V3: identity ranker (no neighbour features), out-of-fold ...")
X1_v3 = pooled(TBL_TR, ID_FEATS_V3); X1_v3e = pooled_opt(TBL_EV, ID_FEATS_V3)
cvp(X1_v3, y1, mf1, ID_FEATS_V3, X1_v3e, mf1e, TBL_TR, TBL_EV, "p_v3")
TOP_V3 = {s: top_from_table(TB[s], len(TCODE[s]), "p_v3", 5) for s in ALLS}
RES_F = {s: evaluate("F  V3a: learned ranker + tau", s, *TOP_V3[s]) for s in ALLS}

QT_V3 = {s: query_table(POOL[s], TB[s], "p_v3") for s in ALLS}
Xq_v3 = pooled([QT_V3[s] for s in TRS], NEW_FEATS_V3); Xq_v3e = pooled_opt([QT_V3[s] for s in EVS], NEW_FEATS_V3)
log("V3: learned P(new), out-of-fold ...")
cvp(Xq_v3, yq, mfq, NEW_FEATS_V3, Xq_v3e, mfqe, [QT_V3[s] for s in TRS], [QT_V3[s] for s in EVS], "p_new")
RES_G = {s: evaluate("G  V3b: + learned P(new)", s, *TOP_V3[s], tau_q=QT_V3[s].p_new.to_numpy()) for s in ALLS}

log("V4 stage 1: identity ranker with neighbour-free features + P(new) ...")
X1 = pooled(TBL_TR, ID_FEATS); X1e = pooled_opt(TBL_EV, ID_FEATS)
cvp(X1, y1, mf1, ID_FEATS, X1e, mf1e, TBL_TR, TBL_EV, "p")
TOP1 = {s: top_from_table(TB[s], len(TCODE[s]), "p", 5) for s in ALLS}
QT = {s: query_table(POOL[s], TB[s], "p") for s in ALLS}
Xq1 = pooled([QT[s] for s in TRS], NEW_FEATS); Xq1e = pooled_opt([QT[s] for s in EVS], NEW_FEATS)
cvp(Xq1, yq, mfq, NEW_FEATS, Xq1e, mfqe, [QT[s] for s in TRS], [QT[s] for s in EVS], "p_new")
RES_H = {s: evaluate("H  V4a: stage 1 (calibrated)", s, *TOP1[s], tau_q=QT[s].p_new.to_numpy()) for s in ALLS}

log("V4 stage 2: neighbour evidence from other test queries ...")
for s in ALLS:
    TB[s] = add_neighbour_feats(POOL[s], TB[s], "p")
    QT[s]["s1_p1"] = QT[s]["p1"]; QT[s]["s1_pnew"] = QT[s]["p_new"]
TBL_TR, TBL_EV = [TB[s] for s in TRS], [TB[s] for s in EVS]
X2 = pooled(TBL_TR, ID_FEATS2); X2e = pooled_opt(TBL_EV, ID_FEATS2)
y2 = np.concatenate([TB[s].y.to_numpy() for s in TRS]); mf2 = np.concatenate([tb_mf(s) for s in TRS]); mf2e = np.concatenate([tb_mf(s) for s in EVS]) if EVS else None
cvp(X2, y2, mf2, ID_FEATS2, X2e, mf2e, TBL_TR, TBL_EV, "p2")
QT2, TOP2 = {}, {}
for s in ALLS:
    TOP2[s] = top_from_table(TB[s], len(TCODE[s]), "p2", 5)
    QT2[s] = query_table(POOL[s], TB[s], "p2"); QT2[s]["s1_p1"] = QT[s]["p1"]; QT2[s]["s1_pnew"] = QT[s]["p_new"]
Xq2 = pooled([QT2[s] for s in TRS], NEW_FEATS2); Xq2e = pooled_opt([QT2[s] for s in EVS], NEW_FEATS2)
cvp(Xq2, yq, mfq, NEW_FEATS2, Xq2e, mfqe, [QT2[s] for s in TRS], [QT2[s] for s in EVS], "p_new")
RES_I = {s: evaluate("I  V4b: + neighbour evidence", s, *TOP2[s], tau_q=QT2[s].p_new.to_numpy()) for s in ALLS}

gains = {s: RES_I[s]["ap"].mean() - RES_H[s]["ap"].mean() for s in ALLS}
n_tr_q = sum(len(TCODE[s]) for s in TRS)
cv1 = sum(RES_H[s]["ap"].sum() for s in TRS) / n_tr_q; cv2 = sum(RES_I[s]["ap"].sum() for s in TRS) / n_tr_q
USE_S2 = all(gains[s] > 0.002 for s in TRS)                     # must help on EVERY test-like seed
log("stage-2 gain: " + " | ".join(f"{s} {gains[s]:+.4f}" for s in ALLS))
log(f"pooled test-like CV MAP@5: stage 1 {cv1:.4f} -> stage 2 {cv2:.4f} | submission uses {'stage 2' if USE_S2 else 'stage 1'}")
FIN_RES, FIN_TOP, FIN_PNEW = (RES_I, TOP2, {s: QT2[s].p_new.to_numpy() for s in ALLS}) if USE_S2 else (RES_H, TOP1, {s: QT[s].p_new.to_numpy() for s in ALLS})

RESULTS_DF = pd.DataFrame(RESULTS)
RESULTS_DF = pd.concat([RESULTS_DF[RESULTS_DF.scheme == s] for s in ALLS]).set_index(["scheme", "version"])
RESULTS_DF.to_csv(os.path.join(OUT_DIR, "results_cv.csv"))
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
display(RESULTS_DF.round(4))

SUMMARY = RESULTS_DF.loc[list(TRS)].groupby("version", sort=False)["MAP@5"].agg(["mean", "std"])
print("\nMAP@5 on the test-like folds, mean / std over %d seed(s):" % len(TRS)); print(SUMMARY.round(4).to_string())
if EVS:
    print("\nSTRESS (evaluation only, deliberately much harder than test - read as a lower bound, not an estimate):")
    print(RESULTS_DF.loc[EVS[0]]["MAP@5"].round(4).to_string())

import matplotlib.pyplot as plt
COL = {"A": "#999999", "B": "#999999", "C": "#999999", "D": "#999999", "E": "#4C72B0", "F": "#DD8452", "G": "#55A868", "H": "#C44E52", "I": "#8172B3"}
fig, axs = plt.subplots(1, 2 if EVS else 1, figsize=(15 if EVS else 9, 5.2), squeeze=False)
for ax, (title, vals, err) in zip(axs[0], [("test-like folds (mean ± std over seeds)", SUMMARY["mean"], SUMMARY["std"].fillna(0))] +
                                  ([("STRESS (evaluation only)", RESULTS_DF.loc[EVS[0]]["MAP@5"], None)] if EVS else [])):
    ax.barh(range(len(vals)), vals.values, xerr=None if err is None else err.values, color=[COL.get(v[0], "#999") for v in vals.index])
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(vals.index, fontsize=8); ax.invert_yaxis()
    lo = vals.min(); ax.set_xlim(max(0, lo - 0.04), min(1, vals.max() + 0.03)); ax.set_title(title); ax.set_xlabel("MAP@5")
    for i, v in enumerate(vals.values): ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "ablation.png"), dpi=130); plt.close("all")

for s in ALLS:
    lists_chk = make_lists(*FIN_TOP[s], FIN_PNEW[s])
    truth_names = np.where(TCODE[s] >= 0, CLASSES[np.maximum(TCODE[s], 0)], NEW)
    ref = map5([l.split() for l in lists_chk], truth_names)
    assert abs(ref - FIN_RES[s]["ap"].mean()) < 1e-9, (s, ref, FIN_RES[s]["ap"].mean())
    print(f"[{s}] reference MAP@5 on the written-format lists = {ref:.6f}  (matches the vectorised value)")

def sensitivity_table(scheme=None):
    scheme = scheme or PRIMARY; P, tcode = POOL[scheme], TCODE[scheme]; spa = P["sp"].astype(int)
    top, sc = FIN_TOP[scheme]; pn = FIN_PNEW[scheme]; rows = []
    for pi_ in (0.25, 0.30, 0.40, 0.50, 0.55):
        w = np.zeros(len(tcode))
        for i, sp in enumerate(SPECIES):
            m = spa == i; w[m] = np.where(tcode[m] < 0, pi_ / PI_CV[sp], (1 - pi_) / (1 - PI_CV[sp]))
        base = np.average(per_query_ap(top, sc, pn, tcode)[0], weights=w)
        saved = {k: dict(v) for k, v in PIS.items()}
        for sp in SPECIES: PIS[sp]["use"] = pi_
        sc2, pn2 = prior_adjust(sc, pn, spa)
        for sp in SPECIES: PIS[sp]["use"] = saved[sp]["use"]
        adj = np.average(per_query_ap(top, sc2, pn2, tcode)[0], weights=w)
        rows.append(dict(test_new_share=pi_, MAP_uncorrected=base, MAP_corrected=adj, gain=adj - base))
    return pd.DataFrame(rows).set_index("test_new_share")
SENS = sensitivity_table(); print(f"re-weighted {PRIMARY} queries (CV share {np.mean(list(PI_CV.values())):.2f}):"); print(SENS.round(4).to_string())
SENS.to_csv(os.path.join(OUT_DIR, "prior_shift_sensitivity.csv"))

def failure_taxonomy(sch):
    ap, r, pos = FIN_RES[sch]["ap"], FIN_RES[sch]["r"], FIN_RES[sch]["pos"]
    tcode, has_truth = TCODE[sch], HAS[sch]
    new_q = tcode < 0; rank_final = np.where(r >= pos, r + 2, r + 1)
    cat = np.empty(len(ap), dtype=object)
    cat[new_q & (pos == 0)] = "new: NEW ranked 1st (correct)"
    cat[new_q & (pos > 0)]  = "new: a known identity ranked above NEW (false match)"
    kn = ~new_q
    cat[kn & (rank_final == 1)]                        = "known: correct @1"
    cat[kn & (r == 0) & (rank_final == 2)]             = "known: right identity first, but NEW placed above it"
    cat[kn & (rank_final >= 2) & (rank_final <= 5) & ~((r == 0) & (rank_final == 2))] = "known: correct @2-5 (ranking error)"
    cat[kn & (rank_final > 5) & has_truth]             = "known: in candidates but outside top-5 (ranking error)"
    cat[kn & ~has_truth]                               = "known: true identity not retrieved (retrieval miss)"
    tax = pd.DataFrame({"queries": pd.Series(cat).value_counts(), "share": pd.Series(cat).value_counts(normalize=True).round(4)})
    tax["MAP lost"] = [ (1 - ap[cat == c]).sum() / len(ap) for c in tax.index]
    tax = tax.sort_values("MAP lost", ascending=False)
    print(f"[{sch}]"); print(tax.round(4).to_string())
    print("\ntotal MAP lost = %.4f  (1 - MAP@5)" % (1 - ap.mean()))
    fig, ax = plt.subplots(figsize=(9, 3.8)); ax.barh(range(len(tax)), tax["MAP lost"], color="#d9534f")
    ax.set_yticks(range(len(tax))); ax.set_yticklabels(tax.index, fontsize=8); ax.invert_yaxis(); ax.set_xlabel("contribution to (1 - MAP@5)")
    ax.set_title(f"Where the remaining error comes from (V4, {sch} CV)"); savefig(f"failure_taxonomy_{sch}.png")
for sch in (PRIMARY,) + EVS: failure_taxonomy(sch)

def slice_report(sch):
    ap, new_q, P = FIN_RES[sch]["ap"], TCODE[sch] < 0, POOL[sch]
    tmp = pd.DataFrame({"ident": tr.identity.to_numpy(dtype=object), "fold": FOLDS[sch][0]}); tot = tmp.groupby("ident").size(); infold = tmp.groupby(["ident", "fold"]).size()
    ident_q = tmp.ident.to_numpy()[P["q_rows"]]
    gal_size = tot.reindex(ident_q).to_numpy() - infold.reindex(pd.MultiIndex.from_arrays([ident_q, P["fold"].astype(np.int64)])).to_numpy()
    D_ = pd.DataFrame({"ap": ap, "new": new_q, "vp": np.array(VP_VOCAB)[P["q_vp"]], "kq0": P["kq"][:, 0], "gal": gal_size, "sp": np.array(SPECIES)[P["sp"]]})
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    g = D_.groupby("vp").ap.agg(["mean", "size"]).sort_values("size", ascending=False); axs[0].bar(g.index, g["mean"], color="#3b7dd8"); axs[0].set_title(f"MAP@5 by query viewpoint ({sch})"); axs[0].tick_params(axis="x", rotation=60)
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])): axs[0].text(i, m + 0.01, f"n={n}", ha="center", fontsize=7)
    K_ = D_[~D_.new].copy(); K_["bin"] = pd.cut(K_.gal, [0, 1, 2, 4, 8, 16, 10_000], labels=["1", "2", "3-4", "5-8", "9-16", ">16"])
    g2 = K_.groupby("bin", observed=True).ap.agg(["mean", "size"]); axs[1].bar(g2.index.astype(str), g2["mean"], color="#5aa469"); axs[1].set_title("known queries: MAP@5 by #gallery images of the true identity")
    for i, (m, n) in enumerate(zip(g2["mean"], g2["size"])): axs[1].text(i, m + 0.01, f"n={n}", ha="center", fontsize=7)
    D_["kbin"] = pd.qcut(D_.kq0.replace(0, np.nan), 5, duplicates="drop"); g3 = D_.groupby("kbin", observed=True).ap.mean()
    axs[2].bar([f"Q{i+1}" for i in range(len(g3))], g3.values, color="#e0a030"); axs[2].set_title("MAP@5 by query keypoint-count quintile (image quality proxy)")
    for a in axs: a.set_ylim(0, 1.05)
    savefig(f"slices_{sch}.png")
    print(f"[{sch}]"); print(D_.groupby(["sp", "new"]).ap.agg(["mean", "size"]).round(4))
for sch in (PRIMARY,) + EVS: slice_report(sch)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
P_h, t_h = POOL[PRIMARY], TCODE[PRIMARY]
Mx = min(TOP_M_EFF, P_h["inl"].shape[1]); same_m = (P_h["gc"][:, :Mx] == t_h[:, None]) & (t_h[:, None] >= 0)
li = np.log1p(P_h["inl"][:, :Mx]); kn_rows = (t_h >= 0)[:, None] & np.ones((1, Mx), bool)
axs[0].hist(li[kn_rows & ~same_m], bins=30, alpha=.6, density=True, label="different identity"); axs[0].hist(li[same_m], bins=30, alpha=.6, density=True, label="same identity")
axs[0].set_xlabel("log(1 + RANSAC inliers)"); axs[0].set_title(f"SIFT evidence separates same vs different ({PRIMARY})"); axs[0].legend()
axs[1].plot([0, 1], [0, 1], "k--", lw=1)
for s in ALLS:
    pn = FIN_PNEW[s]; ynew = (TCODE[s] < 0).astype(int); bins = np.quantile(pn, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(pn, bins[1:-1]), 0, 9)
    cal = pd.DataFrame({"pred": pn, "obs": ynew, "b": idx}).groupby("b").mean(); axs[1].plot(cal.pred, cal.obs, "o-" if s in TRS else "x--", label=s)
    print(f"[{s}] mean predicted P(new) on CV: %.3f | true new share: %.3f" % (pn.mean(), ynew.mean()))
axs[1].set_xlabel("predicted P(new)"); axs[1].set_ylabel("observed share new"); axs[1].set_title("Calibration of P(new_individual) (CV)"); axs[1].legend()
savefig("evidence_and_calibration.png")

# importance measured on HELD-OUT identities (model fold 0), not on the rows the model was fitted on
hold = mf1 == 0; rs = np.random.default_rng(SEED); ev_idx = np.flatnonzero(hold); samp = rs.choice(ev_idx, min(40000, len(ev_idx)), replace=False)
m_id_tmp = fit_hgb(X1[~hold], y1[~hold], ID_FEATS)
pi_imp = permutation_importance(m_id_tmp, X1[samp], y1[samp], scoring="roc_auc", n_repeats=3, random_state=SEED)
imp = pd.Series(pi_imp.importances_mean, index=ID_FEATS).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5)); imp.head(15)[::-1].plot.barh(ax=ax, color="#3b7dd8"); ax.set_xlabel("drop in AUC when shuffled (held-out identities)"); ax.set_title("Top features of the identity ranker"); savefig("feature_importance.png")
print(imp.head(10).round(4).to_string())

preds_by_version = {}
def to_series(lists, P):
    s = np.empty(len(te), dtype=object); s[P["q_rows"]] = lists; return s
PT = POOL["test"]; nT = len(PT["qid"]); spi = PT["sp"].astype(int)

# V1
top, sc = agg_topk(PT["r_sc"], PT["r_gc"], "max")
preds_by_version["v1"] = to_series(make_lists(top, sc, np.mean([RES_A[s]["taus"] for s in TRS], axis=0)[spi]), PT)
# V2
top, sc = agg_topk(blend(PT, BEST_PARAMS), PT["gc"], BEST_PARAMS["agg"])
preds_by_version["v2"] = to_series(make_lists(top, sc, np.mean([RES_E[s]["taus"] for s in TRS], axis=0)[spi]), PT)
# V3
m_id_v3 = fit_model(X1_v3, y1, ID_FEATS_V3); m_new_v3 = fit_model(Xq_v3, yq, NEW_FEATS_V3)
T_te = identity_table(PT); T_te["p_v3"] = m_id_v3.predict_proba(T_te[ID_FEATS_V3].to_numpy(np.float32))[:, 1]
top_te3, sc_te3 = top_from_table(T_te, nT, "p_v3", 5)
Q_te3 = query_table(PT, T_te, "p_v3"); pnew_te3 = m_new_v3.predict_proba(Q_te3[NEW_FEATS_V3].to_numpy(np.float32))[:, 1]
sc3a, pn3a = prior_adjust(sc_te3, pnew_te3, spi)
preds_by_version["v3"] = to_series(make_lists(top_te3, sc3a, pn3a), PT)
m_id1 = fit_model(X1, y1, ID_FEATS); m_new1 = fit_model(Xq1, yq, NEW_FEATS)
T_te["p"] = m_id1.predict_proba(T_te[ID_FEATS].to_numpy(np.float32))[:, 1]
top_te1, sc_te1 = top_from_table(T_te, nT, "p", 5)
Q_te = query_table(PT, T_te, "p"); pnew_te1 = m_new1.predict_proba(Q_te[NEW_FEATS].to_numpy(np.float32))[:, 1]
sc1a, pn1a = prior_adjust(sc_te1, pnew_te1, spi)
preds_by_version["v4_s1"] = to_series(make_lists(top_te1, sc1a, pn1a), PT)
T_te = add_neighbour_feats(PT, T_te, "p")
m_id2 = fit_model(X2, y2, ID_FEATS2); m_new2 = fit_model(Xq2, yq, NEW_FEATS2)
T_te["p2"] = m_id2.predict_proba(T_te[ID_FEATS2].to_numpy(np.float32))[:, 1]
top_te2, sc_te2 = top_from_table(T_te, nT, "p2", 5)
Q_te2 = query_table(PT, T_te, "p2"); Q_te2["s1_p1"] = Q_te["p1"]; Q_te2["s1_pnew"] = pnew_te1
pnew_te2 = m_new2.predict_proba(Q_te2[NEW_FEATS2].to_numpy(np.float32))[:, 1]
sc2a, pn2a = prior_adjust(sc_te2, pnew_te2, spi)
preds_by_version["v4_s2"] = to_series(make_lists(top_te2, sc2a, pn2a), PT)
log("mean predicted P(new) on TEST: V3 %.3f | stage1 %.3f | stage2 %.3f   (CV true new share: %.3f)" % (pnew_te3.mean(), pnew_te1.mean(), pnew_te2.mean(), yq.mean()))
print("  by species (stage 2):", {s: round(float(pnew_te2[PT['sp'] == i].mean()), 3) for i, s in enumerate(SPECIES)})
log("prior-shift correction (%s): test new-share used per species %s | CV prior %s" % (PRIOR_SHIFT, {k: round(v["use"], 3) for k, v in PIS.items()}, {k: round(v, 3) for k, v in PI_CV.items()}))

ss_path = f"{DATA_DIR}/sample_submission.csv"
ss = pd.read_csv(ss_path) if os.path.exists(ss_path) else None
ident_species = tr.drop_duplicates("identity").set_index("identity").species.to_dict(); test_species = te.set_index("id").species.to_dict()

prev = next((p for p in [os.environ.get("PREV_SUB"), "submission_v3.csv", os.path.join(DATA_DIR, "submission_v3.csv"), "submission.csv", os.path.join(DATA_DIR, "submission.csv")] if p and os.path.isfile(p)), None)
old = pd.read_csv(prev).set_index("id").predictions.str.split().str[0] if prev else None

def check_and_write(tag, arr):
    sub = pd.DataFrame({"id": te.id.to_numpy(), "predictions": arr})
    assert not sub.predictions.isna().any(), "NaN predictions"
    if ss is not None: assert len(sub) == len(ss) and set(sub.id) == set(ss.id) and sub.id.is_unique, "ids differ from sample_submission"
    nr = Counter()
    for i, p in zip(sub.id, sub.predictions):
        labs = p.split(); assert 1 <= len(labs) <= 5 and len(set(labs)) == len(labs), f"bad label count/dupes in {i}"
        for l in labs: assert l == NEW or (l in ident_species and ident_species[l] == test_species[i]), f"invalid label {l} for {i}"
        nr[labs.index(NEW) + 1 if NEW in labs else 0] += 1
    path = os.path.join(OUT_DIR, f"submission_{tag}.csv"); sub.to_csv(path, index=False)
    print(f"{tag}: OK -> {path} | new@1 rate {nr[1] / len(sub):.1%} | NEW absent from list: {nr[0]}")
    return sub

subs = {k: check_and_write(k, v) for k, v in preds_by_version.items()}
print(subs["v4_s2"].sample(min(6, len(te)), random_state=0).to_string(index=False))
chosen = {"v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4_s2" if USE_S2 else "v4_s1"}[SUBMIT_VERSION]
subs[chosen].to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print(f">>> submission.csv written from version {chosen.upper()}")

if old is not None:
    for k, s in subs.items():
        cur = s.set_index("id").predictions.str.split().str[0]; common = old.index.intersection(cur.index)
        print(f"top-1 agreement of {k} with previous {prev}: {(old[common] == cur[common]).mean():.1%}")

