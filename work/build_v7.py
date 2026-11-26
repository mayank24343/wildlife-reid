"""Builds submission10 (v7) from submission9_25_11 (v6) by explicit, asserted edits of individual cells."""
import json, copy, sys
SRC = "../submission9_25_11.ipynb"
OUT = sys.argv[1] if len(sys.argv) > 1 else "submission10_v7.ipynb"
nb = json.load(open(SRC, encoding="utf-8"))
cells = nb["cells"]
for c in cells:
    c["outputs"] = [] if c["cell_type"] == "code" else c.get("outputs")
    if c["cell_type"] == "code": c["execution_count"] = None
    if c["cell_type"] != "code": c.pop("outputs", None)

def src(i): return "".join(cells[i]["source"])
def setsrc(i, s): cells[i]["source"] = s.splitlines(True)
def rep(i, old, new, count=1):
    s = src(i); n = s.count(old)
    assert n == count, f"cell {i}: expected {count} occurrence(s), found {n}: {old[:80]!r}"
    setsrc(i, s.replace(old, new))
def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}
def code(text): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}

# ---------------------------------------------------------------- cell 0: header
setsrc(0, open("v7_header.md", encoding="utf-8").read())

# ---------------------------------------------------------------- cell 1: settings
rep(1, 'THETA, CAP_NEW = 0.80, 120', '''THETA, CAP_NEW = 0.80, 120                                      # theta of the v6 folds (kept as the evaluation-only scheme "s0")
# v7: mixed-difficulty folds. Every identity draws its own burst threshold, so known validation queries range from
# "near-burst" (0.9) to "different encounter" (0.5). Weights = rounded mixture fit of the TEST top-1 similarity distribution (section 2a).
THETA_MIX = json.loads(os.environ.get("THETA_MIX", '{"0.5": 0.2, "0.7": 0.15, "0.8": 0.4, "0.9": 0.25}'))
THETA_MIX = {float(k): float(v) for k, v in THETA_MIX.items()}
EVAL_V6   = os.environ.get("EVAL_V6", "1") == "1"                # also build the v6 theta=0.8 folds as an evaluation-only scheme
ROOT_SIFT = os.environ.get("ROOT_SIFT", "1") == "1"             # v7: RootSIFT (Hellinger-normalised descriptors) for matching''')
rep(1, 'RUN_STRESS     = os.environ.get("RUN_STRESS", "1") == "1"', 'RUN_STRESS     = os.environ.get("RUN_STRESS", "0") == "1"')
rep(1, 'CACHE_DIR, FIG_DIR = os.environ.get("CACHE_DIR", "cache"), "figures"', 'CACHE_DIR, FIG_DIR = os.environ.get("CACHE_DIR", "cache"), os.environ.get("FIG_DIR", "figures")')
rep(1, '"/home/urshita/Documents/college/sem5/ml/project/projectwild3/reid_student_data_main", ', '')

# ---------------------------------------------------------------- cell 4/5: folds
setsrc(4, open("v7_folds.md", encoding="utf-8").read())
rep(5, '''def burst_clusters(S, y, theta):
    """Connected components of {cos > theta} inside each identity = pseudo-encounters/bursts."""
    lab = np.zeros(len(y), int); off = 0
    for i in np.unique(y):
        idx = np.flatnonzero(y == i)
        if len(idx) == 1: c = np.zeros(1, int)
        else: _, c = connected_components(csr_matrix(S[np.ix_(idx, idx)] > theta), directed=False)''',
'''def burst_clusters(S, y, theta):
    """Connected components of {cos > theta} inside each identity = pseudo-encounters/bursts. theta: a number or {identity: theta}."""
    lab = np.zeros(len(y), int); off = 0
    for i in np.unique(y):
        idx = np.flatnonzero(y == i); th = theta[i] if isinstance(theta, dict) else theta
        if len(idx) == 1: c = np.zeros(1, int)
        else: _, c = connected_components(csr_matrix(S[np.ix_(idx, idx)] > th), directed=False)''')
rep(5, '''def testlike_split(S, y, seed, k=N_FOLDS, theta=THETA, pi=PI_NEW, cap=CAP_NEW):
    """-> fold (n,) with -1 = never validated, and the set of identities held out entirely (the 'new' ones)."""
    n = len(y); rng = np.random.default_rng(seed)
    groups = burst_clusters(S, y, theta); fold = np.full(n, -1)''',
'''def testlike_split(S, y, seed, k=N_FOLDS, theta=THETA, pi=PI_NEW, cap=CAP_NEW):
    """-> fold (n,) with -1 = never validated, the set of identities held out entirely (the 'new' ones), and the theta of every row.
    theta: a number (v6) or a {theta: weight} mixture (v7: every identity draws its own burst threshold)."""
    n = len(y); rng = np.random.default_rng(seed)
    if isinstance(theta, dict):
        vals, w = np.array(list(theta)), np.array(list(theta.values())); w = w / w.sum()
        theta = {i: float(rng.choice(vals, p=w)) for i in np.unique(y)}
    th_row = np.array([theta[i] if isinstance(theta, dict) else theta for i in y], float)
    groups = burst_clusters(S, y, theta); fold = np.full(n, -1)''')
rep(5, '''    for i, kk in zip(newids, rng.permutation(len(newids)) % k): fold[y == i] = kk
    return fold, set(newids)''', '''    for i, kk in zip(newids, rng.permutation(len(newids)) % k): fold[y == i] = kk
    return fold, set(newids), th_row''')
rep(5, '''FOLDS = {}                                                   # scheme -> (fold per train row [-1 = never a query], held-out-identity flag per train row)
for seed in CV_SEEDS:
    fold = np.full(len(tr), -1); unseen = np.zeros(len(tr), bool)
    for sp in SPECIES:
        f_sp, newids = testlike_split(_SIMS[sp], ID_SP[sp], seed)
        fold[ROWS_SP[sp]] = f_sp; unseen[ROWS_SP[sp]] = np.isin(ID_SP[sp], list(newids))
    FOLDS[f"s{seed}"] = (fold, unseen)
TRS = tuple(FOLDS); PRIMARY = TRS[0]                         # TRS = schemes used for training AND validation''',
'''FOLDS, THETA_ROW = {}, {}                                    # scheme -> (fold per train row [-1 = never a query], held-out-identity flag per train row)
def build_scheme(name, seed, theta):
    fold = np.full(len(tr), -1); unseen = np.zeros(len(tr), bool); th = np.zeros(len(tr))
    for sp in SPECIES:
        f_sp, newids, th_sp = testlike_split(_SIMS[sp], ID_SP[sp], seed, theta=theta)
        fold[ROWS_SP[sp]] = f_sp; unseen[ROWS_SP[sp]] = np.isin(ID_SP[sp], list(newids)); th[ROWS_SP[sp]] = th_sp
    FOLDS[name] = (fold, unseen); THETA_ROW[name] = th
for seed in CV_SEEDS: build_scheme(f"m{seed}", seed, THETA_MIX)          # v7 mixed-difficulty schemes: train + validate
TRS = tuple(FOLDS); PRIMARY = TRS[0]                         # TRS = schemes used for training AND validation
if EVAL_V6: build_scheme("s0", 0, THETA)                     # v6 folds (theta 0.8), evaluation only''')
rep(5, '''                assert ms.max() <= THETA + 1e-6, f"burst leakage: same-identity similarity {ms.max():.3f} > {THETA}"''',
'''                assert (ms <= THETA_ROW[name][rows][kn] + 1e-6).all(), f"burst leakage: same-identity similarity above the identity's theta"''')
rep(5, 'for name in TRS: audit_folds(name)', 'for name in FOLDS: audit_folds(name)')
rep(5, 'log(f"validation folds {TRS}: leakage audit PASSED (disjoint; held-out identities absent from gallery; no same-identity pair above {THETA})")',
       'log(f"validation folds {tuple(FOLDS)}: leakage audit PASSED (disjoint; held-out identities absent from gallery; no same-identity pair above the identity\'s theta)")')
rep(5, 'for name in TRS:\n    f_, u_ = FOLDS[name]', 'for name in FOLDS:\n    f_, u_ = FOLDS[name]')
# section 2a: test-likeness of the fold designs (inserted after cell 5)
FOLD_DIAG = code(open("v7_fold_diag.py", encoding="utf-8").read())

# ---------------------------------------------------------------- cell 9: evaluation-only schemes
rep(9, '''EVS = ()                                                   # evaluation-only schemes
if RUN_STRESS:''', '''EVS = ("s0",) if EVAL_V6 else ()                            # evaluation-only schemes (v6 folds, optional stress)
if RUN_STRESS:''')

# ---------------------------------------------------------------- cell 17: RootSIFT
rep(17, '''                out[i] = (src.get(f"{i}_xy").reshape(-1, 2).astype(np.float32), src.get(f"{i}_desc").reshape(-1, 128)); break''',
'''                d = src.get(f"{i}_desc").reshape(-1, 128)
                out[i] = (src.get(f"{i}_xy").reshape(-1, 2).astype(np.float32), root_sift(d) if ROOT_SIFT else d); break''')
rep(17, '''_TR_IDS, _TE_IDS = set(tr.id.to_numpy()), set(te.id.to_numpy())''', '''_TR_IDS, _TE_IDS = set(tr.id.to_numpy()), set(te.id.to_numpy())
def root_sift(d):
    """RootSIFT (Arandjelovic & Zisserman, 2012): L1-normalise, then element-wise sqrt. Euclidean distance between RootSIFT vectors
    = Hellinger distance between the original histograms, which down-weights the few dominant gradient bins of a descriptor."""
    d = d.astype(np.float32); d /= np.maximum(d.sum(1, keepdims=True), 1e-9); return np.sqrt(d).astype(np.float16)   # float16 in memory (half of float32); matched as float32''')
# incremental pair-cache saves (a stop loses at most one batch, not a whole scheme)
rep(17, '''        q_list, g_lists = list(grp.index), list(grp.values)
        SIFT = load_sift_for(list(dict.fromkeys(q_list + [g for gl in g_lists for g in gl])))
        st = run_sift(SIFT, q_list, g_lists, label)
        PAIRS.add(pair_key(np.repeat(np.array(q_list, dtype=object), [len(l) for l in g_lists]), np.array([g for gl in g_lists for g in gl], dtype=object)), st)
        PAIRS.save(); pos = PAIRS.lookup(keys); assert (pos >= 0).all()''',
'''        q_list, g_lists = list(grp.index), list(grp.values)
        SIFT = load_sift_for(list(dict.fromkeys(q_list + [g for gl in g_lists for g in gl])))
        csum = np.cumsum([len(l) for l in g_lists]); BATCH = 150_000; a = 0
        while a < len(q_list):
            b = int(np.searchsorted(csum, (csum[a - 1] if a else 0) + BATCH)) + 1; b = min(max(b, a + 1), len(q_list))
            ql, gl_ = q_list[a:b], g_lists[a:b]
            st = run_sift(SIFT, ql, gl_, f"{label} batch {a}-{b} of {len(q_list)} queries")
            PAIRS.add(pair_key(np.repeat(np.array(ql, dtype=object), [len(l) for l in gl_]), np.array([g for gl in gl_ for g in gl], dtype=object)), st)
            PAIRS.save(); a = b
        del SIFT
        pos = PAIRS.lookup(keys); assert (pos >= 0).all()''')
rep(17, '''        self.sig = dict(ratio=SIFT_RATIO, ransac=RANSAC_THRESH, iters=RANSAC_ITERS, ids=hashlib.md5("|".join(ALL_IDS).encode()).hexdigest()[:12])''',
'''        self.sig = dict(ratio=SIFT_RATIO, ransac=RANSAC_THRESH, iters=RANSAC_ITERS, ids=hashlib.md5("|".join(ALL_IDS).encode()).hexdigest()[:12])
        if ROOT_SIFT: self.sig["root"] = True''')
rep(17, '''PAIRS = PairCache(os.path.join(CACHE_DIR, "sift_pair_cache.npz"))''', '''PAIRS = PairCache(os.path.join(CACHE_DIR, "sift_pair_cache_root.npz" if ROOT_SIFT else "sift_pair_cache.npz"))''')
rep(17, '''
seed_from_legacy()
''', '''
if not ROOT_SIFT: seed_from_legacy()
''')

json.dump({**nb, "cells": cells}, open("_tmp.ipynb", "w", encoding="utf-8"))
exec(open("build_v7_part3.py", encoding="utf-8").read())
exec(open("build_v7_part2.py", encoding="utf-8").read())
