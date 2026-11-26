# ---- 7c. stage 2 without pool-size statistics ----
POOL_STATS = {"q2q1", "q2q3", "nn_sim1", "nn_sim_mean"}
ID_FEATS2R  = [f for f in ID_FEATS2 if f not in POOL_STATS]
NEW_FEATS2R = [f for f in NEW_FEATS2 if f not in POOL_STATS]
print("pool statistics, test vs CV schemes (medians) - these depend on the size/composition of the query pool:")
for c in ["q2q1", "q2q3"]:
    print(f"  {c:12s} test median {np.median(POOL['test'][c]):.3f} | " + " ".join(f"{s} {np.median(POOL[s][c]):.3f}" for s in ALLS))
for c, j in [("nn_sim1", 0)]:
    print(f"  {c:12s} test median {np.median(POOL['test']['nn_sim'][:, j]):.3f} | " + " ".join(f"{s} {np.median(POOL[s]['nn_sim'][:, j]):.3f}" for s in ALLS))
log("V4c: stage 2 without the pool-size statistics " + str(sorted(POOL_STATS)) + " ...")
X2r = pooled(TBL_TR, ID_FEATS2R); X2re = pooled_opt(TBL_EV, ID_FEATS2R)
cvp(X2r, y2, mf2, ID_FEATS2R, X2re, mf2e, TBL_TR, TBL_EV, "p2r")
QT2R, TOP2R = {}, {}
for s in ALLS:
    TOP2R[s] = top_from_table(TB[s], len(TCODE[s]), "p2r", 5)
    QT2R[s] = query_table(POOL[s], TB[s], "p2r"); QT2R[s]["s1_p1"] = QT[s]["p1"]; QT2R[s]["s1_pnew"] = QT[s]["p_new"]
Xq2r = pooled([QT2R[s] for s in TRS], NEW_FEATS2R); Xq2re = pooled_opt([QT2R[s] for s in EVS], NEW_FEATS2R)
cvp(Xq2r, yq, mfq, NEW_FEATS2R, Xq2re, mfqe, [QT2R[s] for s in TRS], [QT2R[s] for s in EVS], "p_new")
RES_J = {s: evaluate("J  V4c: stage 2 w/o pool-size stats", s, *TOP2R[s], tau_q=QT2R[s].p_new.to_numpy()) for s in ALLS}
cv3 = sum(RES_J[s]["ap"].sum() for s in TRS) / n_tr_q

# choice of the final stage: the robust variant is preferred unless it costs more than 0.002 CV MAP@5
CAND = {"H": (cv1, RES_H, TOP1, QT), "I": (cv2, RES_I, TOP2, QT2), "J": (cv3, RES_J, TOP2R, QT2R)}
best_cv = max(v[0] for v in CAND.values())
FIN_NAME = "J" if cv3 >= best_cv - 0.002 else max(CAND, key=lambda k: CAND[k][0])
log("pooled CV MAP@5 over " + str(TRS) + ": " + " | ".join(f"{k} {v[0]:.4f}" for k, v in CAND.items()) + f"  -> final stage {FIN_NAME}")
_, FIN_RES, FIN_TOP, _QF = CAND[FIN_NAME]
FIN_PNEW = {s: _QF[s].p_new.to_numpy() for s in ALLS}
FIN_PSUM = {s: _QF[s].p_sum.to_numpy() for s in ALLS}

# ---- 7d. where to insert new_individual: compare P(new) with the identity probabilities ----
def list_scores(sc, pnew, psum, mode):
    """raw : identity score = ranker probability p_i (v6)
    norm : joint distribution - q_i = (1 - P(new)) * p_i / sum_j p_j  (identities and NEW then sum to one)
    cap  : q_i = (1 - P(new)) * p_i / max(sum_j p_j, 1)   (renormalise only when the identity probabilities over-count)"""
    if mode == "raw": return sc, pnew
    den = np.maximum(psum, 1e-6) if mode == "norm" else np.maximum(psum, 1.0)
    return np.where(sc > NEG / 2, (1 - pnew)[:, None] * sc / den[:, None], sc).astype(np.float32), pnew
LIST_TAB = {}
for mode in ["raw", "norm", "cap"]:
    aps = {s: per_query_ap(FIN_TOP[s][0], *list_scores(FIN_TOP[s][1], FIN_PNEW[s], FIN_PSUM[s], mode), TCODE[s])[0] for s in ALLS}
    LIST_TAB[mode] = {**{s: aps[s].mean() for s in ALLS}, "pooled_train_schemes": sum(aps[s].sum() for s in TRS) / n_tr_q}
LIST_TAB = pd.DataFrame(LIST_TAB).T; print(LIST_TAB.round(4).to_string())
LIST_MODE = max(["norm", "cap"], key=lambda m: LIST_TAB.loc[m, "pooled_train_schemes"])
if LIST_TAB.loc[LIST_MODE, "pooled_train_schemes"] < LIST_TAB.loc["raw", "pooled_train_schemes"] + 0.001: LIST_MODE = "raw"
log(f"list construction: '{LIST_MODE}'")
if LIST_MODE != "raw":
    RES_K = {}
    for s in ALLS:
        scq, pnq = list_scores(FIN_TOP[s][1], FIN_PNEW[s], FIN_PSUM[s], LIST_MODE)
        RES_K[s] = evaluate(f"K  v7: {FIN_NAME} + list '{LIST_MODE}'", s, FIN_TOP[s][0], scq, tau_q=pnq)
    FIN_RES = RES_K; FIN_TOP = {s: (FIN_TOP[s][0], list_scores(FIN_TOP[s][1], FIN_PNEW[s], FIN_PSUM[s], LIST_MODE)[0]) for s in ALLS}
