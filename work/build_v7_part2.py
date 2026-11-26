# part 2 of the v7 builder (exec'd from build_v7.py; shares its namespace)

# cell 5: keep _SIMS for the fold diagnostic cell, which deletes it
rep(5, '\ndel _SIMS', '\n')

# cell 1: submission version
rep(1, 'SUBMIT_VERSION = "v4"', 'SUBMIT_VERSION = "v7"')
rep(1, 'assert SUBMIT_VERSION in ("v1", "v2", "v3", "v4")', 'assert SUBMIT_VERSION in ("v1", "v2", "v3", "v4", "v7")')

# cell 34: labels for the evaluation-only scheme
rep(34, 'print("\\nSTRESS (evaluation only, deliberately much harder than test - read as a lower bound, not an estimate):")',
        'print(f"\\nEVALUATION-ONLY scheme {EVS[0]} (models never trained on it):")')
rep(34, '([("STRESS (evaluation only)", RESULTS_DF.loc[EVS[0]]["MAP@5"], None)] if EVS else [])',
        '([(f"{EVS[0]} (evaluation only)", RESULTS_DF.loc[EVS[0]]["MAP@5"], None)] if EVS else [])')
rep(34, '"I": "#8172B3"}', '"I": "#8172B3", "J": "#937860", "K": "#DA8BC3"}')

# cell 32 -> FIN_* is set there; the new cells below may override it
NEW_AFTER_32 = [md(open("v7_robust.md", encoding="utf-8").read()), code(open("v7_robust.py", encoding="utf-8").read())]

# cell 45: final inference adds the v7 model
rep(45, '''preds_by_version["v4_s2"] = to_series(make_lists(top_te2, sc2a, pn2a), PT)''',
'''preds_by_version["v4_s2"] = to_series(make_lists(top_te2, sc2a, pn2a), PT)
# ---- v7: stage 2 variant chosen in section 7c, list construction chosen in section 7d ----
if FIN_NAME == "J":
    m_id2r = fit_model(X2r, y2, ID_FEATS2R); m_new2r = fit_model(Xq2r, yq, NEW_FEATS2R)
    T_te["p2r"] = m_id2r.predict_proba(T_te[ID_FEATS2R].to_numpy(np.float32))[:, 1]
    top_f, sc_f = top_from_table(T_te, nT, "p2r", 5)
    Q_f = query_table(PT, T_te, "p2r"); Q_f["s1_p1"] = Q_te["p1"]; Q_f["s1_pnew"] = pnew_te1
    pnew_f = m_new2r.predict_proba(Q_f[NEW_FEATS2R].to_numpy(np.float32))[:, 1]
elif FIN_NAME == "I": top_f, sc_f, Q_f, pnew_f = top_te2, sc_te2, Q_te2, pnew_te2
else:                 top_f, sc_f, Q_f, pnew_f = top_te1, sc_te1, Q_te, pnew_te1
sc_l, pn_l = list_scores(sc_f, pnew_f, Q_f["p_sum"].to_numpy(), LIST_MODE)
sc7a, pn7a = prior_adjust(sc_l, pn_l, spi)
preds_by_version["v7"] = to_series(make_lists(top_f, sc7a, pn7a), PT)
log(f"v7 = stage '{FIN_NAME}' + list mode '{LIST_MODE}': mean P(new) on TEST {pnew_f.mean():.3f} "
    f"(hyena {pnew_f[spi == 0].mean():.3f}, leopard {pnew_f[spi == 1].mean():.3f}) | independent mixture estimate of the new share: "
    + ", ".join(f"{k} {v['hat']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" for k, v in PIS.items()))''')
rep(46, '''chosen = {"v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4_s2" if USE_S2 else "v4_s1"}[SUBMIT_VERSION]''',
        '''chosen = {"v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4_s2" if USE_S2 else "v4_s1", "v7": "v7"}[SUBMIT_VERSION]''')

# assemble: insert the diagnostic cell after 5 and the robust cells after 32
new_cells = []
for i, c in enumerate(cells):
    new_cells.append(c)
    if i == 5: new_cells.append(FOLD_DIAG)
    if i == 32: new_cells.extend(NEW_AFTER_32)
nb["cells"] = new_cells
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", OUT, "with", len(new_cells), "cells")
