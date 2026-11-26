# part 3 of the v7 builder (v7b): raw-cosine evidence for the ranker, training on both difficulty regimes, prior shift off

# cell 1: flags
rep(1, 'PRIOR_SHIFT    = os.environ.get("PRIOR_SHIFT", "auto")          # off | auto (conservative, within the estimated band) | force (point estimate)',
       'PRIOR_SHIFT    = os.environ.get("PRIOR_SHIFT", "off")           # v7: off by default - the mixture estimate is unreliable when the CV components differ from test (section 5b) | auto | force')
rep(1, 'ROOT_SIFT = os.environ.get("ROOT_SIFT", "1") == "1"             # v7: RootSIFT (Hellinger-normalised descriptors) for matching',
       '''ROOT_SIFT = os.environ.get("ROOT_SIFT", "1") == "1"             # v7: RootSIFT (Hellinger-normalised descriptors) for matching
TRAIN_V6  = os.environ.get("TRAIN_V6", "1") == "1"              # v7: also TRAIN on the v6 folds s0 (both difficulty regimes; all predictions stay out-of-fold by identity)
RAW_FEATS = os.environ.get("RAW_FEATS", "1") == "1"             # v7: raw (un-whitened) MiewID cosine evidence in the identity ranker''')

# cell 5: s0 joins the training schemes when TRAIN_V6
rep(5, '''TRS = tuple(FOLDS); PRIMARY = TRS[0]                         # TRS = schemes used for training AND validation
if EVAL_V6: build_scheme("s0", 0, THETA)                     # v6 folds (theta 0.8), evaluation only''',
'''if EVAL_V6 or TRAIN_V6: build_scheme("s0", 0, THETA)        # v6 folds (theta 0.8)
TRS = tuple(k for k in FOLDS if k != "s0" or TRAIN_V6); PRIMARY = TRS[0]   # TRS = schemes used for training AND validation''')
# cell 9: s0 is evaluation-only only when it is not trained on
rep(9, '''EVS = ("s0",) if EVAL_V6 else ()                            # evaluation-only schemes (v6 folds, optional stress)''',
       '''EVS = ("s0",) if (EVAL_V6 and not TRAIN_V6) else ()          # evaluation-only schemes (v6 folds, optional stress)''')

# cell 19: raw cosine of every candidate + raw centroid similarity
rep(19, '''                r_sc=np.take_along_axis(Rs, rc, 1), r_gc=gcode[rc].astype(np.int32))''',
'''                r_sc=np.take_along_axis(Rs, rc, 1), r_gc=gcode[rc].astype(np.int32),
                Sw=np.take_along_axis(Rs, cand, 1), cenr=centroid_sims(Rq, Rg, gcode, gc))''')

# cell 29: identity-table features from the raw cosine
rep(29, '''                       "kq": P["kq"][:, :M].ravel(), "kg": P["kg"][:, :M].ravel()})''',
'''                       "kq": P["kq"][:, :M].ravel(), "kg": P["kg"][:, :M].ravel(), "sw": P["Sw"][:, :M].ravel(), "cenr": P["cenr"][:, :M].ravel()})''')
rep(29, '''n_img=("one", "sum"), best_rk=("rk", "min"), gn=("gn", "first"), bk=("bk", "first"), frac_max=("frac", "max"))''',
'''n_img=("one", "sum"), best_rk=("rk", "min"), gn=("gn", "first"), bk=("bk", "first"), frac_max=("frac", "max"),
            sw_max=("sw", "max"), cenr=("cenr", "first"))''')
rep(29, '''    T["inl_top2"] = top2["inl"].mean()''', '''    T["inl_top2"] = top2["inl"].mean()
    T["sw_top2"] = df.sort_values(["qi", "code", "sw"], ascending=[True, True, False], kind="stable").groupby(["qi", "code"], sort=False).head(2).groupby(["qi", "code"], sort=False)["sw"].mean()''')
rep(29, '''("cen", "q_cen_max"), ("frac_max", "q_frac_max")]:''', '''("cen", "q_cen_max"), ("frac_max", "q_frac_max"), ("sw_max", "q_sw_max"), ("cenr", "q_cenr_max")]:''')
rep(29, '''    T["cenl_rank"] = gq["cenl"].rank(ascending=False, method="first")''', '''    T["cenl_rank"] = gq["cenl"].rank(ascending=False, method="first"); T["sw_rank"] = gq["sw_max"].rank(ascending=False, method="first")''')
rep(29, '''    T = T[(T.v2_rank <= TOP_ID) | (T.sl_rank <= TOP_L) | (T.cenl_rank <= 3)].reset_index(drop=True)''',
'''    keep = (T.v2_rank <= TOP_ID) | (T.sl_rank <= TOP_L) | (T.cenl_rank <= 3)
    if RAW_FEATS: keep |= (T.sw_rank <= TOP_L)
    T = T[keep].reset_index(drop=True)''')
rep(29, '''            "frac_max", "inl_top2", "q_frac_max", "frac_max_rel"]''', '''            "frac_max", "inl_top2", "q_frac_max", "frac_max_rel"]
RAW_ID = ["sw_max", "sw_top2", "cenr", "q_sw_max", "q_cenr_max", "sw_max_rel", "cenr_rel", "sw_rank"]
if RAW_FEATS: ID_FEATS = ID_FEATS + RAW_ID''')
rep(29, '''NEW_FEATS = ["p1", "p2", "p3", "p_sum", "p_gap", "n_id", "n_strong_ids", "q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "q_sl_max", "q_cenl_max", "sp", "q_vp", "kq0"]''',
'''NEW_FEATS = ["p1", "p2", "p3", "p_sum", "p_gap", "n_id", "n_strong_ids", "q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "q_sl_max", "q_cenl_max", "sp", "q_vp", "kq0"]
if RAW_FEATS: NEW_FEATS = NEW_FEATS + ["q_sw_max", "q_cenr_max"]''')
rep(29, '''    for c in ["q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "q_sl_max", "q_cenl_max"]: Q[c] = g[c].first().reindex(ix).to_numpy()''',
'''    for c in ["q_v2_max", "q_sr_max", "q_inl_max", "q_sc_max", "q_sl_max", "q_cenl_max"] + (["q_sw_max", "q_cenr_max"] if RAW_FEATS else []): Q[c] = g[c].first().reindex(ix).to_numpy()''')
