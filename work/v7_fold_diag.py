# ---- 2a. Which fold design looks like the test set? (features only; no test labels exist) ----
# For every scheme: top-1 centred cosine of each validation query to its fold gallery, vs the top-1 of each TEST image to the full train set.
qs_ = [.05, .1, .25, .5, .75, .9, .95, .99]
DIAG = []
for i_, sp in enumerate(SPECIES):
    m_tr, m_te = (tr.species == sp).to_numpy(), (te.species == sp).to_numpy()
    Xs_ = l2n(np.vstack([Xtr[m_tr], Xte[m_te]])); Xc_ = l2n(Xs_ - Xs_.mean(0))
    t1_ = (Xc_[int(m_tr.sum()):] @ Xc_[:int(m_tr.sum())].T).max(1)
    S_ = _SIMS[sp]; print(f"{sp:8s} TEST        top-1 quantiles {np.round(np.quantile(t1_, qs_), 3)}  share > 0.8: {np.mean(t1_ > 0.8):.3f}")
    for name in FOLDS:
        f_ = FOLDS[name][0][ROWS_SP[sp]]; v_ = []
        for k in range(N_FOLDS):
            q_, g_ = np.flatnonzero(f_ == k), np.flatnonzero(f_ != k)
            if len(q_): v_.append(S_[np.ix_(q_, g_)].max(1))
        v_ = np.concatenate(v_)
        DIAG.append(dict(species=sp, scheme=name, KS_vs_test=ks_2samp(v_, t1_).statistic, share_gt_08=np.mean(v_ > 0.8)))
        print(f"{sp:8s} {name:11s} top-1 quantiles {np.round(np.quantile(v_, qs_), 3)}  share > 0.8: {np.mean(v_ > 0.8):.3f} | KS vs test {DIAG[-1]['KS_vs_test']:.3f}")
DIAG = pd.DataFrame(DIAG); print(DIAG.pivot(index="scheme", columns="species", values="KS_vs_test").round(3).to_string())
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, sp in zip(axs, SPECIES):
    m_tr, m_te = (tr.species == sp).to_numpy(), (te.species == sp).to_numpy()
    Xs_ = l2n(np.vstack([Xtr[m_tr], Xte[m_te]])); Xc_ = l2n(Xs_ - Xs_.mean(0)); t1_ = (Xc_[int(m_tr.sum()):] @ Xc_[:int(m_tr.sum())].T).max(1)
    ax.plot(np.sort(t1_), np.linspace(0, 1, len(t1_)), "k", lw=2.5, label="test")
    for name in FOLDS:
        f_ = FOLDS[name][0][ROWS_SP[sp]]; S_ = _SIMS[sp]
        v_ = np.concatenate([S_[np.ix_(np.flatnonzero(f_ == k), np.flatnonzero(f_ != k))].max(1) for k in range(N_FOLDS) if (f_ == k).any()])
        ax.plot(np.sort(v_), np.linspace(0, 1, len(v_)), "--" if name.startswith("s") else "-", lw=1.2, label=name)
    ax.set_title(f"{sp}: CDF of top-1 similarity to the gallery"); ax.set_xlabel("top-1 centred cosine"); ax.legend(fontsize=8)
savefig("fold_design_vs_test.png")
del _SIMS
