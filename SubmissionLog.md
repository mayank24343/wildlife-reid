# Wildlife Re-ID (Problem 3): Submission Log

Shared log for the team. **Adding an entry every time we make a Kaggle submission**, including things that did *not* work, so nobody redoes or undoes them.
Metric: MAP@5 (public LB = 30% of test; final ranking = other 70%). 
Final Kaggle deadline: **27 Nov 2026**. 
Presentation: **10 Dec**.

Course requirements for this repo: tag every submission's notebook with a unique version and date, keep the TA as a collaborator, and keep commits attributable (individual contributions are graded).

---

## 1. Submission Summary

| # | Date | Who | Notebook | What it is | Kaggle public LB | Best CV (own notebook) |
|---|------|-----|----------|------------|------------------|------------------------|
| 1 | 19-11 | Mayank | `submission1_19_11.ipynb` | MiewID cosine baseline, centre + PCA-whiten, tuned τ | **0.78817** | not recorded in the notebook |
| 2 | 20-11 | Mayank | `submission2_20_11.ipynb` | + re-ranking + SIFT/RANSAC + hand-tuned blend (V2) | **0.81168** | 0.8738 (older tuning notebook, different CV) |
| 3 | 22-11 | Anup | `submission3_22_11.ipynb` | + learned identity ranker + learned P(new) (V3) | **0.83115** (see note a) | rand 0.9049 |
| 4 | 23-11 | Nandika | `submission4_23_11.ipynb` | + supervised projection, extra candidates, hard CV (V4 stage 1) | **0.84023** | rand 0.9069 / hard 0.7767 |
| 6 | 23-11 | TBD | `submission6_23_11.ipynb` | V4 re-tuned (wider α grid, `CAND_LDA` 60) + transductive stage 2 submitted | **0.84842** | rand 0.9135 / hard 0.7826 |
| 7 | 24-11 | Nandika | `submission7_24_11.ipynb` | Sub 6 + extra RootSIFT evidence (strict inliers, coverage) in both stages, wider candidate pool, CV-gated selection | **0.84870** | rand 0.9151 / hard 0.7866 |
| 8 | 24-11 | TBD | `submission8_24_11.ipynb` | Per-species V4 strategy (pooled hyena / separate leopard), viewpoint-coverage features and viewpoint-specific identity prototypes; stage 1 submitted because stage 2 regressed on hard CV | not recorded | rand 0.9103 / hard 0.7847 (stage 1) |
| 9 | 25-11 | TBD | `submission9_25_11.ipynb` | v6: test-calibrated open-set CV with burst-grouped known queries, controlled 0.30 new-identity share, identity-size cap, and conservative prior-shift correction; stage 2 selected | not recorded | test-like CV 0.9145 (stage 2, mean over 3 seeds); stress 0.7359 |
| 11 | 26-11 | TBD | `submission11_26_11.ipynb` | v7: Sub 9 pipeline + flank-coverage features (`gn_fl`, `fl_frac`) for candidate identities in the learned ranker | **0.86063** | test-like CV 0.9166 (stage 2, mean over 3 seeds); stress 0.7484 |
| 12 | 27-11 | Nandika | `submission12_26_11.ipynb` | v8: Sub 11 + score-normalised similarities (Z/T-norm style) + identity compactness, neighbour P(new) in stage 2, 4 LightGBM seeds | **0.86690** | test-like CV 0.9210 (stage 2, mean over 3 seeds); stress 0.7541 |


---

## 2. Submission entries

### Sub 1: Baseline (Mayank, 19-11): LB 0.78817

**What it does**
- Per species (species are never mixed), embeds MiewID descriptors: L2-norm → centre → PCA(512, whiten) → L2-norm. Fitted on train+test features, no labels.
- Cosine retrieval, best image per identity (`agg=max`), top-5 identities.
- `new_individual` is inserted at the rank equal to the number of candidate identities with score ≥ τ. τ is tuned per species on pooled out-of-fold scores.
- Greedy config search on open-set CV (30% of validation queries are held-out identities). It tried `top2/top3` aggregation, centring, PCA 256/512, DBA, query expansion (QE) and a viewpoint bonus. A change was kept only if CV improved by >1e-4.
- Inferred from Sub 2's header: the kept config was centre + PCA-512 with `agg=max`, no DBA/QE and no viewpoint bonus.

**Diagnostics worth remembering**
- CV *known* queries had a cluster of near-duplicate matches (top-1 cosine > 0.9) that the real test doesn't have, so random CV is optimistic. The notebook started "burst" grouping (keep near-duplicates on one side of the split) but never finished calibrating it (`BURST_COS` was left as `None`).
- This is the same idea Sub 4 later implemented as the "hard" folds.
- Section 7 estimates the test new-individual share by mixture-fitting top-1 score distributions (reusable).

**Left open (addressed next):** no use of SIFT keypoints at all; no re-ranking; `new_individual` handled by a single threshold.

---

### Sub 2: V2 = re-ranking + SIFT/RANSAC + blend (Mayank, 20-11): LB 0.81168 (+0.0235)

**What it does** (inference-only notebook; params come from a separate tuning notebook, `reid_submission_v2.ipynb`, and `best_config_v2.json`)
1. Same embedding as Sub 1 (centre + PCA-512 whiten).
2. **k-reciprocal re-ranking** (k1=20, k2=6, λ=0.3) over stacked query+gallery, keeping the top-200 gallery candidates per query.
3. **SIFT verification** of the top-100 candidates: Lowe ratio 0.78 + mutual nearest neighbour + RANSAC partial-affine (5 px, 2000 iters). Output: number of RANSAC inliers.
4. **Image score blend:** `0.2812·S_rerank + 0.7188·S_cos + 0.1906·log(1+max(inliers−3,0)) + viewpoint bonus`.
   - Viewpoint bonus: same −0.0062, adjacent +0.0094, opposite +0.0125.
5. Identity score = mean of the identity's best 2 images (`top2`).
6. `new_individual` inserted by a per-species τ (hyena 0.2100, leopard 0.3439).

**Improves on Sub 1:** adds local-feature geometry (SIFT/RANSAC) and re-ranking, and replaces the viewpoint bonus idea with a bucketed one.

**Reproducibility warning:** this notebook needs `best_config_v2.json` and the tuning notebook, which are *not* in the folder we looked at. Make sure both are committed.

**Left open (addressed next):** blend weights and τ are hand-tuned with a handful of numbers; `new_individual` is a hard threshold, not a probability.

---

### Sub 3: V3 = learned identity ranker + learned P(new) (Anup, 22-11): LB 0.83115 (+0.0195), see note (a)

**What it does**
- Same upstream pipeline as V2, but candidates widened to `CAND_K=300`, `TOP_M_SIFT=150`, with a **pair-level SIFT cache** (`cache/sift_pair_cache.npz`) so reruns are fast.
- **New CV:** whole identities are held out to simulate unseen animals (about 40% of queries are "new"), and singletons can never be known queries.
- **Stage 1: identity ranker.** One row per (query, candidate identity) with aggregated evidence (re-rank score, cosine, V2 blend, inliers, number of strong matches, gallery size, viewpoint bucket, query-relative features, and so on). Ensemble of 3 HistGradientBoosting configs + 2 LightGBM, trained out-of-fold.
- **Stage 2: P(new_individual) model.** Gradient boosting on the top-3 identity probabilities, their gap and the query-level evidence. The `new_individual` token is ranked among identities by probability.
- Error-analysis section added (failure taxonomy, MAP by viewpoint / gallery size / keypoint count, calibration, feature importance).

**CV ladder (one notebook, same folds):**

| Row | Adds | MAP@5 | Δ |
|---|---|---|---|
| A (V1) | raw cosine | 0.8347 | |
| B | centre + PCA-whiten | 0.8802 | +0.0455 |
| C | k-reciprocal re-rank | 0.8814 | +0.0012 |
| D | top-2 aggregation | 0.8825 | +0.0011 |
| E (V2) | SIFT/RANSAC + viewpoint | 0.8968 | +0.0143 |
| F | learned ranker, tuned τ | 0.9070 | +0.0102 |
| G (V3, **submitted**) | + learned P(new) | 0.9049 | −0.0021 |
| H | + query-to-query cue | 0.9164 | +0.0115 (**not used**) |

**Things to know**
- The query-to-query cue (H) was switched off (`USE_Q2Q=0`) because each CV fold has ~1,432 queries while the test set has 2,791, so the "nearest other query" feature isn't distributed the same way. Sub 4 retried the idea with SIFT-verified neighbours (see below).
- On CV, F (tuned τ) was slightly better than G (learned P(new)). G was submitted anyway; Sub 4's pooled CV later showed G ≥ F.
- Predicted mean P(new) on test = 0.347 vs 0.403 in CV.
- Retrieval recall ceiling (true identity among candidates) = 97.6%.
- Biggest remaining losses (of 1 − MAP = 0.095): true identity ranked outside top-5 (0.025), unseen animal matched to a known identity (0.022), correct at rank 2–5 (0.019).
- Top feature: `sc_max_rel` (cosine of the identity relative to the query's best candidate).

**Left open (addressed next):** CV can't tell when a query's nearest siblings are missing (random CV is too easy); no supervised similarity learning; only top-re-ranked candidates get SIFT.

---

### Sub 4: V4 = supervised projection + hard CV (Nandika, 23-11): LB 0.84023 (+0.0091)

**What it does**
- **Second, harder CV ("hard"):** each known identity is split into clusters of look-alike images (average-linkage on the PCA embedding) and whole clusters are held out, so a query's nearest siblings are not in the gallery. Every learned model is trained on pooled rand+hard queries (identity-grouped folds) and every table row is scored under both schemes.
- **Supervised linear projection (within-class covariance whitening, WCCN)** fitted on the gallery only (inside every fold and for test). Shrinkage α chosen per species by closed-set top-1; **both species chose α=0.1, the smallest value in the grid**.
- The projection adds image-level and identity-centroid similarity channels and proposes up to `CAND_LDA=40` extra candidate images per query (all get SIFT; `TOP_M_EFF=190`).
- **Focused learner:** the ranker only sees the best `TOP_ID=30` identities (+ `TOP_L=10` from the learned space + top-3 centroids).
- **Stage 2 (transductive):** each query's 5 nearest queries are SIFT-verified and their stage-1 identity probabilities are fed to a second ranker and a second P(new) model. It only gets used if it gains >0.002 on **both** CV schemes.
- **Submitted: stage 1** (`submission.csv` = `v4_s1`). Stage 2 gained on rand (+0.0049) but *lost* on hard (−0.0067), so it was not used.

**CV results (same folds for every row):**

| Row | Adds | rand | hard |
|---|---|---|---|
| A (V1) | raw cosine | 0.8347 | 0.6941 |
| B | + centre / PCA-whiten | 0.8802 | 0.6795 |
| C | + re-rank | 0.8814 | 0.6900 |
| D | + top-2 agg | 0.8829 | 0.7007 |
| E (V2) | + SIFT/RANSAC + viewpoint | 0.8963 | 0.7510 |
| F | learned ranker, tuned τ | 0.9001 | 0.7481 |
| G (V3) | + learned P(new) | 0.9009 | 0.7543 |
| H (V4 stage 1, **submitted**) | + learned-space channels/candidates | 0.9069 | 0.7767 |
| I (V4 stage 2, not used) | + transductive neighbours | 0.9118 | 0.7700 |

**Things to know**
- **PCA-whitening hurts under hard CV** (B < A: 0.6795 vs 0.6941) even though it was the biggest single gain under random CV (+0.0455). Much of that gain may be near-duplicate leakage. **SIFT/RANSAC is the most robust gain** (+0.050 on hard) and the learned space is next (+0.022 on hard).
- Retrieval recall ceiling is **lower than Sub 3's** (rand 96.9% vs 97.6%; hard 87.4%), probably because of the `TOP_ID=30` focus. Retrieval misses went 102 → 132 rand queries.
- Hard-CV losses (1 − MAP = 0.22): retrieval misses (0.072), correct at rank 2–5 (0.046), outside top-5 (0.044), `new_individual` placed above the true identity (0.041).
- Hyena is the weak species on hard CV (0.68–0.70 vs leopard 0.79).
- Predicted mean P(new) on test = 0.30 (stage 1) vs 0.42 pooled CV share. The predicted new@1 rate on test is ~40–42% for every version, so the model isn't wildly off, but the true test new-share is unknown.
- Top-1 agreement between V3 and V4-s1 on test is only 93.4%, so the two models disagree on ~7% of queries.

**Left open:** see section 4.

---

### Sub 6: V4 re-tuned + stage 2 (TBD, 23-11): LB 0.84842 (+0.0082 vs Sub 4)

**Notebook / git tag:** `submission6_23_11.ipynb` / tag to be added
**Based on:** Sub 4 (same V4 pipeline, `SUBMIT_VERSION = "v4"`; Sub 5 not logged)

**What it does**
- Same pipeline as Sub 4: WCCN projection, rand + hard CV, focused learner (`TOP_ID=30`, `TOP_L=10`), transductive stage 2.
- **Projection shrinkage grid extended** to α ∈ {1.0, 0.6, 0.35, 0.2, 0.1, 0.05, 0.02, 0.01}. Chosen: **hyena α=0.05, leopard α=0.01** (Sub 4: both 0.1).
- **`CAND_LDA` 40 → 60**, so `TOP_M_EFF` 190 → 210. New SIFT pairs were computed on top of the cache (about 56k test, 115k rand, 99k hard).
- **Submitted stage 2** (`submission.csv` = `v4_s2`). Stage 2 now passes the "gain > 0.002 on both CV schemes" gate: rand +0.0050, hard +0.0046. Pooled CV MAP@5 0.8432 (stage 1) → 0.8481 (stage 2).

**What changed vs the previous submission and why**
- Sub 4 left α at the edge of the grid and had to drop stage 2 (it lost on hard CV). This run tests both: wider α and more candidates, and stage 2 ends up usable.

**CV results (same folds for every row within this notebook)**

| Row | Adds | rand | hard |
|---|---|---|---|
| A (V1) | raw cosine | 0.8347 | 0.6941 |
| B | + centre / PCA-whiten | 0.8802 | 0.6795 |
| C | + re-rank | 0.8814 | 0.6900 |
| D | + top-2 agg | 0.8840 | 0.7031 |
| E (V2) | + SIFT/RANSAC + viewpoint | 0.8967 | 0.7533 |
| F | learned ranker, tuned τ | 0.9007 | 0.7511 |
| G (V3) | + learned P(new) | 0.9027 | 0.7625 |
| H (V4 stage 1) | + learned-space channels/candidates | 0.9085 | 0.7780 |
| I (V4 stage 2, **submitted**) | + transductive neighbours | 0.9135 | 0.7826 |

**What worked / what did NOT work**
- **Wider α grid gave almost nothing.** Closed-set top-1: hyena 0.8889 (α=0.1) → 0.8896 (α=0.05), then falls (0.8875 at 0.02, 0.8860 at 0.01). Leopard is flat from α=0.6 to 0.01 (0.8507 → 0.8521). So α is not a real lever; don't spend more time on it alone.
- **Stage 2 helps hyena, not leopard.** Hard CV: hyena 0.6831 → 0.7043, leopard 0.8126 → 0.8111. Rand: hyena 0.8854 → 0.8957, leopard 0.9169 → 0.9200.
- **Not isolated:** Stage 1 CV barely moved vs Sub 4 (rand 0.9069 → 0.9085, hard 0.7767 → 0.7780), so most of the LB gain is probably stage 2. We have no LB score for this run's stage 1, so α, `CAND_LDA` and stage 2 are not separated.
- **Unexplained:** stage 2 went from −0.0067 (Sub 4) to +0.0046 on hard CV. Sub 4's fold-size hypothesis (CV folds ~1,430 queries vs 2,791 test) is neither confirmed nor ruled out.
- Rows A–G differ slightly from Sub 4's table (e.g. hard E 0.7533 vs 0.7510), so compare rows within one notebook, not across notebooks.

**Diagnostics worth remembering**
- **Recall funnel (known queries, rand / hard):** true identity in `CAND_K` 98.6% / 95.1%; in SIFT-verified top-210 98.3% / 94.1%; kept by the `TOP_ID`/`TOP_L` filter **97.0% / 87.5%**. On hard CV the filter loses ~6.6 points of recall, so the regression flagged in Sub 4 is still there, and more SIFT candidates did not fix it.
- Losses (1 − MAP): rand 0.0865 = new matched to a known identity 0.0208, outside top-5 0.0191, retrieval miss 0.0181, rank 2–5 0.0173, NEW above true identity 0.0111. Hard 0.2174 = retrieval miss 0.0709, NEW above true identity 0.0483, rank 2–5 0.0446, outside top-5 0.0324, new matched to a known identity 0.0211.
- Test: mean predicted P(new) 0.311 (stage 2) vs 0.295 (stage 1), 0.298 (V3); the pooled CV share is 0.417. Predicted new@1 rate on test: 41.7% (stage 2), 41.1% (stage 1), 41.8% (V3).
- Top-1 agreement with V3 on test: 91.9% (stage 2), 93.3% (stage 1).

**Reproducibility warning:** the notebook has `DATA_DIR` hardcoded to one person's machine and needs the SIFT `.npz` files, the pair cache and `best_config_v2.json`. Make sure the cache and the config are committed or documented.

**Left open / ideas for the next person**
- Raise `TOP_ID` / `TOP_L` (next approach 1 in section 4): this is now the clearest known loss.
- Stage 2 is the main source of gain; see section 4 item 5 for making it more robust.

---

### Sub 7: V7 = extra RootSIFT evidence + CV-gated selection (Nandika, 24-11): LB 0.84870 (+0.0003 vs Sub 6; CV +0.0027 pooled)

**Notebook / git tag:** `submission7_24_11.ipynb` / tag to be added
**Based on:** Sub 6 (V4 pipeline). Submitted file = `submission.csv` written from `V7 = 's2x'` (extra-SIFT stage 2).

**What it does**
- **Extra SIFT pass** on the same provided keypoints (classical matching + RANSAC only): **RootSIFT** descriptors, ratio 0.85 + mutual NN, partial-affine RANSAC (5 px, 2000 iters). Per pair it records matches, inliers, **strict inliers** (ratio < 0.70), inlier **coverage** in both images (bounding-box share of all keypoints), and transform sanity (|log scale|, |rotation|, mean residual).
- Computed only for the `XM=80` best candidates per query (ranked by the V2 blend), zeros elsewhere. Cached separately in `sift_pair_cache_x.npz`; the original pair cache is untouched.
- **Identity table:** 15 new features per (query, identity): max/sum of inliers, strict inliers, matches, coverage-weighted inliers, coverage, transform stats of the best image, query-relative versions and a rank. P(new) model gets `q_x_inl_max` and `q_x_cw_max`.
- Same two stages as V4, retrained with these features: **J** = stage 1, **K** = stage 2 (transductive neighbours, built on J's probabilities).
- **Wider candidate view:** `TOP_ID` 30 → 80, `TOP_L` 10 → 20. **4** bagged LightGBM seeds (was 2). Everything else as Sub 6 (`CAND_LDA=60`, α per species 0.05 / 0.01).
- **Selection:** candidates are tried in a fixed order and one replaces the current choice only if it gains > 0.001 on **both** rand and hard CV. Order: s1 (default) → s2 → s1x → s2x → fuse_x → fuse_all. Result: **s2x**.

**What changed vs the previous submission and why**
- Sub 6's error analysis showed the remaining loss is mostly ranking quality, and SIFT was the most robust signal, so this adds more SIFT evidence rather than more candidates.

**CV results (same folds for every row within this notebook)**

| Row | Adds | rand | hard |
|---|---|---|---|
| A (V1) | raw cosine | 0.8347 | 0.6941 |
| B | + centre / PCA-whiten | 0.8802 | 0.6795 |
| C | + re-rank | 0.8814 | 0.6900 |
| D | + top-2 agg | 0.8840 | 0.7031 |
| E (V2) | + SIFT/RANSAC + viewpoint | 0.8967 | 0.7533 |
| F | learned ranker, tuned τ | 0.8985 | 0.7476 |
| G (V3) | + learned P(new) | 0.9009 | 0.7601 |
| H (V4 stage 1, Sub 6 recipe, `TOP_ID` 80) | + learned-space channels/candidates | 0.9078 | 0.7773 |
| I (V4 stage 2, Sub 6 recipe) | + transductive neighbours | 0.9138 | 0.7791 |
| J | H + extra SIFT evidence | 0.9112 | 0.7842 |
| K (**submitted**) | J + transductive neighbours | 0.9151 | 0.7866 |
| L | fuse J + K | 0.9136 | 0.7902 |
| M | fuse H + I + J + K | 0.9144 | 0.7888 |

Pooled rand+hard MAP@5: s1 0.8426, s2 0.8464, s1x 0.8477, **s2x 0.8508**, fuse_x 0.8519, fuse_all 0.8516. Sub 6's submitted model had 0.8481, so Sub 7 is about **+0.0027 pooled** (+0.0016 rand, +0.0040 hard).

**What worked / what did NOT work**
- **The CV gain did not show up on the public LB:** 0.84870 vs 0.84842 for Sub 6 (+0.0003), against +0.0027 pooled CV. The public LB is only 30% of test, so this is within noise, but there is no sign of a real step up either. Treat Sub 7 as roughly equal to Sub 6.
- **Extra SIFT evidence helps on both schemes and both stages (CV):** H → J +0.0034 rand / +0.0069 hard; I → K +0.0013 rand / +0.0075 hard. The gain is mostly leopard (hard: +0.008 to +0.010); hyena moves only +0.002 to +0.003.
- **Strict-ratio RootSIFT inliers carry the signal.** Mean `x_strict`: same identity 9.3 vs different 0.15 (rand), 2.4 vs 0.15 (hard).
- **Inlier coverage is weak:** mean `x_covq` 0.118 vs 0.089 (rand) and 0.086 vs 0.090 (hard, no separation). Not worth more effort.
- **Wider candidate pool alone did nothing:** H here (0.9078 / 0.7773) equals Sub 6's H (0.9085 / 0.7780). Retrieval misses fell (hard 508 → 277, rand 130 → 77), but "outside top-5" errors grew by about the same amount.
- **Fusion** had the best pooled CV (fuse_x 0.8519) but lost 0.0015 on rand vs s2x, so the gate rejected it. Not overridden by hand.
- **Not isolated:** the 4 LightGBM seeds and the `TOP_ID` / `TOP_L` change are mixed into every row, so their separate effect is unknown.
- **Slight selection optimism:** 6 candidates were compared on the same CV; the gate (> 0.001 on both schemes) limits but does not remove this.

**Diagnostics worth remembering**
- **Recall funnel (known queries, rand / hard):** true identity in `CAND_K` 98.6% / 95.1%; in SIFT-verified top-210 98.3% / 94.1%; kept by `TOP_ID`/`TOP_L` 98.2% / **93.2%** (Sub 6: 97.0% / 87.5%). Retrieval is no longer the bottleneck.
- **Losses of the submitted model (1 − MAP):** rand 0.0849 = outside top-5 0.0265, new matched to a known identity 0.0209, rank 2–5 0.0165, retrieval miss 0.0107, NEW above true 0.0103. Hard 0.2134 = outside top-5 0.0567, NEW above true 0.0498, rank 2–5 0.0471, retrieval miss 0.0387, new matched to a known identity 0.0212 (Sub 6: 0.2174).
- **Test:** mean predicted P(new) 0.313 (extra-SIFT stage 2), 0.297 (stage 1), vs 0.417 pooled CV share; new@1 rate 41.6%, in line with all earlier versions. Top-1 agreement with V3: 92.9%.
- **Cost:** the extra SIFT pass computed ~990k pairs in ~16 min on 10 threads (test 223k, rand 573k, hard 193k new; 380k hard pairs were reused from rand). Full notebook ~40 min.

**Reproducibility warning:** needs the original pair cache **and** `sift_pair_cache_x.npz` (rebuilt automatically if missing, about 16 min), the SIFT `.npz` files and `best_config_v2.json`. `DATA_DIR` / `PARENT_DIR` can be set by environment variable. Commit the caches or document how to rebuild them.

**Left open / ideas for the next person**
- Ranking errors (outside top-5, rank 2–5) and NEW placed above the true identity are still ~0.15 of the 0.21 hard-CV loss.
- Remaining SIFT ideas: ratio variants, full affine / homography channel, inlier count normalised by keypoint count, second RANSAC pass (see section 4 item 3).

---

### Sub 8: Per-species V4 + viewpoint coverage (TBD, 24-11): LB not recorded

**Notebook / git tag:** `submission8_24_11.ipynb` / tag to be added  
**Based on:** Sub 7 / V4 pipeline

**What it does**
- Adds **viewpoint-coverage features** (`cov_same`, `cov_adj`, `cov_nvp`, `q_top_cov`, `q_frac_cov`) and one prototype per (identity, viewpoint), rather than relying on one centroid that can blur left/right flank appearances.
- Uses a per-species router: the CV-selected strategy is **pooled models for hyena** and **separate models for leopard**.
- Retains the V4 learned identity ranker, learned P(new), WCCN candidate generation and transductive query-neighbour evidence.

**What changed vs the previous submission and why**
- The EDA found that left and right flanks are not reliably similar within an identity, and leopard has substantially less viewpoint coverage than hyena. The revision therefore represents viewpoint coverage explicitly and chooses the modelling strategy separately by species.

**CV results**
- Stage 1 (submitted): rand 0.9103 / hard 0.7847.
- Stage 2: rand 0.9161 / hard 0.7827. It gained +0.0058 on rand but lost −0.0020 on hard, so the notebook selected **stage 1** for `submission.csv`.
- The notebook reports pooled CV MAP@5 of 0.8475 (stage 1) → 0.8494 (stage 2), but the hard-CV regression prevented stage 2 from being selected.

**What worked / what did NOT work**
- Viewpoint-specific prototypes and coverage features were retained in the species-specific V4 stage 1.
- Transductive stage 2 improved rand CV but failed the hard-CV gate; do not treat its pooled gain alone as sufficient evidence.

**Left open**
- Test whether the per-species viewpoint strategy generalises to the test set; Kaggle score was not recorded in the supplied log/notebook.

---

### Sub 9: v6 = test-calibrated validation + new-share correction (TBD, 25-11): LB not recorded

**Notebook / git tag:** `submission9_25_11.ipynb` / tag to be added  
**Based on:** V4 pipeline, with validation redesigned following the audit of the earlier CV

**What it does**
- Replaces random/hard training validation with **test-like open-set folds**: whole new identities are held out, while known identities are split by burst clusters so near-duplicate encounters do not leak across gallery and validation.
- Uses three CV seeds; caps held-out new-identity size at 120 and targets a **0.30 new-individual share** (`PI_NEW=0.30`).
- Keeps the former hard split only as an evaluation-only stress test; it is not used to train the models.
- Estimates the test new-identity share from feature distributions and applies a conservative prior-shift correction (`PRIOR_SHIFT="auto"`) when the mixture fit is sufficiently good.
- Retains the V4 two-stage pipeline and selects stage 2 when it improves on every test-like seed.

**What changed vs the previous submission and why**
- The earlier random folds leaked near-duplicate images, while the previous hard folds were substantially harder than the inferred test distribution. The new burst-grouped folds aim to match test similarity statistics more closely.
- The new-identity share is controlled per species/fold instead of varying widely with the sampled identities. The 0.30 target follows the feature-only estimate of the test new share (~0.28–0.30).

**CV results**
- Test-like CV MAP@5: stage 1 0.9112 → stage 2 0.9145 (mean over three seeds).
- Stage 2 gains: s0 +0.0044, s1 +0.0028, s2 +0.0027; it was selected.
- Stress-only MAP@5: stage 1 0.7410, stage 2 0.7359. The stress split remains a lower-bound diagnostic, not the model-selection scheme.

**What worked / what did NOT work**
- The test-like validation and controlled new share provide a more representative basis for model selection than the previous random/hard mixture.
- Stage 2 clears the gain threshold on all three test-like seeds, but loses on the deliberately difficult stress split.

**Left open**
- Public LB score was not present in the supplied notebook; record it if available before comparing this submission with later ones.

---

### Sub 11: v7 = flank-coverage features (TBD, 26-11): LB 0.86063

**Notebook / git tag:** `submission11_26_11.ipynb` / tag to be added  
**Based on:** Sub 9 (v6 test-calibrated pipeline)

**What it does**
- Adds **flank-level gallery coverage** for each candidate identity. Viewpoints are grouped into left, right, and other/front-back/unknown; the candidate table receives `gn_fl` (number of gallery images from the query's flank group) and `fl_frac` (that count divided by the identity's gallery size).
- Adds these flank-coverage values to the learned identity-ranker features, alongside the existing exact-viewpoint coverage (`gn_vp`).
- Keeps the v6 test-like burst-grouped CV, 0.30 new-identity target, prior-shift correction, and two-stage transductive pipeline.

**What changed vs the previous submission and why**
- Exact viewpoint counts can be sparse. Grouping viewpoints by flank gives the ranker a coarser signal about whether a candidate identity is represented from a compatible side, without assuming left- and right-flank images are interchangeable.

**CV results**
- Test-like CV MAP@5: stage 1 0.9122 → stage 2 0.9166 (mean over three seeds).
- Stage 2 gains: s0 +0.0061, s1 +0.0030, s2 +0.0041; it was selected.
- Stress-only MAP@5: stage 1 0.7502, stage 2 0.7484. Stage 2 still loses slightly on the stress split.

**Kaggle result**
- Public leaderboard MAP@5: **0.86063**.

**Left open**
- Check whether the flank-coverage gain is consistent by species and under a more conservative validation setup; the CV gain is modest and the public LB score is the key external result.

---

### Sub 12: v8 = score-normalised similarities + neighbour P(new) + 4 seeds (Nandika, 27-11): LB 0.86690 (+0.0063 vs Sub 11)

**Notebook / git tag:** `submission12_26_11.ipynb` / `sub12-2026-11-27` (tag to be added)
**Based on:** Sub 11 (same folds, same 3 seeds, same pair cache; every CV number is directly comparable with Sub 11)

**What it does**
- **Score normalisation (Z-norm / T-norm style), no labels and no new SIFT.** Every query-gallery similarity is also expressed (a) relative to the gallery image's own impostor cohort (mean/std of its 50 most similar gallery images of *other* identities, `COH_K=50`) and (b) relative to the query's own similarity distribution over the gallery; both in the PCA space and in the learned (WCCN) space (`z_gc_max`, `z_qc_max`, `z_gl_max`, `z_ql_max` and their query-relative `_rel` versions).
- **Identity compactness:** `intra` (mean pairwise cosine inside the identity's gallery images, NaN for single-image identities) and `sc_vs_intra` = `sc_max - intra`.
- Query-level maxima of the four normalised scores (`q_zgc_max`, `q_zqc_max`, `q_zgl_max`, `q_zql_max`) are added to the P(new) model.
- **Neighbour P(new) in stage 2:** `nb_pnew_w` (similarity-weighted stage-1 P(new) of the query's nearest queries) and `nb_pnew_s` (mean over SIFT-verified neighbours, -1 if none), in both the identity ranker and the stage-2 P(new) model.
- **LightGBM seeds 2 -> 4** in every tree ensemble (`LGB_SEEDS`).
- Switches for ablation: `USE_ZNORM`, `USE_NBNEW`, `LGB_SEEDS`.

**What changed vs the previous submission and why**
- Raw cosine is not comparable across gallery images (hub images create false matches; tight and loose identities score differently), and the `new_individual` decision depends on exactly that. Stage 2 already passed neighbours' identity probabilities; their P(new) is the matching signal.

**CV results (same folds / seeds as Sub 11)**

| | Sub 11 | Sub 12 | Delta |
|---|---|---|---|
| Test-like, stage 1 | 0.9122 | 0.9133 | +0.0011 |
| Test-like, stage 2 (**submitted**) | 0.9166 | **0.9210** | +0.0044 |
| Stress, stage 1 | 0.7502 | 0.7525 | +0.0023 |
| Stress, stage 2 | 0.7484 | 0.7541 | +0.0057 |

- Stage 2 gains over stage 1: s0 +0.0087, s1 +0.0085, s2 +0.0060 (all > 0.002), stress +0.0015. Stage 2 selected.
- Ladder (mean over 3 seeds): A 0.8611, B 0.8652, C 0.8662, D 0.8674, E 0.8932, F 0.9059, G 0.9097, H 0.9133, I 0.9210.
- Per species (s0, stage 2): hyena 0.9023, leopard 0.9296 (all three seeds: hyena 0.902-0.917, leopard 0.918-0.930).

**Kaggle result**
- Public leaderboard MAP@5: **0.86690** (Sub 11: 0.86063, +0.0063). The public LB is 30% of test, so treat the size of the step with caution, but it moved in the same direction as CV (+0.0044).

**What worked / what did NOT work**
- Stage 2 carries the gain; stage 1 moved only +0.0011 (so the z-norm features alone are a small effect). The z features do appear in the held-out permutation importance (`z_gl_max_rel` 0.0008, `z_ql_max` 0.0006, `sc_vs_intra` 0.0004), but `v2_top2_rel` still dominates (0.0053).
- **Not isolated:** z-norm vs neighbour P(new) vs 4 seeds are all mixed into the +0.0044. Run with `USE_ZNORM=0` / `USE_NBNEW=0` / `LGB_SEEDS=2` to separate them.
- Prior-shift sensitivity is small (gain between -0.0005 and +0.0026 over test new-share 0.25-0.55), so the correction is not a real lever.
- The pair cache was not present on this machine, so the full SIFT pass was recomputed (~40 min); total notebook ~50 min.

**Diagnostics worth remembering**
- Prior-shift (`auto`): estimated test new share hyena 0.29 (band 0.28-0.32), leopard 0.26 (band 0.25-0.29); used hyena 0.302 (CV prior), leopard 0.29.
- Losses of the submitted model on s0 (1 - MAP = 0.0777): correct at rank 2-5 0.0202, outside top-5 0.0188, retrieval miss 0.0135, NEW above the true identity 0.0129, new matched to a known identity 0.0123. No single category dominates. Stress (0.2459): retrieval miss 0.0831, NEW above true 0.0725, rank 2-5 0.0477, outside top-5 0.0348.
- Hyena known queries are the weakest slice (s0: 0.883 stage 1 / hyena 0.902 overall stage 2); new queries are ~0.95-0.96.
- Recall ceiling for known queries: 98.1% / 98.2% / 97.7% (test-like seeds), 87.8% (stress).
- **Test vs CV P(new):** mean predicted P(new) on test 0.342 (V3) / 0.348 (stage 1) / 0.370 (stage 2), vs 0.303 true share in CV and a feature-only estimate of 0.26-0.32. New@1 rate on test 40.9% (CV s0: 34.6%). The model calls more test images "new" than CV would suggest; not acted on.
- Top-1 agreement with the V3 file on test: 92.4% (stage 2).

**Left open / ideas for the next person**
- Ablate the three additions (see above).
- Ranking errors (rank 2-5, outside top-5) are still ~0.04 of the 0.078 CV loss; hyena known queries are the weakest slice.
- Investigate the gap between predicted test P(new) (0.37) and the estimated test new share (0.26-0.32).

**Reproducibility warning:** needs the SIFT `.npz` files and `best_config_v2.json`; `sift_pair_cache.npz` is rebuilt automatically if missing (~40 min first run). `DATA_DIR` can be set by environment variable.

---

## 3. What has been tried (don't redo / don't undo)

| Idea | Result | Where |
|---|---|---|
| Centre + PCA-512 whiten | Big win on rand CV (+0.045), **hurts on hard CV (−0.015)**. Kept, but treat with suspicion | Sub 1, Sub 4 |
| DBA / query expansion on the embedding | Tried in Sub 1's greedy search; not in the final config, so it presumably did not help (outcome not recorded) | Sub 1 |
| k-reciprocal re-ranking | Tiny on rand (+0.001), +0.010 on hard. Kept | Sub 2/3/4 |
| Top-2 identity aggregation | +0.001 rand, +0.011 hard. Kept | Sub 2/3/4 |
| SIFT ratio 0.78 + mutual-NN + partial-affine RANSAC | Best robust gain (+0.014 rand, +0.050 hard). Kept | Sub 2+ |
| Hand-tuned viewpoint bonus | Included in V2 gain; not isolated | Sub 2 |
| Learned identity ranker (GBM ensemble) | +0.010 rand, ~0 on hard vs V2 | Sub 3/4 |
| Learned P(new) vs tuned τ | −0.002 rand in Sub 3, +0.001 rand / +0.006 hard in Sub 4. Marginal | Sub 3/4 |
| Query-to-query cue (embedding only) | +0.012 rand CV but disabled (CV/test query-count mismatch) | Sub 3 |
| WCCN projection + extra candidates | +0.006 rand, +0.022 hard. Kept | Sub 4 |
| Transductive stage 2 (SIFT-verified neighbours) | Sub 4: +0.005 rand, **−0.007 hard**, not used. Sub 6: +0.005 rand, +0.005 hard, **used** (LB 0.84842). Helps hyena, not leopard | Sub 4, Sub 6 |
| WCCN α grid down to 0.01, `CAND_LDA` 40 → 60 | α: ~+0.001 closed-set top-1 only (hyena 0.05, leopard 0.01). Stage 1 CV +0.001–0.002. Small | Sub 6 |
| RootSIFT + ratio 0.85 + strict inliers + inlier coverage / transform stats (as features, both stages) | +0.003 rand, +0.007 hard (stage 1); strict inliers carry it, coverage is weak. LB only +0.0003 (0.84870). Kept | Sub 7 |
| `TOP_ID` 30 → 80, `TOP_L` 10 → 20 | Hard-CV recall kept 87.5% → 93.2%, retrieval misses ~halved, but MAP flat (ranking errors grew). Kept (harmless) | Sub 7 |
| Probability fusion of V4 / extra-SIFT stages | Best pooled CV (0.8519) but −0.0015 rand vs s2x, rejected by the both-schemes gate | Sub 7 |
| 4 LightGBM seeds instead of 2 | Not isolated | Sub 7 |
| Burst-grouped CV | Started in Sub 1, never finished; replaced by hard folds | Sub 1/4 |
| Per-species viewpoint prototypes / coverage features | Added viewpoint-specific prototypes and coverage features; CV chose pooled hyena and separate leopard models | Sub 8 |
| Test-like burst-grouped CV + controlled new share (`PI_NEW=0.30`) | Replaces the rand/hard training mix; stage 2 improves all three test-like seeds, while stress remains evaluation-only | Sub 9 |
| Flank-level gallery coverage (`gn_fl`, `fl_frac`) | Added as identity-ranker evidence on top of exact-viewpoint coverage; stage 2 test-like CV mean 0.9166, public LB 0.86063 | Sub 11 |
| Score normalisation (impostor-cohort Z-norm + query T-norm, PCA and learned space), identity `intra`, neighbour P(new) in stage 2, 4 LGBM seeds | Stage 1 +0.0011, stage 2 +0.0044 test-like CV (0.9210), stress +0.0057; public LB 0.86690 (+0.0063). Contribution of each part not isolated | Sub 12 |

---

## 4. Template: copy this for every new submission

```markdown
### Sub N: <short name> (<your name>, <dd-mm>): LB <score> (<Δ vs previous>)

**Notebook / git tag:** `submissionN_dd_mm.ipynb` / `subN-yyyy-mm-dd`
**Based on:** Sub <N-1> (or something else: say what)

**What it does** (pipeline steps, key parameters, anything someone would need to reproduce it)
-

**What changed vs the previous submission and why**
-

**CV result(s)** (which CV scheme, same folds as before? rand / hard)
-

**What worked / what did NOT work** (including things you tried but didn't submit)
-

**Left open / ideas for the next person**
-
```