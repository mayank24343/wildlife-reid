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

No Sub 5 entry yet (the notebook for it wasn't in the folder when Sub 6 was logged). Add it above Sub 6 if it exists.

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

---

## 4. Possible next approaches

Ordered roughly by expected value per effort. Test each on **both** rand and hard CV before submitting. These are suggestions; confirm any rule-sensitive ones with the TA.

1. **Fix the candidate-recall regression (cheap, no new SIFT).** `TOP_ID=30` / `TOP_L=10` cap what the learned model can see; recall ceiling dropped vs Sub 3. Try `TOP_ID` 60–100, `TOP_L` 20. **Done in Sub 7 (`TOP_ID` 80, `TOP_L` 20):** hard-CV recall kept rose 87.5% → 93.2% but MAP did not move, because the recovered identities are not ranked into the top 5. Retrieval is no longer the bottleneck; don't spend more time here.
2. **Re-tune the projection.** α=0.1 is at the edge of the grid for both species, and hyena's accuracy was still rising as α shrank (0.8612 → 0.8889). α ∈ {0.05, 0.02, 0.01} was tried in Sub 6 and gave almost nothing. Still to try: LDA as a projection, PCA dim 256 / 768, and partial whitening (power < 1), since full whitening hurt on hard CV.
3. **Strengthen the SIFT evidence** (the most robust signal so far). **Partly done in Sub 7:** RootSIFT, ratio 0.85, strict-ratio inliers, inlier coverage and transform stats (+0.003 rand / +0.007 hard; coverage itself was weak). Still to try: more RANSAC iterations, full affine or homography as an extra channel, inlier count normalised by keypoint count, and a second RANSAC pass.
4. **Ranking objective.** Swap the binary GBM classifier for a LightGBM `lambdarank` ranker grouped per query (directly optimises ranking). Keep the P(new) model separate.
5. **New-individual handling** (≈0.035 of MAP lost on rand, ≈0.06 on hard):
   - Reuse Sub 1's mixture-fit to estimate the real test new-share and compare with CV's 40%; if far off, re-weight or re-tune `NEW_QUERY_FRAC`.
   - Cluster test queries into groups with SIFT-verified query-query edges (graph / connected-components clustering is allowed). The statement says the same new individual can appear in several test images, so group members should get consistent decisions.
   - Hypothesis for why stage 2 failed on hard CV: CV folds have ~1,430 queries vs 2,791 in test, so neighbour structure differs. Try CV with fewer folds (bigger query sets) or subsampling test to fold size for the neighbour features.
6. **Hyena-specific tuning.** Hyena is clearly weaker on hard CV. Try per-species values of k1/k2, α, TOP_ID and separate models per species.
7. **Cheap ensemble.** V3 and V4-s1 disagree on top-1 for ~7% of test queries. **Tried in Sub 7:** probability fusion of the V4 and extra-SIFT stages gave the best pooled CV (0.8519) but lost on rand, so it was not adopted; the 4-seed LightGBM bagging was not isolated. Not a likely source of a big gain.
8. **Small learned embedding (higher effort, medium risk).** An MLP from scratch on the 2152-d features (≤1M params total) used as an additional similarity channel. Only worth it if 1–7 plateau.
9. **Pseudo-labelling confident test queries into the gallery.** Same spirit as the transductive re-ranking we already use, but check with the TA first since it isn't spelled out in the Problem 3 rules.

---

## 6. Template: copy this for every new submission

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