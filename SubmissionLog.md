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
| Transductive stage 2 (SIFT-verified neighbours) | +0.005 rand, **−0.007 hard**. Not used | Sub 4 |
| Burst-grouped CV | Started in Sub 1, never finished; replaced by hard folds | Sub 1/4 |

---

## 4. Possible next approaches

Ordered roughly by expected value per effort. Test each on **both** rand and hard CV before submitting. These are suggestions; confirm any rule-sensitive ones with the TA.

1. **Fix the candidate-recall regression (cheap, no new SIFT).** `TOP_ID=30` / `TOP_L=10` cap what the learned model can see; recall ceiling dropped vs Sub 3. Try `TOP_ID` 60–100, `TOP_L` 20. If that helps, then raise `TOP_M_SIFT` / `CAND_LDA` (needs new SIFT pairs, cache covers current ones).
2. **Re-tune the projection.** α=0.1 is at the edge of the grid for both species, and hyena's accuracy was still rising as α shrank (0.8612 → 0.8889). Add α ∈ {0.05, 0.02, 0.01}. Also try LDA as a projection, PCA dim 256 / 768, and partial whitening (power < 1), since full whitening hurt on hard CV.
3. **Strengthen the SIFT evidence** (the most robust signal so far). Ideas: RootSIFT normalisation of the provided descriptors, ratio 0.80–0.85 with more RANSAC iterations, full affine or homography as an extra channel, inlier count normalised by keypoint count, inlier spatial spread / coverage as features, and a second RANSAC pass.
4. **Ranking objective.** Swap the binary GBM classifier for a LightGBM `lambdarank` ranker grouped per query (directly optimises ranking). Keep the P(new) model separate.
5. **New-individual handling** (≈0.035 of MAP lost on rand, ≈0.06 on hard):
   - Reuse Sub 1's mixture-fit to estimate the real test new-share and compare with CV's 40%; if far off, re-weight or re-tune `NEW_QUERY_FRAC`.
   - Cluster test queries into groups with SIFT-verified query-query edges (graph / connected-components clustering is allowed). The statement says the same new individual can appear in several test images, so group members should get consistent decisions.
   - Hypothesis for why stage 2 failed on hard CV: CV folds have ~1,430 queries vs 2,791 in test, so neighbour structure differs. Try CV with fewer folds (bigger query sets) or subsampling test to fold size for the neighbour features.
6. **Hyena-specific tuning.** Hyena is clearly weaker on hard CV. Try per-species values of k1/k2, α, TOP_ID and separate models per species.
7. **Cheap ensemble.** V3 and V4-s1 disagree on top-1 for ~7% of test queries. Try rank/probability fusion of V3, V4-s1 and V4-s2 (simple model combination is allowed), and average a few GBM seeds. Check on both CVs.
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