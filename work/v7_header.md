# Wildlife Re-ID — v7: test-like *mixed-difficulty* validation + RootSIFT + shift-robust transductive stage

Predict up to five identities per test image (a known identity of the same species, or `new_individual`), scored by MAP@5, using only the provided MiewID and SIFT features.

| Version | What it is |
|---|---|
| **V1** | raw MiewID cosine + tuned `new_individual` threshold |
| **V2** | V1 + PCA/whitening, re-ranking, SIFT+RANSAC, viewpoint bonus (hand-tuned blend) |
| **V3** | learned identity ranker (gradient-boosted trees on pairwise evidence) + learned `P(new)` |
| **V4** | V3 + transductive evidence from neighbouring test queries |
| **V7** | V4 retrained on mixed-difficulty folds, RootSIFT matching, transductive stage without pool-size statistics, decision-theoretic list construction (each chosen by CV) |

## Why v7: the gap between CV and the leaderboard
v6 scored **0.915 MAP@5 on its own CV folds but 0.858 on the leaderboard**. The gap is the main thing to fix: a model selected and calibrated on folds that are easier than the test set makes its `new_individual` decisions for the wrong difficulty.

Findings of the error/shift analysis (`work/eda_*.py`, reproduced in section 2a):

1. **The test set is a mixture of difficulties.** Its top-1 similarity to train has a heavy upper tail (2.5 % hyena / 5.1 % leopard of test images above 0.8, vs 0.2 % / 1.1 % in the v6 folds) *and* more mass in the hard middle range (leopard median 0.454 vs 0.515). One burst threshold θ cannot reproduce both. A non-negative mixture fit of the test distribution on CV components (new queries + known queries at θ ∈ {0.5,…,0.95}) fits within 0.02 (max CDF error), with ≈ 20 % "different encounter" (θ = 0.5), ≈ 15 % θ = 0.7, ≈ 40 % θ = 0.8 and ≈ 25 % near-burst (θ = 0.9) known queries.
2. **v6 trains and calibrates only on the θ = 0.8 slice**, so on the test set its `P(new)` is too high for the hard known queries (mean test P(new) 0.363 vs an estimated new share of ≈ 0.30).
3. **Stage-2 pool statistics shift.** The query–query similarity features (`q2q1`, `q2q3`, `nn_sim*`) depend on the size and composition of the query pool (test pool: 1217 hyena / 1574 leopard images; a CV fold: ≈ 480 / ≈ 1300).
4. **SIFT matching can be stronger.** A paired experiment on 16 k train pairs (`work/exp_rootsift.py`): RootSIFT improves the same-vs-different AUC of RANSAC inliers (hyena 0.586 → 0.603, leopard 0.691 → 0.697) and the top-1 accuracy of "most inliers" (hyena 0.750 → 0.766, leopard 0.913 → 0.935) at identical ratio-test/RANSAC settings. Other variants (looser ratio, full affine, homography) were not consistently better.

## What changed in v7
| # | Change | Where |
|---|---|---|
| 1 | **Mixed-difficulty test-like folds**: every identity draws its own burst threshold from {0.5: 0.2, 0.7: 0.15, 0.8: 0.4, 0.9: 0.25}; all models are trained and validated on these (3 seeds). The v6 folds (θ = 0.8) are kept as an **evaluation-only** scheme `s0` | §2, §2a |
| 2 | **RootSIFT** descriptors (L1-normalise + sqrt = Hellinger kernel) for every SIFT/RANSAC match | §6 |
| 3 | **Stage 2 without pool-size statistics** (V4c), selected unless it costs > 0.002 CV MAP@5 | §7c |
| 4 | **List construction**: `new_individual` is inserted by comparing P(new) with *normalised* identity probabilities when CV says so (+0.001 required) | §7d |

Everything else (projection, re-ranking, ranker features, out-of-fold training, leakage audits, prior-shift guard) is unchanged from v6, so every gain in the ablation table can be attributed to one change.

**Run.** `DATA_DIR` = folder with `train.csv`; the SIFT folders/`.npz` next to it. First run computes ≈ 4 M RootSIFT pairs (≈ 1.5–2 h on 22 threads), later runs read `cache/sift_pair_cache_root.npz`.
