## 2. Metric and a test-like open-set cross-validation (v7: mixed difficulty)

**What the test set looks like** (features only, no labels): about half of the test images have another *test* image as nearest neighbour, so test was built by holding out whole encounters plus whole new identities that appear several times. Its top-1 similarity to train is a *mixture*: a near-duplicate tail (2.5 % / 5.1 % above 0.8) and a large hard component.

**Validation folds (per species, K = 4 folds, every image validated at most once):**
1. *New identities*: random identities (≤ `CAP_NEW` images) are held out **entirely** into one fold until ≈ `PI_NEW` = 0.30 of the images are covered → label `new_individual`.
2. *Known identities*: every identity draws a burst threshold θ from `THETA_MIX`; its images are grouped into **burst clusters** (connected components of centred cosine > θ) and whole clusters are spread over folds, so no same-identity gallery image is above that identity's θ. θ = 0.5 ≈ "a different encounter", θ = 0.9 ≈ "same day, different photo".
3. Identities with a single burst stay in the gallery (never validated).

Schemes `m0`, `m1`, `m2` (three seeds) are used for training and validation. `s0` = the v6 folds (θ = 0.8 for everyone), **evaluation only**, to compare with the previous notebook. All folds are asserted leak-free below.
