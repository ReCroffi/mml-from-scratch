# MML from scratch

**🌐 Language:** [Português](README.md) · **English**

> From-scratch implementation (NumPy) of PCA and linear regression, applied to a real dataset —
> capping off the *Mathematics for Machine Learning* specialization (Imperial College / Coursera).

> ⚠️ **Work in progress.** PCA is implemented, validated against scikit-learn, and covered by
> automated tests (`pytest`). Regression and the SVD comparison are the next steps
> (see [Pipeline](#pipeline)).

## Goal

Implement PCA (and, next, linear regression) **by hand**, in pure NumPy — covariance matrix,
eigendecomposition, projection, and reconstruction — without calling `sklearn.decomposition.PCA`.
The point isn't dimensionality reduction (any library does that in one line); it's **understanding
the underlying math** and proving the implementation is correct by validating every result against
scikit-learn.

## Dataset

**Wine** (`sklearn.datasets.load_wine`): 178 samples, 13 numeric features, 3 wine classes
(~59 samples each — well balanced).

Chosen deliberately for the **disparate feature scales**, which make standardization
non-negotiable before PCA:

| feature   | standard deviation | variance (≈ σ²) |
|-----------|-------------------:|----------------:|
| `proline` |              ~315  |         ~99,000 |
| `magnesium` (2nd) |        ~14  |            ~200 |
| the rest  |          0.1 – 2.3 |             < 6 |

PCA chases **variance**, and variance depends on the unit of measurement. Without standardizing,
the first principal component ends up nearly collinear with the `proline` axis — not because it
explains the wine, but because its numbers (mg/L, in the thousands) are large. Standardizing
(`(X - μ) / σ`) removes that unfairness: every feature gets variance 1 and PCA compares them on
equal footing.

## Pipeline

| # | Step | Status |
|---|------|--------|
| 1 | Load + standardize by hand (`normalize`) | ✅ |
| 2 | PCA by hand: covariance → eigendecomposition → projection (`cov_matrix`, `eig`, `PCA`) | ✅ |
| 3 | Reconstruction (`reconstruct`) — inverse of the projection | ✅ |
| 4 | Validate against sklearn — automated tests in `tests/test_pca.py` (`normalize` centers · components match sklearn · `reconstruct` roundtrip) | ✅ |
| 5 | Linear regression via gradient descent on the reduced data | ⬜️ |
| 6 | Compare against the closed-form solution (normal equation) | ⬜️ |
| 7 | SVD + comparison with the eigendecomposition (ill-conditioned case) | ⬜️ |

## Results

Cumulative explained variance (verified — matches `sklearn` to the 4th decimal):

| components | cumulative variance |
|------------|--------------------:|
| PC1        |              36.2 % |
| PC1 + PC2  |              55.4 % |
| PC1 + PC2 + PC3 |         66.5 % |
| ~10 of 13  |              ~96 % |

That is: reducing from 13 → 2 dimensions already preserves more than half of the variance —
enough for a 2D scatter colored by class.

<!-- TODO: add the cumulative explained-variance plot + the 2D scatter by class once the
     analysis notebook is ready. -->

## The math behind it

<!-- Renan: this content was assembled from what YOU argued during development. Rewrite it in
     your own voice and make sure you can defend each point out loud — this is what comes up in
     interviews. Don't memorize my phrasing. -->

- **Why the covariance matrix, and not the raw data?** Because it is **symmetric** and
  **positive semidefinite (PSD)**. Symmetry gives **real** eigenvalues and **orthonormal**
  eigenvectors (via `np.linalg.eigh`); PSD gives **non-negative** eigenvalues (≥ 0) — which makes
  physical sense, since each eigenvalue is a variance, and variance can't be negative.

- **What are the eigenvectors and eigenvalues here?** The **eigenvectors** are the directions of
  maximum variance (the principal components, mutually orthogonal); the **eigenvalues** say *how
  much* variance lies along each direction. The sum of the eigenvalues equals the trace of the
  covariance matrix — total variance is **conserved**, PCA just redistributes it across orthogonal
  axes.

- **Why is `Xnᵀ @ Xn` the covariance?** With the data already centered, each `(i,j)` entry of that
  product sums the products of the deviations of features `i` and `j` over all samples — the very
  definition of covariance; dividing by `N-1` turns the sum into a mean.

- **Sign ambiguity.** Eigenvectors are defined up to sign (both `v` and `-v` are valid). That's
  why the scores match sklearn's *in magnitude*, sometimes with a flipped sign — not a bug, a
  property. Validation tests must be robust to this.

<!-- TODO (after implementing): gradient descent vs. normal equation — when to use each. -->

## How to run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

To run the tests:

```bash
python -m pytest -v
```

The tests validate PCA against `sklearn` (`sklearn.decomposition.PCA`) and check
math properties of their own (centering and reconstruction roundtrip).

## Structure

```
mml-from-scratch/
├── src/         # testable implementations (pca.py; regression.py coming soon)
├── notebooks/   # the narrative: EDA → PCA → regression → comparison
├── tests/       # validation against sklearn
└── data/        # dataset (Wine ships with sklearn; folder reserved)
```

## What I learned

<!-- Renan: 3-4 HONEST bullets, in your own voice. Pull from what you actually stumbled on in
     these sessions. Topic suggestions that produced good "aha" moments (write them yourself):
     - the difference between standardizing (scale) and centering, and why Wine needs both
     - why `eigh` guarantees REAL eigenvalues, but non-negativity comes from covariance being PSD
       (different sources — don't conflate them)
     - the ddof choice (1 vs 0) and how it explains the tiny difference when comparing to StandardScaler
     - why SVD is more stable than the eigendecomposition of the covariance (squaring the data
       amplifies rounding error) — to be filled in when the SVD is implemented -->
