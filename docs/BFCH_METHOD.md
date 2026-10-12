# BFCH-SMOTE implementation notes

Source: H. Wen et al., “BFCH-SMOTE: Boundary fusion convex hull synthetic
minority oversampling technique,” *Neurocomputing* 702 (2026) 134595,
DOI: 10.1016/j.neucom.2026.134595, Sections 3.1–3.4, Algorithms 1–4.

`src/bfch_smote.py` provides `BFCHSMOTE.fit_resample(X, y)` for finite,
numerical, binary fitting data. The default requested count is the difference
between original majority and minority counts; `n_samples` can override it.
The sampler never receives held-out rows. It returns the original rows first,
followed by generated minority rows. A diagnosed no-op is returned if its
geometric criteria yield no valid hulls; it does not substitute SMOTE.

The stages follow the paper's method:

1. Fit a random forest on the fitting partition; retain minority/majority
   anchors whose own-class probabilities meet `theta_min=0.3` and
   `theta_maj=0.8`.
2. Give each retained minority anchor its `k_neighbors` closest retained
   majority anchors. Keep majority anchors selected by at least two distinct
   minority anchors. Reverse the search and keep minority anchors selected by
   at least two boundary majority anchors.
3. Pair each boundary minority anchor with its nearest other boundary
   minority anchor; deduplicate reciprocal pairs. For each midpoint `m`,
   find its nearest boundary majority anchor `b`, then set the fusion point
   `f=(1-L)m+Lb` with `L=fusion` (default 0.5).
4. Reject zero-area triangles and those containing any **observed original**
   majority point. Allocate the requested samples in proportion to triangle
   area with the largest-remainder rule. Draw uniform barycentric weights
   within each retained triangle.

Explicit implementation choices and limits:

- The paper supplies thresholds but not a definitive `k` or random-forest
  specification. Here `k_neighbors=5`, 100 trees, `class_weight="balanced"`,
  a fixed seed, and in-sample own-class probabilities are declared defaults.
  These choices can affect which anchors survive.
- The paper's Eq. 4–7 rank test checks affine-span membership, not convex
  triangle membership. We use barycentric coefficients, nonnegative weights,
  and an affine-plane residual check instead. This is a mathematical repair,
  not a claim that the paper's printed test works as stated.
- The paper validates against boundary majority anchors. We check every
  *observed original* majority row, a more conservative local rule. It cannot
  exclude overlap with unseen majority observations or the true distribution.
  In high dimensions most points lie outside a given triangle's two-dimensional
  affine plane, so the observed-point check becomes less informative.
- Triangle area uses a Gram determinant, algebraically the same area as
  Heron's formula. We omit degenerate triangles. Reciprocal nearest pairs
  form one hull. Uniform Dirichlet(1,1,1) barycentric weights specify the
  paper's otherwise unspecified random-simplex distribution.
- Numerical geometry after one-hot encoding is operationally possible but
  may interpolate categorical indicators into noncategorical values. Use
  numerical datasets first, and interpret encoded-categorical comparisons
  cautiously.

## Run a small paired comparison

The existing ten-fold runner accepts `bfch_smote` through `--methods` and
retains its old default method set. Supply a local binary dataset and an
explicit source-domain positive label. For example, with a suitable CSV:

```powershell
.\.venv\Scripts\python.exe main_stage4r5.py --dataset data\raw\my_binary.csv --target class --positive-label positive --output results\bfch-my-binary --classifiers LogReg --methods nores smote ugo_smote bfch_smote
```

Use the **same input file, preprocessing, folds, classifier, and seed** for
all methods in one run. Results under `results/` are local and ignored by Git.
The example is a command template, not a completed benchmark result.
