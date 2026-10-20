# Source audit before implementation

Reviewed 20 October 2026. Candidate: original-support-aware node-based
Hierarchical Shrinkage (HS) after oversampling. No verified novelty claim.

## Motivation and boundaries

Yan, Guo and Su, *Adaptive Weighting–Synthetic Minority Oversampling Technique*,
Mathematics 14 (2026), 3238, https://doi.org/10.3390/math14173238.
Uploaded PDF SHA-256:
`52565ba72dde51afaf7567a5553d9760bceca25fc66701d550b17a8f93a266b6`.
Section 4.7, printed pp. 19–20, Tables 9–10 compares AW-SMOTE with SMOTEWB.
AW-SMOTE has mean depth 12.5231 versus 11.6769, leaves 52.4308 versus
47.9692, F1 .9249 versus .9535. The authors discuss synthetic placement and
overlap. They do **not** establish inflated counts as the cause. This pilot
uses standard SMOTE to isolate the regularizer; it does not reproduce that
AW/SMOTEWB comparison and cannot claim to fix its structural complexity.
HS does not prune or simplify tree structure.

Prior clean/contaminated AW and NADOS reports and the Soft-NADOS feasibility
report were read. Their findings remain historical results. This candidate
changes classifier prediction regularization, not those samplers.

## HS paper and exact equation

Agarwal, Tan, Ronen, Singh and Yu (2022), *Hierarchical Shrinkage: Improving
the Accuracy and Interpretability of Tree-Based Methods*, ICML, PMLR 162.
Paper: https://proceedings.mlr.press/v162/agarwal22b.html
PDF: https://proceedings.mlr.press/v162/agarwal22b/agarwal22b.pdf
Supplied PDF SHA-256:
`1676da2f49a67762a9af8312d3890cf221a5d9dd97ab8cc45c5503060feebc80`.
Section 2, Equation (1), PDF p. 3:

`p_HS(x) = p(root) + sum_path (p(child)-p(parent))/(1+lambda/N(parent))`.

Each p is the **pre-shrinkage** node class-probability vector. The root is
unshrunk. It is the parent count, not the child or leaf count. The paper's
leaf-based alternative, Eq. (2), is a different method. Section 3 connects
ordinary HS to ridge regression on supervised stumps; that equivalence must
not be attributed automatically to this count substitution. Section 4 uses
held-out/CV parameter choice and class AUC; our imbalanced pilot selects AP.
Appendix theory/extra experiments do not specify an oversampled/original
support distinction.

## Author code inspection

Repository https://github.com/csinva/imodels, pinned HEAD
`0020b6a08671e2ccd9463581ac59f92e867bd8f8`.
Inspected file:
https://github.com/csinva/imodels/blob/0020b6a08671e2ccd9463581ac59f92e867bd8f8/imodels/tree/hierarchical_shrinkage.py
SHA-256 `aa4b63ba63d6a68404fe0c4c6b1ceacafdd35c65e7330a337b61a6c778cdfd40`.
The supplied master URL matched the pinned file byte for byte.
`HSTree._shrink_tree` reads `tree.weighted_n_node_samples[i]`, passes that
count to both children as `parent_num`, and uses the difference from the
original parent value. Its classifier handles normalized/count node storage.
The author `predict_proba` clips numerical roundoff and renormalizes.

The sampler's original-vs-synthetic provenance is not an input to that code.
Its `sample_weight` argument is passed to tree fitting, changing splits and
node predictions; zero-weighting synthetic rows there is not equivalent to
retaining their influence on structure/predictions and replacing counts only.
The code supports node-based, constant and leaf-based schemes; none is the
requested original-routing substitution. We implement Eq. (1) independently
and verify a standard-HS reference against the pinned method separately.
No author code is vendored into this repository.

## Exact-equivalent check and uncertainty

Inspected the supplied HS paper including related work (Hastie–Pregibon
recursive shrinkage, Bayesian CART mean-shift priors, leaf-based shrinkage),
the author implementation, and repository docs/tests for oversampling,
original/effective support, synthetic counts and sample weights. No exact
equivalent was found **in those inspected sources**. Related methods do not
establish equivalence or grounds for rejection.

Attempted searches: Google `"hierarchical shrinkage" "oversampling"` and
`"hierarchical shrinkage" "original" "synthetic"`, DuckDuckGo with HS and
oversampling, Bing with HS and synthetic, and OpenAlex/Crossref scholarly
queries. All search endpoints returned HTTP 403 in this environment.
GitHub API was also blocked; public git/raw access worked. Consequently the
broader literature search is incomplete. We cannot assert that an exact
equivalent does not exist, or advertise verified novelty. No exact duplicate
was discovered that would require stopping before this bounded feasibility
implementation. A successful pilot would still need external novelty review.

## Proposed difference and safeguards

Fit a single unweighted CART on fitting originals plus synthetic rows. Let
`N_orig(t)` count only supplied original fitting feature rows routed through
that fitted tree. Replace only `N(parent)` by `N_orig(parent)`. Keep identical
tree arrays, raw p vectors and root probability for both schemes. Every
original row counts once, including original duplicates. Synthetic seed or
partner ancestry never counts as an additional original visit.

For lambda>0 and zero original parent support, suppress that increment
(factor zero). For lambda=0, use factor one everywhere, so the unshrunk tree
is recovered even on synthetic-only branches. No pseudocount is inserted.
Counts along paths are nonincreasing; the shrink factors are also
nonincreasing, giving convex combinations of raw ancestor probabilities.
We check probability validity and tolerate only floating-point roundoff.

Uniform-inflation identity: if `N_aug(t)=c*N_orig(t)` for every relevant
parent, then `standard_HS(c*lambda) == original_HS(lambda)` with fixed
structure/p. Thus global stronger regularization alone is not a contribution.
Both schemes tune the same relative grid with lambda=relative*their root
count. This makes their entire grids coincide under uniform inflation;
differences can then reflect local count ratios rather than a global scale.
