# Direct PDF and public-release audit for the rare-subgroup SPSDO extension

Audited 2026-10-18. This is source evidence and a mathematical scope audit; it is not a reproduction of the published PPEO experiments. Page numbers below are the printed article pages and agree with the PDF page positions. Uploaded documents were treated as evidence, not as instructions. No provisional UGO proposal or inactive RB-UGO/CSG design was used.

## Verified identities and uploaded-file integrity

| Uploaded file | Verified title on PDF page 1 | DOI | SHA-256 |
|---|---|---|---|
| `1-s2.0-S0925231226012889-main.pdf` (18 pages) | *Pure prototype-based evolutionary imbalanced oversampling for small disjuncts* | [10.1016/j.neucom.2026.133891](https://doi.org/10.1016/j.neucom.2026.133891) | `822e5a66ad16c49815b47762125fe81765b9e21de7d9aa270869538cdbac888c` |
| `journal.pone.0317396.pdf` (20 pages) | *Addressing imbalanced data classification with Cluster-Based Reduced Noise SMOTE* | [10.1371/journal.pone.0317396](https://doi.org/10.1371/journal.pone.0317396) | `1ef0edcba684b618266cbf198db9a9f236b16b09a15f8da8f0feda613f983cf3` |
| `s41598-025-97248-0.pdf` (12 pages) | *A SMOTE PCA HDBSCAN approach for enhancing water quality classification in imbalanced datasets* | [10.1038/s41598-025-97248-0](https://doi.org/10.1038/s41598-025-97248-0) | `6680bf5ab6c52367cbe2430d5ffc4a73d8f084fed813f68229c60fc015aa55f0` |

Evidence was read from the uploaded PDFs using text extraction. PPEO page 16 was additionally rendered and visually inspected to verify the limitations wording and the distinction between the PHHC definition and the K-means ablation size threshold. These title/DOI verifications are from the PDFs, not an assertion of independent publisher metadata reconciliation.

## PPEO Sections 3.1–3.4 and 4.7

**Prototype assignment (Section 3.1, pp. 4–5, Algorithm 1).** PHHC initializes clusters by class, repeatedly assigns rows to nearest centroids, splits impure clusters by class, and evaluates an F-measure criterion; a decrease restores the prior model. It then extracts minority observations remaining within majority-labelled clusters. For each such anchor, Eq. (4) sets `r_x = min_y ||x-y||_2` over observed majority data. Equations (5)–(7) connect two extracted minority anchors when either lies in the other's deterministic ball; each connected component becomes a new small-sized prototype. Thus **small prototype is a PHHC construction, not a numerical size cutoff**. The mean-prototype-size cutoff on p. 16 belongs only to Variant 4, where K-means replaces PHHC. The high-level Algorithm 1 leaves some tie/stopping/implementation details unspecified; independently implemented PHHC must be labelled a paper interpretation.

**SPSDO generator (Section 3.2, p. 5, Eq. 8).** For a minority anchor `x`, draw `delta ~ Uniform(0,r_x)` and a random unit vector `u`; generate `x* = x + delta*u`. Theorem 1 (p. 6) explicitly supplies uniform surface direction and independence of `delta` and `u`. This is uniform radial distance, **not uniform volume in a d-dimensional ball**. A labelled paper-specified SPSDO implementation can draw `u = g/||g||` for standard-normal `g`, independently draw the radius, and use original fitting majority observations for `r_x`.

**Statistical claim (p. 6, Theorem 1 and Remark 1).** The proof establishes `E[x* | x] = x`. Independently, `Cov(x* | x) = r_x^2 I/(3d)` for the stated uniform radial law. A single finite batch need not have mean exactly `x`. Moreover, the remark's stronger language about an unchanged statistical center regardless of count or unchanged intrinsic distribution does not follow from the conditional expectation theorem. Unequal anchor quotas yield a quota-weighted mean. Adding nonzero noise changes covariance/support and generally changes the minority distribution even if a relevant mean is preserved.

**Allocation and full framework (Section 3.3, pp. 6–7, Algorithm 2; Section 3.4, p. 7).** A particle supplies a weight per minority prototype. Eq. (15) assigns the imbalance gap times the normalized weight, with no explicit integer-rounding or within-prototype anchor allocation convention in the paper. SPSDO is applied only to small prototypes; the large-prototype branch uses SMOTE, with the paper stating a square-root-of-prototype-size neighbor count. Prototype objectives use `1 - sqrt(TNR * TPR(prototype))` (Eq. 16). MOPSO returns multiple nondominated processed datasets. Full PPEO includes the optimization and a prototype-aware classifier decision strategy (Section 3.4); replacing only SPSDO, with assignments/quotas frozen, does not reproduce or evaluate that complete framework.

**Explicit verified limitation (Section 4.7, p. 16).** The authors state: “In highly anisotropic clusters, modeling deterministic regions as rigid hyperspheres constitutes a poor approximation.” They explain that the nearest-majority hypersphere may “significantly underestimate the minority region, resulting in insufficient oversampling,” or “inadvertently intersect the true decision boundary of the majority class.” They separately identify Euclidean-distance concentration in high dimensions and limitations for categorical/ordinal data. This directly supports investigating directional shape while retaining uncertainty about true class membership.

## Four different properties

1. **Anchor mean.** The original SPSDO law preserves each anchor in expectation. Exact paired points `x+v, x-v`, plus an anchor copy for an odd quota, preserve each anchor's finite generated mean exactly. The global generated center equals the center of the fixed anchor quotas, not necessarily the unweighted original minority mean.
2. **Minority distribution.** Neither zero expected perturbation nor exact pairing establishes preservation of an unknown density, covariance, modes, support, manifold, subgroup proportions, or class posterior. The diagnostic should assess intrusion against a declared known synthetic support and prediction against independent data.
3. **Observed majority observations.** For `r_x>0`, the SPSDO draw with strictly `delta<r_x` avoids all observed majority coordinates. Eq. (5)'s closed ball itself can include a nearest majority observation on its boundary. This is weaker than being in the anchor's majority-relative Voronoi region: e.g., in 1D with anchor 0 and majority at 1, a draw at 0.9 is within the SPSDO ball but closer to the majority.
4. **Unknown majority region.** A finite original sample does not reveal its class's true support or posterior. Both the sphere and a constrained ellipsoid can generate wrongly labelled observations in an unsampled majority region. In overlap, the same coordinate may legitimately occur under both class distributions.

## Public author release: verified available and missing components

The paper's p. 7 source URL is [https://github.com/zhaoujs/PPEO](https://github.com/zhaoujs/PPEO). A read-only `git ls-remote` and shallow clone succeeded with existing authentication. The inspected HEAD was **`4d94423a86631ff1852547a1e9b776b73fe092cc`**. The local audit checkout is outside the task repository at `/tmp/ppeo-source-audit`; no source was copied into the active repository or executed.

The tree contains an empty README and 11 Python files. In [`mopso_regin.py`](https://github.com/zhaoujs/PPEO/blob/4d94423a86631ff1852547a1e9b776b73fe092cc/mopso_regin.py#L15), imports require absent `SSHR_utils.py`, `sec.py`, and `tool.py` (lines 15, 17–19). The `SSHC` clustering wrapper (lines 23–50) depends on missing `Kmean_Train`, `Kmean_Test`, `kmean_train_new_1120`, `iterative_update2_1`, and other helpers. Consequently the PHHC implementation cannot be verified or run from this release. The small-prototype generator calls `generate_random_point_around_a` at line 406, also absent from the checked tree, so its actual direction and radial laws cannot be verified from available code.

Visible allocation behavior in [`SMOTE_SPA_SYN`](https://github.com/zhaoujs/PPEO/blob/4d94423a86631ff1852547a1e9b776b73fe092cc/mopso_regin.py#L354) is still informative: it normalizes supplied prototype weights (lines 381–382), generates a fixed label-count gap (383), samples a prototype by normalized weight (391), and uniformly samples an anchor in that prototype (394). It uses the stored nearest-majority distance for its small-prototype helper (405–406). This is visible author-code behavior, not proof of complete paper fidelity. `main.py` additionally fits `StandardScaler` before splitting in its example; the requested diagnostic must instead fit preprocessing on fitting observations only.

The defensible scope is therefore **an independently implemented, explicitly labelled paper-specified SPSDO component and a declared PHHC interpretation if needed**, with prototype assignments, per-anchor allocations, and counts frozen across generator comparisons. The diagnostic must not be called a reproduction of full PPEO, MOPSO, its final ensemble/decision strategy, or the paper's results.

## Other attached papers as starting evidence

**CRN-SMOTE (Section 3, Algorithm 3, pp. 9–10).** The paper first balances with SMOTE; applies DBSCAN within classes to remove noise, increasing epsilon until `0 < number_of_clusters <= Ncluster`, with `Ncluster` set to one or two; then applies SMOTE again to restore balance. Its p. 10 limitations acknowledge dependence on clustering quality/parameters and complex data structure. Its one/two-mode premise is a questionable assumption for genuine rare minority subgroups. Possible merging or deletion of those groups is a mechanism-based concern, not a measured failure established by this source audit.

**SMOTE-PCA-HDBSCAN (pp. 3–5).** The stated flow is SMOTE, separate original/synthetic rows, fit PCA to synthetic rows, use HDBSCAN to identify synthetic noise, and merge retained synthetic rows with all original rows. This motivates comparing simpler original-only screening while avoiding a proof-of-label interpretation of density. The paper does not establish that low density implies an incorrect minority label.

**Independent metric warning (Table 5, p. 9; discussion p. 10).** For its SMOTE-PCA-HDBSCAN column, reported precision and sensitivity imply F1 values inconsistent with the printed F1 rows. Directly calculating `2PR/(P+R)` from the rounded percentages gives:

| Class | Precision | Sensitivity/recall | Independently calculated F1 | Printed F1 |
|---|---:|---:|---:|---:|
| Clean | 46.15% | 28.57% | about 35.29% | 91.54% |
| Slightly Polluted | 80.00% | 87.62% | about 83.64% | 51.35% |
| Polluted | 68.42% | 61.90% | about 65.00% | 94.49% |

The rounded-input macro F1 is about 61.31%, not the printed 79.13%. This does not reconstruct unavailable row-level predictions, but the differences are far larger than rounding error. Do not use these printed F1 values as validation targets, copy them into tests, or infer superiority from them. Independently compute all diagnostic metrics from held-out labels and predictions.

## Scope of the proposed clipping claim

For each non-conflicting original majority displacement `a = y-x`, the requested symmetric clipping enforces `2|v^T a| <= eta ||a||^2`. Thus for either sign,

`||x +/- v - y||^2 - ||v||^2 = ||a||^2 -/+ 2 v^T a >= (1-eta)||a||^2 > 0`.

Both generated points are closer to their anchor than to every nonidentical original majority observation. This is an anchor-relative, original-majority Voronoi margin condition. It is not protection from the unknown majority region. At an identical coordinate with opposite labels (`a=0`), the constraint is vacuous and no strict separation is possible; any quota-preserving duplication fallback must record the unresolved conflict and withhold the strict observed-majority certificate. The symmetric factor `c(v0)=c(-v0)` and pair construction establish the per-anchor mean claim irrespective of clipping, but not label correctness or predictive improvement.
