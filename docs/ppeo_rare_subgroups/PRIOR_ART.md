# Focused prior-art audit for symmetric covariance/Voronoi SPSDO extension

Date: 2026-10-18. Scope: compare mechanisms; do not claim an exhaustive novelty search. Read `/workspace/rb_ugo/AGENTS.md`. No existing tracked file was edited by this audit. Local source checkouts are in `/tmp`; they are evidence, not added dependencies.

## Access and verification status

Primary author implementation repositories on GitHub were accessible. HTTPS requests to `doi.org`, `api.crossref.org`, `arxiv.org`, `www.sciencedirect.com`, `api.openalex.org`, `scholar.google.com` and `www.google.com` failed at the configured proxy with `CONNECT 403 Forbidden`; TLS verification was not disabled. Consequently exact paper page references were not available for the external precedents below. Line references refer to retrieved source snapshots and are explicitly distinct from paper verification. In particular, EMDO's primary paper, exact title, DOI and detailed constraints were **not independently verified**; do not fill those gaps with a guessed citation. This prevents a positive originality claim.

## Verified author-source comparisons

### G-SMOTE

Author repository: <https://github.com/georgedouzas/imbalanced-learn-extra>, commit `9e523b84af629f5a7dc85c419528d835a5774bce`.

- The author README lines 104–108 identify **Geometric SMOTE: a geometrically enhanced drop-in replacement for SMOTE**, G. Douzas and F. Bacao, *Information Sciences* 501 (2019), 118–135; DOI **10.1016/j.ins.2019.06.007**. This title/DOI association is verified in the author README; the DOI landing page was not reachable.
- Direct source: <https://github.com/georgedouzas/imbalanced-learn-extra/blob/9e523b84af629f5a7dc85c419528d835a5774bce/src/imblearn_extra/gsmote/geometric_smote.py#L26>.
- `make_geometric_sample` lines 57–64 samples a uniform unit direction and radius `U**(1/d)` in the unit ball, unlike SPSDO's uniform radius `U*r`.
- Lines 66–79 split parallel/perpendicular components along a center-to-surface-point axis, reflect a point when it lies beyond a truncation condition, and multiply the perpendicular component by `1 - deformation_factor`. The author example `docs/examples/plot_gsmote_data_generation.py` lines 98–116 explicitly describes a hypersphere that can be truncated; lines 130–133 describe deformation to an ellipse and finally a line segment.
- `_make_geometric_samples` lines 357–418 offers minority, majority and combined selection strategies. The combined strategy uses the nearer of a selected minority neighbour and nearest majority neighbour for its surface point/radius.

Thus deformable, non-spherical, truncated and majority-informed generation is established prior art. The candidate is distinct at the implementation level in using a per-prototype multiaxis covariance transform and a single analytic scale determined by **every original majority observation**, shared by both antithetic proposals, with fixed anchor quotas. G-SMOTE's verified source does not implement that exact combination. Absence from this one implementation is not proof of novelty.

### OHIT

Author repository: <https://github.com/zhutuanfei/OHIT>, commit `4db0e9b6cd79d4847c4532d96a5e5ee9700346c7`.

- Author README identifies the paper as **Minority oversampling for imbalanced time series classification**. Its DOI was not verified and must remain unspecified.
- Direct source: <https://github.com/zhutuanfei/OHIT/blob/4db0e9b6cd79d4847c4532d96a5e5ee9700346c7/OHIT.m#L46>.
- `OHIT.m` lines 38–53 performs DRSNN minority clustering then explicitly computes a shrinkage covariance separately for each cluster. Lines 100–104 compute the cluster mean and covariance eigenstructure.
- Lines 78–96 generates a batch of standard Gaussian vectors, scales/rotates them using the cluster covariance eigenstructure, translates to the **cluster mean**, and keeps high Gaussian-density proposals after overgeneration.
- `shrinkageCov` at lines 226–271 is a single-index shrinkage estimator; its source copyright is Olivier Ledoit and Michael Wolf (2014). It is not the same estimator/target as sklearn's identity-target `LedoitWolf`, so the candidate must declare the estimator rather than imply implementation equivalence.
- Lines 3–15 select minority data; the verified generation routine does not constrain proposals using original majority points.

Per-mode covariance shrinkage and covariance-shaped generation therefore cannot be the novelty claim. Differences remaining in the candidate are individual fixed anchors, trace normalization to isolate shape from average scale, empirical majority halfspace constraints, and paired radial clipping with exact finite-sample anchor means. This is a mechanism comparison to author code, not a verified complete paper comparison.

### SWIM (relevant majority-aware precedent)

Author repository: <https://github.com/cbellinger27/SWIM>, commit `2994daf93974c95b4aba55ce2f88de243e233e9a`.

- Author `readme.md` identifies **Synthetic oversampling with the majority class: A new perspective on handling extreme imbalance**, S. Sharma, C. Bellinger, B. Krawczyk, O. Zaiane, N. Japkowicz, IEEE ICDM (2018), 447–456; exact DOI not verified.
- `SWIM_RBF/README.md` lines 1–3 identifies **Framework for extreme imbalance classification: SWIM—sampling with the majority class**, Bellinger et al., *Knowledge and Information Systems* (2019), DOI **10.1007/s10115-019-01380-z**. Title/DOI association is verified from author README; publisher page not accessible.
- `SWIM_Maha/Swim_Maha.py` centres observations on the majority mean, whitens using the **majority covariance**, perturbs selected minority anchors in whitened space, and rescales a proposal to its anchor's whitened norm before reversing the transform. This is majority-distribution-aware generation, not a per-majority-site halfspace restriction.

Majority-aware geometry is established; a new contribution cannot be claimed merely because majority observations influence generation.

## Verified secondary-source Voronoi precedent

**Using Voronoi Diagrams to Improve Classification Performances when Modeling Imbalanced Datasets**, William A. Young II, Scott L. Nykl, Gary R. Weckman and David M. Chelberg, *Neural Computing and Applications* 26(5) (2015), 1041–1054; DOI **10.1007/s00521-014-1780-0**.

This bibliographic association was retrieved from the `smote_variants` implementation citation, not independently checked against the publisher/full paper: <https://github.com/analyticalmindsltd/smote_variants/blob/c26cdd9d4a2a61886924fc9b081f0480f24b4108/smote_variants/oversampling/_v_synth.py#L20>.

At this commit, `_v_synth.py` lines 150–196 selects a mixed-label Voronoi face, samples a weighted point on the face, finds the minority site, then interpolates between the face point and minority site; lines 259 onwards construct the tessellation in PCA coordinates. The module explicitly warns that some details are its implementation interpretations. Accordingly, cite this as a Voronoi-region oversampling **lead / secondary implementation evidence**, not a verified exact statement of every paper rule. It is sufficient to reject a broad claim that using Voronoi regions for imbalanced generation is new.

The candidate's analytic inequalities avoid explicitly constructing the entire tessellation and impose an anchor-centred symmetric region; they do not invent majority-informed Voronoi safe-region generation.

## EMDO: unresolved primary-source gap

The requested comparison is to **EMDO's evolutionary ellipsoidal oversampling**. Treat this requested description as a lead, not as independently verified bibliographic evidence. Primary paper access, exact title, DOI, page references, and the precise optimized ellipsoid/radius/majority conditions remain unresolved under the network restrictions above. No affirmative statement that EMDO lacks the candidate's geometry is justified. A full proposal would first require this primary-source check. An ellipsoid itself and evolutionary fitting of an ellipsoid cannot be claimed as new even provisionally.

## Precise candidate construction and closest mathematical identity

For an original minority anchor `x`, let `a_y = y - x` for each original majority site `y`. For `z = x + v`,

`||z-y||² - ||z-x||² = ||a_y||² - 2 vᵀ a_y`.

Hence requiring both `x+v` and `x-v` to be closer to `x` than to every distinct majority site with the declared margin is exactly

`2 |vᵀ a_y| <= eta ||a_y||²` for all `y`.

This is the centrally symmetric intersection of the anchor-versus-majority Voronoi halfspaces with their reflections and the margin factor. It ignores other minority sites; it is not the full Voronoi cell with all sites. Given a proposal ray `v0`, the requested `c(v0)` is the largest multiplier in `[0,1]` satisfying these constraints. Thus the precise remaining design difference is:

> A fixed-anchor, quota-preserving replacement for small-prototype SPSDO: trace-normalized shrinkage covariance proposals, analytically retracted by one shared radial multiplier into an original-majority-defined symmetric margin region, followed by paired emission and odd-quota duplication.

This combination is a reasonable **candidate mechanism**; the verified evidence does not establish a new family or a patent-like novelty conclusion. Covariance, ellipsoids, shrinkage, symmetry, majority-awareness and Voronoi regions are not separate novelty claims. Exact finite-sample centring can also be obtained by paired rejection, so that control is essential.

Clipping and rejection differ: along a fixed proposal direction, clipping replaces radius `R` by `min(R, R_limit)` and therefore places probability mass at the directional constraint limit when the raw radius exceeds it. Rejection samples proposals conditionally inside the constraint region (up to a declared attempt limit and fallbacks). These may have different intrusions, dispersion and runtimes even though both preserve finite-sample paired means.

Conflicting labels at identical coordinates require separate handling. If `a_y=0`, the inequality is `0<=0` for every `v`, yet the anchor and majority site are equidistant from every point. Ignoring this zero denominator does **not** establish strict minority proximity; duplicate and flag the anchor as label-conflicted (or declare another policy), and exclude it from any claimed strict positive-margin geometric guarantee. Duplication still does not certify its label.

The guarantee is only relative to observed sites. It cannot exclude the unknown true majority region, preserve the true minority distribution, recover missing rare modes, or imply better predictive recall/AP. A minority anchor plus its covariance could already lie in true overlap; an empirical Voronoi-compatible point can still belong to the majority distribution.

## Decision relevance

Before recommending development, require the proposed clipping mechanism to add benefit beyond tuned sphere, ellipsoid, original-only screening and paired rejection under matched quotas/anchors and validation-only threshold selection. If these simpler controls remove the apparent benefit, retain this as a negative/neutral mechanistic diagnostic and do not expand into a full proposal. The unresolved EMDO source audit prevents a positive novelty claim regardless of diagnostic outcome.
