# Reference status and source requirements

Access checked 2026-10-11; initial audit was 2026-10-10. Status describes inspected
evidence, not whether a publication exists. **Full text not verified** for all
six core sources, and their abstracts remain unverified too. The six exact titles
below are supplied by the user; no title-to-identifier match was independently
established. DOI-year strings are not verified publication dates.

## Core bibliography and access

| ID | Exact user-supplied title | DOI/source | Independently verified authors/year | Access level |
| --- | --- | --- | --- | --- |
| UGO | Uncertainty guided oversampling: A classifier-specific approach for effective handling of class imbalance in machine learning | [10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886) | Not verified; earlier user attribution is preserved in the Stage 3 record | Unverified; full text and abstract not verified |
| SLEO | Semi-supervised local entropy-decayed oversampling for imbalanced data | [10.1016/j.knosys.2025.115009](https://doi.org/10.1016/j.knosys.2025.115009) | Not verified | Unverified; full text and abstract not verified |
| SSHR | A semi-supervised resampling method for class-imbalanced learning | [10.1016/j.eswa.2023.119733](https://doi.org/10.1016/j.eswa.2023.119733) | Not verified | Unverified; full text and abstract not verified |
| GDEO | GDEO: Gini-guided discriminative equilibrium oversampling for imbalanced classification | [10.1016/j.neucom.2026.134121](https://doi.org/10.1016/j.neucom.2026.134121) | Not verified | Unverified; full text and abstract not verified |
| Boundary | Don't Oversample the Boundary: A Remedy Decomposition of the Minority Deficit in Imbalanced Learning | [10.1016/j.neucom.2026.134831](https://doi.org/10.1016/j.neucom.2026.134831) | Not verified | Unverified; full text and abstract not verified |
| Augmentation | Synthetic Augmentation in Imbalanced Learning: When It Helps, When It Hurts, and How Much to Add | [arXiv:2601.16120](https://arxiv.org/abs/2601.16120) | Not verified | Unverified; full text and abstract not verified |

Each DOI/arXiv landing-page request returned `Tunnel connection failed: 403
Forbidden`; so did the augmentation PDF and Crossref UGO lookup. No core paper
section/page was inspected. This is a network-policy failure, not evidence of a
paywall, incorrect identifier, absent method, or deficient paper. The earlier
audit also recorded blocked export.arxiv/OpenAlex routes. No network configuration
was changed; no paywall was bypassed. Access logs/PDFs stay under `/tmp`.

## Verified claims and unavailable evidence per core source

| Source | Claims verified from publication | Needed source material and specific unresolved questions |
| --- | --- | --- |
| UGO | None | Legitimate PDF, metadata and author code if available; exact Sections 3.1–3.4, Tables 2–4 and experimental protocol. These locations were supplied earlier, not inspected. Resolve cap/pools/rounding/noise and class choice before fidelity claims |
| SLEO | None | Metadata and full algorithm/entropy/disjunct sections; determine whether semi-supervision actually uses a distinct U pool rather than internal labels or clustering |
| SSHR | None | Metadata, hierarchy/resampling/pseudo-label sections and protocol; identify U provenance, anchors, generated labels, classifier influence and fair label budget |
| GDEO | None | Metadata, exact Gini/discriminative constraints, point-generation and overlap treatment, assumptions and complexity |
| Boundary | None | Metadata and complete remedy-decomposition definitions/proofs/experiments; verify threshold-recoverable, data-reducible and irreducible criteria and whether local intervention benefit is already estimated |
| Augmentation | None | Metadata and full population-risk/mismatch/augmentation-size theory, assumptions and VTSS procedure. VTSS is a user-reported topic, not a verified equation or acronym definition |

These questions come from the user brief and are not findings about the papers.
Abstracts could verify broad objectives; exact equations, absent-comparator claims,
guarantees and complexity generally need full method/experiment sections. The
[matrix](PRIOR_ART_MATRIX.md) keeps these cells UNVERIFIED, never “No”.

## Additional papers verified from full text

The official PMLR repository PDFs were read during the initial audit and
redownloaded successfully on 2026-10-11 with identical hashes. Publisher metadata
was also verified in the initial audit. Re-fetching is an access/integrity check,
not a claim that all supplemental proofs were reviewed again.

| ID | Verified bibliography and primary full text | Sections/equations inspected | Verified claims and limits |
| --- | --- | --- | --- |
| R | Ren, Zeng, Yang & Urtasun (2018). Learning to Reweight Examples for Robust Deep Learning. ICML, PMLR 80:4334–4343. [PDF](https://raw.githubusercontent.com/mlresearch/v80/gh-pages/ren18a/ren18a.pdf) | §§3.1–3.4; Eqs. 1–2, 5, 7–9, 12, 15; Algorithm 1 and reported experiments | Validation-guided example weighting/gradient alignment; authors' stated smoothness/gradient/step-size optimization conditions. Not synthetic-label safety or outer-risk guarantee; supplemental proof audit outstanding |
| D | Ghorbani & Zou (2019). Data Shapley: Equitable Valuation of Data for Machine Learning. ICML, PMLR 97:2242–2251. [PDF](https://raw.githubusercontent.com/mlresearch/v97/gh-pages/ghorbani19c/ghorbani19c.pdf) | §2 Proposition 2.1, §3 Eq. 2/group extension, §§4.1/4.5 and evaluation separation | Marginal/group data value and acquisition guidance. Our counterexample challenges sufficiency of the three displayed uniqueness axioms; does not invalidate the defined valuation. Supplemental approximation details not reviewed |
| I | Koh & Liang (2017). Understanding Black-box Predictions via Influence Functions. ICML, PMLR 70:1885–1894. [PDF](https://raw.githubusercontent.com/mlresearch/v70/gh-pages/koh17a/koh17a.pdf) | §2.1 Eqs. 1–2, §§4.2–4.3, §7 | Infinitesimal training influence under smooth optimum/positive-definite Hessian conditions; finite/nonconvex effects are approximate. Supplemental proofs not reviewed |

Exact publisher-metadata links, author equations and PDF hashes are in
[PRIOR_ART_MATRIX.md](PRIOR_ART_MATRIX.md). **Abstract-only verified publications:
none.** Abstracts of R/D/I were also checked, but their status is full text.

## Documentation/source evidence, not original-paper review

The matrix's S1–S7 are pinned scikit-learn 1.5.2 and imbalanced-learn 0.12.4
documentation/source. They support threshold tuning, metrics, separate-U APIs,
self-training/graphs, cross-validation, ordinary oversampling and KMeansSMOTE
implementation claims. The papers cited by those source files were not thereby
read. In particular Last/Douzas/Bacao and Zhou et al. are comparator attributions,
not verified original equations or theorems.

KEEL's [official project README](https://raw.githubusercontent.com/SCI2SUGR/KEEL/master/README.md)
was accessible (HTTP 200); it describes the software/data-repository ecosystem
and general dataset import. It does not verify the shortlist's counts, targets,
feature schemas or folds. The [imbalanced index](https://sci2s.ugr.es/keel/imbalanced.php)
was blocked. The software's stated GPLv3 license is not proof that every dataset
has that license. Dataset-specific evidence remains required.

## Related-paper collection and expanded search gaps

No identifiable collection containing the following papers was found in the
repository or earlier accessible Stage 1–3 pasted instructions. These are search
labels supplied by the user, not independently verified bibliographic records:

- CFS-SMOTE, BFCH-SMOTE and EC-EBO: need exact titles, authors, DOI/URL and papers.
- Mahalanobis oversampling: need specific publications rather than a metric family.
- Instance-hardness methods: need the particular selection/resampling publications.
- Overlap-minimization methods: need the actual algorithms, objectives and sources.

Also still needed: direct full-text checks of synthetic-data bilevel selection,
local/group acquisition, remedy decomposition, robust graph SSL, harmonic-extension
uncertainty/effective resistance and augmentation mismatch theory. These are
literature gaps in **this audit**, not established gaps in the field. Suggested
searches are a future plan, not claimed completed searches.

## What to supply or unblock

Supply legitimate PDFs or source excerpts/abstracts with provenance, particularly
the boundary-remedy and augmentation-theory papers, plus the related-paper
collection's bibliographic list and authoritative KEEL metadata/fold manifests.
Public destinations needed for a renewed check include `doi.org`, `arxiv.org`,
`export.arxiv.org`, `api.crossref.org`, `api.openalex.org`, publisher redirect hosts
and `sci2s.ugr.es`; PR creation separately needs authorized GitHub API access.
Reachability is not entitlement to a publisher PDF. No token values are requested
in chat. Until evidence arrives, method selection and implementation remain gated.
