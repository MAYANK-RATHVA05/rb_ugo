# Reference access and verification status: Stage 4A-R

Updated 2026-10-11. Six supplied core PDFs were opened, parsed and read in the
sections listed below. Earlier network-access failures remain historical and
are **superseded by the supplied full texts**. DOI strings are not publication
dates: SLEO's PDF prints KBS 333 (2026), although its DOI contains 2025.
The [matrix](PRIOR_ART_MATRIX.md) supplies verified bibliography, equations,
page citations, critical interpretations and guarantee limits. No paper's
reported experiment was reproduced in this audit.

## Core full texts actually inspected

| ID / title | Supplied filename | Pages | Inspected content |
| --- | --- | --- | --- |
| U: Uncertainty guided oversampling: A classifier-specific approach for effective handling of class imbalance in machine learning | 1-s2.0-S095070512600612X-main.pdf | 20 | pp5–8 methods/Tables 2–4; pp8–10 setup; exact batch and cap comparison |
| L: Semi-supervised local entropy-decayed oversampling for imbalanced data | 1-s2.0-S0950705125020477-main.pdf | 18 | pp4–7 Eqs.1–8/Algorithm 1; entropy equation visually checked |
| H: A semi-supervised resampling method for class-imbalanced learning | 1-s2.0-S0957417423002348-main.pdf | 13 | pp4–6 clustering/resampling; p7 setup; pp11–12 no-U ablation |
| G: GDEO: Gini-guided discriminative equilibrium oversampling for imbalanced classification | 1-s2.0-S0925231226015195-main.pdf | 12 | pp4–6 objectives/update, Algorithm 1, complexity/evaluation protocol |
| B: Don't oversample the boundary: A remedy decomposition of the minority deficit in imbalanced learning | 1-s2.0-S0925231226022290-main.pdf | 17 | pp4–6 decomposition/Proposition 1; pp8–9 category interventions/threshold parity and OOF qualification |
| T: Synthetic Augmentation in Imbalanced Learning: When It Helps, When It Hurts, and How Much to Add | 2601.16120v3.pdf | 90 | Main pp9–12 risk/theorems, pp15–19 mismatch/rates, pp21–22 VTSS; v3 dated 7 March 2026 |

U is under ignored `papers/base/`; the other five under ignored
`papers/additional/`. Exact local/access paths, bytes, SHA-256 and parse status
are in [resource_inventory.csv](../../data/manifests/resource_inventory.csv).
Metadata were checked on the supplied PDFs; remote publisher authentication was
not independently performed. Full-text availability means access plus inspection
of the stated sections, not line-by-line verification of every supplemental proof.
Author code and cap/pool clarification are still absent.

## Additional previously verified full-text publications

R: Ren, Zeng, Yang & Urtasun (2018), *Learning to Reweight Examples for Robust
Deep Learning*, ICML/PMLR 80:4334–4343. §§3.1–3.4 objectives, meta-gradient,
stated optimization assumptions were verified in the earlier audit.

D: Ghorbani & Zou (2019), *Data Shapley: Equitable Valuation of Data for Machine
Learning*, ICML/PMLR 97:2242–2251. §2/§3 group marginal valuation and §4 acquisition /
evaluation separation were verified previously.

I: Koh & Liang (2017), *Understanding Black-box Predictions via Influence
Functions*, ICML/PMLR 70:1885–1894. §2.1 influence equations and local smoothness /
Hessian limits were verified previously.

Primary PDF links, hashes and exact limits remain in the matrix. These three
were not newly supplied in the six attachments and their supplemental proofs
were not comprehensively re-audited. **Abstract-only verified papers: none.**

## Supplied collection that cannot be transferred

`drive-download-20261009T160315Z-1-001.zip` was offered as the 13-paper collection.
The attachment download tool rejected it as **exceeding its 32 MiB transfer limit**.
There is no accessible local archive, member list, size or SHA-256 to invent.
Whether any member duplicates a core paper is also unknown.

| User-named related group | Current evidence |
| --- | --- |
| CFS-SMOTE / BFCH-SMOTE / EC-EBO | FULL TEXT UNAVAILABLE; exact bibliography/member names unverified |
| Mahalanobis-based oversampling | FULL TEXT UNAVAILABLE; no specific publication identified from archive |
| Instance-hardness-guided oversampling | FULL TEXT UNAVAILABLE for requested collection members |
| Overlap-minimization / hyper-cylinder oversampling / membership-based resampling | FULL TEXT UNAVAILABLE; exact source mapping unverified |
| Other members of the reported 13-paper collection | FULL TEXT UNAVAILABLE; cannot enumerate an inaccessible archive |

Provide individual PDFs or split the archive into parts **each under 32 MiB**,
with original filenames retained. No unofficial paper copy/download was substituted.
Original robust graph/SSL, local acquisition and synthetic bilevel selection papers
still need a focused direct literature search; this audit is not exhaustive.

## Documentation and dataset evidence

S1–S7 in the matrix are pinned scikit-learn 1.5.2 / imbalanced-learn 0.12.4
source/documentation, not original-paper reviews. Their implementation evidence
does not verify every paper cited in a library file. KEEL's general project README
previously accessible is not a dataset-specific license or fold manifest.

Eight actual user-supplied ZIPs now verify file-level schemas/counts/checksums
[DATASET_VERIFICATION.md](DATASET_VERIFICATION.md). They contain no folds;
provider authentication, domain class grouping, license and official fold protocol
remain unverified. File parsing is not authoritative provider authentication.
