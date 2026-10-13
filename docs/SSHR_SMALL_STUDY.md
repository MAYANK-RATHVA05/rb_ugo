# SSHR small real-data study — Stage 4

Stage 4 is complete: **800/800 fixed paired jobs succeeded** on Ecoli4 and
Yeast4, including the empty-U ablation. The protocol, sources, preparation and
reporting code were committed before training at
`6e0f1c31b5c54c694b8107e846f5111719eae85d`. No policies were tuned from outcomes.
The measured results below show mixed metric tradeoffs, with lower mean F1
when U was supplied in all four dataset/classifier conditions.
This is a descriptive pilot on **two local datasets**, not a reproduction of
the SSHR paper's 44-dataset benchmark, UGO's 74-dataset benchmark, a novelty
claim or a claim of general superiority.

## Inputs verified before training

Both supplied ZIP archives and their extracted `.dat` files were hash-checked.
The local inputs match registered hashes and the names, dimensions and class
counts in Jiang et al. (2023), *A semi-supervised resampling method for
class-imbalanced learning*, [DOI:10.1016/j.eswa.2023.119733](https://doi.org/10.1016/j.eswa.2023.119733),
Table 2, printed p.7. This verifies source identity for this run and paper
metadata compatibility; exact original-author files and preprocessing remain
unverified. The archives contain whole datasets and no official CV folds.

| Input | Rows | Numeric input features | Majority / minority | Exact feature-coordinate groups |
| --- | ---: | ---: | ---: | --- |
| Ecoli4 | 336 | 7 | 316 / 20 | 336 unique groups; no duplicates |
| Yeast4 | 1,484 | 8 | 1,433 / 51 | 1,453 groups; 31 pairs, all majority with matching labels |

- `data/raw/ecoli4.dat`, SHA256 `1890c32ba2550cfe106d0524ebf6a42c2b50730f6521bd4f22eb58e7bda86bcb`.
- `data/raw/yeast4.dat`, SHA256 `2e50d1b43c97875d2071859428d718487086e695a1a6ec559336ccadb40c1ff0`.
- Supplied Ecoli4 ZIP, SHA256 `75f3a62a1d23b0081976131f6c18331219e29b07ea616620a650b68a27c831d7`.
- Supplied Yeast4 ZIP, SHA256 `dda966d6e09c665bab0325e3481dc0045e7f2fb9a4ddabdf47607ef4031b88b5`.

All source input attributes are declared `real` and load as numeric columns;
there are no missing or nonfinite values. Labels `negative`/`positive` map to
0/1, with positive fixed as minority. Ecoli4 uses `Mcg,Gvh,Lip,Chg,Aac,Alm1,Alm2`;
Yeast4 uses `Mcg,Gvh,Alm,Mit,Erl,Pox,Vac,Nuc`. Numerical attribute declarations
do not prove physical plausibility of interpolated biological measurements.
No entity IDs or timestamps were supplied; dependence beyond exact coordinate
duplicates cannot be assessed from these files.

Raw rows are **not deleted, modified or committed to Git**. Identical feature
vectors define coordinate groups without using labels. The study rejects
conflicting-label groups instead of inventing a consensus label; none occur
in these two sources. Each group remains together across outer training/test
and also across observed/unlabelled training pools. This prevents an identical
copy from appearing across either boundary, while preserving original row
frequencies. A declared group cannot split an identical-coordinate pair.

## Fixed evaluation and label budget

Use repeated five-fold `StratifiedGroupKFold`, five repeats, root seed 42.
Group splitting matches neither recovered author folds nor necessarily the
authors' dependence assumptions. Fold and mask seeds use the recorded
`SeedSequence` derivation. All folds and masks are saved before learned
operations. Within each outer training fold, known labels are used only for
the stratified label-hiding experiment design. Shuffle same-class coordinate
groups deterministically and retain complete groups until at least
`max(2,floor(0.5*n_training_class))` labelled rows are available. Record any
group overshoot/minimum clamp and actual label fraction. No mask is redrawn
because a method fails.

NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE, EC-EBO and SSHR receive the same reduced
observed labelled set `L`. SSHR additionally receives the disjoint training-only
unlabelled feature pool `U`. Fit the existing imputer, within-class variance
filter and scaler on `L` only. Transform `U` using that fit. Test rows enter
only final transformation/prediction, after resampling and fresh classifier
fitting. Hidden training labels are evaluator-only after resampling and never
enter the SSHC objective, confidence, quotas, synthetic generation or final fit.

Classifiers are the existing `LogReg` and `GNB` implementations with their
recorded defaults and fold seeds. Samplers retain their audited defaults;
no settings are chosen from test scores or retrospective pseudo-label accuracy.
This is not the SSHR paper's reported RBF-SVM setup. Original UGO, BFCH-SMOTE
and EC-EBO source algorithms and existing evaluation defaults are preserved.

The main comparison contains `2 datasets × 2 classifiers × 25 folds × 6 methods
= 600 jobs`. The ablation contains the same folds, masks, classifiers and `L`,
with NORES and SSHR but `use_unlabeled=False`, yielding 200 more jobs. NORES is
intentionally executed in both run identities and must produce identical scores,
predictions and probabilities. The 100 empty-U SSHR jobs estimate the effect
of overall access to `U`, including changes to clustering geometry and selection;
they do not isolate those mechanisms individually.

Actual observed minority counts were 7–8 per Ecoli4 training L and 17–23 per
Yeast4 L. Held-out minority counts ranged from 3–5 and 4–16, respectively.
Stratified group splitting is approximate; Yeast4's uneven minority test counts
are a material source of score variability. Actual labelled fractions ranged
from 0.49814–0.50000 (Ecoli4) and 0.49916–0.50042 (Yeast4); no class minimum
clamp was needed.

The training-only within-class variance filter retained 5 features in 18 Ecoli4
folds and 6 in 7 folds (Chg always removed; Lip removed in 18). It retained 6
features in 20 Yeast4 folds and 7 in 5 folds (Pox always removed; Erl removed
in 20). The same fitted feature selection was used by every method and both
U conditions within a fold; no test features determined these decisions.

## Reporting rules

Report held-out standard minority F1, G-mean, MCC, probability ROC-AUC and
`FPR=FP/(TN+FP)`. The printed SSHR AUC definitions are not assumed equivalent;
see [the source audit](SSHR_METHOD_DECISIONS.md). Preserve each job's class
orientation, source hashes, folds, masks, seeds and classifier parameters.

Canonical metric comparisons use the same successful repeat/fold pairs across
all selected methods, with failures, unavailable and missing jobs retained in
coverage. Also export clearly labelled individual-success summaries. Repeated
fold scores are correlated; sample standard deviations are descriptive, not
confidence intervals, independent replications or significance tests.

SSHR primarily adds **existing pseudo-labelled observations**, not synthetic
feature vectors. Only the SMOTE fallback generates vectors. Report selected
counts, SMOTE counts, majority/minority removals, cluster sizes/anchors,
requested and unmet quotas, sampler stop reasons and coordinate duplicates.
Pseudo-label accuracy is `sum(correct)/sum(selected)` over selection events
within a dataset/classifier condition; zero selection is undefined. A row may
be selected across different repeats, and SSHR generation is classifier-independent,
so aggregate counts are not unique new observations and should not be pooled
across classifier copies.

The training-only geometric proxy counts additions closer to an original
observed majority row than to an original observed minority row, with exact
distance ties separate. Every method uses the original reduced `L` reference
before method-specific deletion. Report pseudo-labelled and generated minority
rows separately. Original UGO can generate either assigned class: generated
majority rows are counted separately and must not be described as synthetic
minority intrusions. This distance proxy does not establish true class overlap
or a generated row's class membership.

The empty-U comparison uses matching successful SSHR fold pairs and checks
that the duplicate NORES controls agree. A valid classifier result with a
resampler quota shortfall remains a successful job with a shortfall, not a
fabricated balanced result. No masks, thresholds or policies are changed after
reviewing outcomes.

## Measured held-out results

All **800/800 jobs succeeded**, with no failed, unavailable or missing jobs:
600 six-method comparisons and 200 NORES/SSHR empty-U controls. Each condition
has all 25 shared repeat/fold events. There was no parameter or mask redraw
after scores were inspected. These are means ± sample SD of held-out fold
scores; FPR is a proportion and lower is preferable. Repeated-fold dependence
precludes interpreting these SDs as confidence intervals.


### Ecoli4

| Classifier | Method | F1 | G-mean | MCC | Probability ROC-AUC | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LogReg | NORES | 0.8720 ± 0.1085 | 0.9009 ± 0.0929 | 0.8738 ± 0.1056 | 0.9955 ± 0.0071 | 0.0032 ± 0.0065 |
| LogReg | SMOTE | 0.8194 ± 0.1350 | 0.9381 ± 0.0732 | 0.8158 ± 0.1397 | 0.9951 ± 0.0070 | 0.0210 ± 0.0199 |
| LogReg | UGO-SMOTE | 0.8890 ± 0.1071 | 0.9287 ± 0.0863 | 0.8885 ± 0.1076 | 0.9951 ± 0.0072 | 0.0051 ± 0.0075 |
| LogReg | BFCH-SMOTE | 0.7949 ± 0.1398 | 0.9492 ± 0.0579 | 0.7971 ± 0.1369 | 0.9960 ± 0.0058 | 0.0292 ± 0.0270 |
| LogReg | EC-EBO | 0.7247 ± 0.1218 | 0.9373 ± 0.0586 | 0.7279 ± 0.1182 | 0.9940 ± 0.0100 | 0.0424 ± 0.0282 |
| LogReg | SSHR | 0.5838 ± 0.1640 | 0.9331 ± 0.0342 | 0.6130 ± 0.1464 | 0.9916 ± 0.0130 | 0.1006 ± 0.0560 |
| GNB | NORES | 0.7694 ± 0.1663 | 0.8661 ± 0.1214 | 0.7670 ± 0.1713 | 0.9871 ± 0.0160 | 0.0152 ± 0.0168 |
| GNB | SMOTE | 0.7363 ± 0.1511 | 0.8788 ± 0.1104 | 0.7348 ± 0.1543 | 0.9888 ± 0.0134 | 0.0260 ± 0.0251 |
| GNB | UGO-SMOTE | 0.7729 ± 0.1679 | 0.8664 ± 0.1218 | 0.7704 ± 0.1728 | 0.9871 ± 0.0160 | 0.0146 ± 0.0165 |
| GNB | BFCH-SMOTE | 0.7201 ± 0.1471 | 0.8521 ± 0.1165 | 0.7141 ± 0.1517 | 0.9850 ± 0.0145 | 0.0215 ± 0.0188 |
| GNB | EC-EBO | 0.5871 ± 0.1392 | 0.9093 ± 0.0748 | 0.6016 ± 0.1355 | 0.9838 ± 0.0254 | 0.0822 ± 0.0446 |
| GNB | SSHR | 0.5714 ± 0.2103 | 0.8580 ± 0.1589 | 0.5824 ± 0.2103 | 0.9757 ± 0.0264 | 0.0779 ± 0.0612 |

### Yeast4

| Classifier | Method | F1 | G-mean | MCC | Probability ROC-AUC | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LogReg | NORES | 0.2018 ± 0.1759 | 0.3050 ± 0.2242 | 0.2331 ± 0.1889 | 0.8718 ± 0.0652 | 0.0046 ± 0.0047 |
| LogReg | SMOTE | 0.2757 ± 0.0811 | 0.7898 ± 0.0757 | 0.3083 ± 0.0798 | 0.8698 ± 0.0574 | 0.1258 ± 0.0290 |
| LogReg | UGO-SMOTE | 0.2519 ± 0.1808 | 0.3961 ± 0.2717 | 0.2717 ± 0.1796 | 0.8737 ± 0.0615 | 0.0106 ± 0.0104 |
| LogReg | BFCH-SMOTE | 0.3025 ± 0.0875 | 0.7739 ± 0.0807 | 0.3234 ± 0.0892 | 0.8545 ± 0.0682 | 0.0995 ± 0.0253 |
| LogReg | EC-EBO | 0.3189 ± 0.0852 | 0.7854 ± 0.0746 | 0.3414 ± 0.0825 | 0.8705 ± 0.0667 | 0.0923 ± 0.0186 |
| LogReg | SSHR | 0.2962 ± 0.0908 | 0.7935 ± 0.1284 | 0.3363 ± 0.0881 | 0.8781 ± 0.0729 | 0.1107 ± 0.0351 |
| GNB | NORES | 0.2768 ± 0.1243 | 0.5895 ± 0.1255 | 0.2580 ± 0.1268 | 0.8357 ± 0.0854 | 0.0488 ± 0.0171 |
| GNB | SMOTE | 0.2301 ± 0.0642 | 0.7598 ± 0.0857 | 0.2599 ± 0.0711 | 0.8187 ± 0.0847 | 0.1535 ± 0.0382 |
| GNB | UGO-SMOTE | 0.2639 ± 0.1220 | 0.5680 ± 0.1313 | 0.2438 ± 0.1254 | 0.8326 ± 0.0836 | 0.0470 ± 0.0177 |
| GNB | BFCH-SMOTE | 0.2239 ± 0.0779 | 0.6872 ± 0.1179 | 0.2312 ± 0.0917 | 0.7782 ± 0.0847 | 0.1246 ± 0.0469 |
| GNB | EC-EBO | 0.2789 ± 0.0807 | 0.7782 ± 0.0918 | 0.3065 ± 0.0881 | 0.8360 ± 0.0969 | 0.1150 ± 0.0246 |
| GNB | SSHR | 0.2722 ± 0.0810 | 0.7032 ± 0.1294 | 0.2825 ± 0.0821 | 0.8356 ± 0.0872 | 0.0933 ± 0.0477 |

On Ecoli4, SSHR had lower mean F1, MCC and probability ROC-AUC than NORES
with both classifiers, with higher FPR. For LogReg, F1 changed from 0.8720
(NORES) to 0.5838 (SSHR), while FPR rose from 0.0032 to 0.1006.
On Yeast4, SSHR improved mean LogReg F1 from 0.2018 to 0.2962 and G-mean
from 0.3050 to 0.7935, with FPR rising from 0.0046 to 0.1107. Yeast4 GNB
F1 was 0.2768 for NORES and 0.2722 for SSHR. These differing results do not
establish SSHR superiority or successful reproduction of the authors' scores.
The complete shared and supplementary summaries are in
[the artifact index](sshr_pilot/README.md).

## Does access to unlabelled training features help here?

Every difference below is **SSHR with U minus SSHR with empty U**, using
the same 25 successful fold pairs, reduced L, preprocessing and final classifier.
Full with-U/empty-U means and descriptive paired-difference SDs are in the
[Ecoli4](sshr_pilot/ecoli4/ablation.csv) and
[Yeast4](sshr_pilot/yeast4/ablation.csv) exports.


| Dataset | Classifier | ΔF1 | ΔG-mean | ΔMCC | ΔProbability ROC-AUC | ΔFPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ecoli4 | LogReg | -0.0777 | -0.0160 | -0.0678 | -0.0028 | +0.0303 |
| Ecoli4 | GNB | -0.0922 | -0.0054 | -0.0810 | -0.0078 | +0.0361 |
| Yeast4 | LogReg | -0.0096 | +0.0172 | -0.0018 | -0.0071 | +0.0145 |
| Yeast4 | GNB | -0.0054 | +0.0810 | +0.0017 | +0.0296 | +0.0333 |

Mean F1 decreased in all four conditions when U was supplied. FPR increased
in all four. Yeast4 GNB nevertheless gained probability ROC-AUC (+0.0296)
and G-mean (+0.0810), so the result is a metric tradeoff rather than a universal
absence of useful information in U. This pilot supplies no consistent F1
benefit from U under the fixed implementation and settings. It cannot show
that semi-supervised resampling never helps: two datasets, these label budgets,
these classifiers, the literal objective and unresolved paper interpretations
limit that conclusion. The 100 duplicated NORES controls had exactly identical
predictions, probabilities, metrics, preprocessing and seeds across the two
run identities.

## SSHR selection, synthesis and quotas

Counts below sum the **25 training-fold events for one classifier**, not unique
individuals. Both classifiers have identical SSHR selection, lineage and
resampling counts. Do not add the two classifier copies together. Correct
pseudo-label counts were independently checked against the original hidden
training gold labels only after generation.

| Dataset | Removed majority | Removed minority | Existing U rows selected | Correct / selected | SMOTE-generated | Quota-shortfall folds | Unmet weighted removals | Unmet additions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ecoli4 | 853 | 0 | 313 | 170 / 313 = 54.31% | 1,436 | 3 / 25 | 105 | 249 |
| Yeast4 | 4,061 | 3 | 273 | 96 / 273 = 35.16% | 8,938 | 1 / 25 | 166 | 388 |

Ecoli4 selected 143 gold-majority rows as minority; Yeast4 selected 177.
Confidence here did not establish minority membership. The default fallback
can use selected pseudo-labelled observations as interpolation anchors; its
synthetic vectors have no available gold labels. Their correctness is not
implied by the pseudo-label audit or by distance to a centroid.

The observed-input/output accounting over the 25 events is:

- Ecoli4: `3,350 - 853 - 0 + 313 + 1,436 = 4,246`.
- Yeast4: `14,829 - 4,061 - 3 + 273 + 8,938 = 19,976`.

All unmet quotas arose when no surviving minority-labelled centroid remained.
Using zero-based `(repeat, fold)`, Ecoli4 shortfalls were `(0,2), (2,0), (4,1)`:
each left 35 weighted removals and 83 additions unmet, with a final class gap
of 118. Yeast4 `(1,1)` left 166 removals and 388 additions unmet, with gap 554.
The same folds also had shortfalls under empty U. Observed labels still
supported both classes and final classifier fitting; these are successful
jobs with incomplete resampling, not balanced outputs.

Without U, Ecoli4 removed 856 majority rows and generated 1,746 SMOTE rows;
Yeast4 removed 4,088 majority and 4 minority rows and generated 9,185 SMOTE rows.
Neither empty-U condition selected pseudo-labelled rows. In the supervised
comparisons, EC-EBO also recorded partial generation in 11 of 25 Yeast4 events
per classifier; all statuses/counts are retained in the diagnostics.

## Clusters, objective and coordinate diagnostics

The following describes final surviving SSHR clusters after cleaning/weighted
removal, before appended rows, in the with-U LogReg copy. Cluster size includes
remaining observed L and assigned U. GNB repeats exactly the same cluster records.

| Dataset | Clusters per fold, min–max | Cluster records over 25 folds | Cluster size, min–max | Unanchored records | Impure records |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ecoli4 | 1–3 | 50 | 15–269 | 0 | 5 |
| Yeast4 | 7–30 | 462 | 1–732 | 68 | 115 |

Detailed observed class counts, U counts and fallback pools are in
[Ecoli4 clusters](sshr_pilot/ecoli4/clusters.csv) and
[Yeast4 clusters](sshr_pilot/yeast4/clusters.csv). An unanchored cluster has no
observed L rows; its birth label is not verified class membership, and it cannot
supply pseudo-labels or fallback seeds without an observed minority anchor.

No split/refinement iteration guard was reached in either U condition.
With U, splitting stopped on objective decline in 19 Ecoli4 and 9 Yeast4
events; the remaining Ecoli4 events stopped on no-progress partitions (3)
or no impure clusters (3), and the other 16 Yeast4 events on no-progress
partitions. Refinement stopped on objective decline in 22 Ecoli4 and 24
Yeast4 events, and on centroid convergence in the other 3 and 1, respectively.

The literal printed Eq.4 quantity called F1 exceeded one at the final SSHC
state in **22/25 Ecoli4 and 6/25 Yeast4 events with U** (22/25 and 5/25 with
empty U). It was recorded with its anomaly flag and was neither clipped nor
replaced after seeing results. This is the audited paper-formula conflict,
not a held-out F1 value. The performance tables use ordinary bounded minority
F1. Author-code confirmation is needed before attributing these pilot outcomes
to the original author implementation; see
[the equation and policy audit](SSHR_METHOD_DECISIONS.md).

Coordinate diagnostics preserve equality counts separately from lineage.
Over 25 events, Yeast4 recorded 304 extra duplicate coordinates in observed
inputs and 316 in U inputs, one selected row duplicating a prior selection,
and seven SMOTE rows duplicating retained, pseudo-labelled or previous generated
coordinates. Ecoli4 recorded zero for these fields. Group boundaries were
defined on original raw coordinates; the training-fitted feature filter may
map distinct original vectors to the same transformed vector. Neither these
coordinate counts nor the raw grouping establish independent biological entities.

## Training-only geometric proxy

The table sums LogReg fold events and uses the same original reduced L
reference for every method, before any sampler deletes rows. Rates are
`nearest original observed majority distance < nearest original observed
minority distance`, with strict ties separately recorded. GNB counts, including
classifier-dependent UGO generation, are in the CSV diagnostics.


| Dataset | Method / added-row type | Rows | Nearest-majority count | Rate |
| --- | --- | ---: | ---: | ---: |
| Ecoli4 | SMOTE / generated minority | 2,956 | 52 | 1.76% |
| Ecoli4 | UGO-SMOTE / generated minority | 210 | 0 | 0.00% |
| Ecoli4 | BFCH-SMOTE / generated minority | 2,956 | 115 | 3.89% |
| Ecoli4 | EC-EBO / generated minority | 2,956 | 59 | 2.00% |
| Ecoli4 | SSHR / existing pseudo-labelled | 313 | 135 | 43.13% |
| Ecoli4 | SSHR / SMOTE-generated | 1,436 | 291 | 20.26% |
| Yeast4 | SMOTE / generated minority | 13,823 | 1,882 | 13.61% |
| Yeast4 | UGO-SMOTE / generated minority | 480 | 0 | 0.00% |
| Yeast4 | BFCH-SMOTE / generated minority | 13,823 | 2,777 | 20.09% |
| Yeast4 | EC-EBO / generated minority | 13,797 | 343 | 2.49% |
| Yeast4 | SSHR / existing pseudo-labelled | 273 | 167 | 61.17% |
| Yeast4 | SSHR / SMOTE-generated | 8,938 | 2,836 | 31.73% |

All distance-tie counts were zero in these listed LogReg groups. UGO-GNB did
generate majority-assigned rows: 42 on Ecoli4 and 1,200 on Yeast4 over the 25
fold events. These rows are kept separate from minority-assigned additions
in the exports and must not be counted as minority intrusions. NORES has no
additions and an undefined proxy rate. A low proxy does not prove boundary
safety; a high proxy
does not prove irreducible overlap. For SSHR pseudo-labelled rows the separate
gold-label audit measures selected-row class correctness; no such truth is
available for newly interpolated vectors.

## Validation and run identity

Software validation before training passed **155 focused tests** in 5.20 s and
**all 738 tests** in 18.76 s, with no failures or skips. Tests cover group/mask
boundaries, source verification, baseline output accounting, metric/report
arithmetic, strict identities, failed/partial coverage and ablation controls.
A separate read-only audit of all 800 saved jobs used original raw gold labels
and independently checked held-out metrics, pseudo-label correctness, source/
manifest/job hashes, coordinate-group boundaries, row lineage, class counts,
quotas, classifier-independent SSHR outputs and NORES equivalence. It performed
no refits. An exact repeat of the study command reused all 800 cached jobs.

The first invocation completed in 60.17 s wall time, with process-reported peak
resident memory 398,720 KiB (389.38 MiB); these are environment-specific resource
observations, not comparative timing claims. Python was 3.11.16, NumPy 1.26.4,
pandas 2.2.3, SciPy 1.14.1, scikit-learn 1.5.2 and imbalanced-learn 0.12.4.
Existing optional backends were available but were not selected classifiers.

| Dataset | Main with-U run ID | Empty-U run ID |
| --- | --- | --- |
| Ecoli4 | `cab13b06468d7cc1d828f74de3bcb56d66a3930a28877380f9fb009c4a34433b` | `fab1b663e4676c6ce97e37d57bdf3ef91d489505b508349898ccdaf10d0e842b` |
| Yeast4 | `f671665b652ece0399ca7e073cd50abfc95a85fe505b00b28caf3214cbe76de7` | `f0056c079a11992b13a43a29af698753c8361f833ea4e7f414dfda83e6de3e18` |

The normalized runner fold/mask hashes are retained in each summary's identity.
The source provenance's `frozen_design` hashes describe the original design
records including actual split/mask seeds and group-target metadata; they are
intentionally different representations. Supplied executed masks have
`mask_seed=None`, while original design seeds remain in provenance. All runs
used the same source fingerprint
`df36f8dbf26fde3956e87c06d4f7856192d87bb7dc68d68566b06111b862a478`.
Source/aggregate checksums and validation counts are preserved in
[validation.json](sshr_pilot/validation.json).

## Reproduction and limits

The [source register](../data/manifests/sshr_pilot_sources.json) contains
portable raw-file paths and expected hashes. Place the unchanged extracted
`ecoli4.dat` and `yeast4.dat` in `data/raw/` in the cloud or your separate Windows
checkout. The ZIP files can optionally remain in `data/incoming/`; if present,
the runner verifies their registered hashes and members. No dataset is downloaded
or uploaded to Git. Source archive presence/verification is part of provenance,
so a different local source state can yield a different run identity.

With the pinned dependencies installed, run from the repository root:

```bash
python -m scripts.run_sshr_small_study --output results/sshr-small-study --report-dir docs/sshr_pilot
```

In this cloud checkout, use `.venv/bin/python` in place of `python`; on Windows,
use the activated project's Python environment. The fixed script exposes output
and dataset selection only, not score-driven parameter/fast overrides. Raw inputs
and detailed job records stay local and ignored by Git. Shared reports retain
descriptive summaries, diagnostics and source/fold/mask identities, without raw
features or individual prediction/gold-label arrays. See
[the artifact index](sshr_pilot/README.md) for schemas and re-exporting saved runs.
Changing code, software, config or provenance requires a fresh result directory;
never silently overwrite a different run's evidence. Generated feature
coordinates are not retained in the saved job records, so the independent
read-only audit checks proxy aggregation and the verified training-only code
scope; it cannot independently recompute each generated row's geometric proxy
from these records. It recomputes held-out metrics and pseudo-label accuracy,
not fitted classifier predictions themselves.

Without tuned-threshold NORES controls, F1 changes alone cannot establish
improved discrimination rather than changed class weighting or decision behavior.
The empty-U control and probability ROC-AUC provide additional descriptive
evidence, not guarantees. The literal printed SSHC objective and declared
pruning, ranking, quota and SMOTE interpretations remain; author source code
and exact unlabelled-data protocol are unverified. Two datasets and two classifiers
cannot establish general superiority, boundary safety or paper-result fidelity.

Stop after the measured Stage 4 report, validation, commit and push. Further
experiments or research changes require separate user instructions.
