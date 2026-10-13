# SSHR small real-data study — Stage 4

The study protocol below is fixed before model scores are inspected. Its
execution and measured results will be recorded after software validation.
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

## Reproduction and limits

The source register will be `data/manifests/sshr_pilot_sources.json`, with
portable raw-file paths and expected hashes. The study entry point will be
`python -m scripts.run_sshr_small_study`; raw inputs and job records stay local
and ignored by Git. Shared reports will retain only descriptive summaries,
diagnostics and source/fold/mask identities.

Without tuned-threshold NORES controls, F1 changes alone cannot establish
improved discrimination rather than changed class weighting or decision behavior.
The empty-U control and probability ROC-AUC provide additional descriptive
evidence, not guarantees. The literal printed SSHC objective and declared
pruning, ranking, quota and SMOTE interpretations remain; author source code
and exact unlabelled-data protocol are unverified. Two datasets and two classifiers
cannot establish general superiority, boundary safety or paper-result fidelity.

Stop after the measured Stage 4 report, validation, commit and push. Further
experiments or research changes require separate user instructions.
