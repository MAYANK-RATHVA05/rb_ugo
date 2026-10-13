---
title: "Controlled comparison of six resampling conditions on 30 binary datasets"
subtitle: "Supervisor briefing — UGO-SMOTE, BFCH-SMOTE, EC-EBO, SSHR, SMOTE and no resampling"
date: "13 October 2026"
---

# Purpose and main finding

This study compares four implemented paper-inspired methods—UGO-SMOTE, BFCH-SMOTE, EC-EBO and SSHR—against standard SMOTE and a no-resampling control on the **same 30 binary dataset variants**. The results show metric and classifier tradeoffs, not a universal winner. Standard SMOTE has the highest variant-average LogReg F1 (0.495) and G-mean (0.800); UGO-SMOTE has the highest variant-average GNB F1 (0.505) and MCC (0.486). The no-resampling control has the lowest average false-positive rate for both classifiers. In the SSHR ablation, supplying unlabelled training examples slightly raises G-mean but lowers F1 and raises FPR on average.

These are **descriptive results from our controlled implementation**, not a replication of the four papers' published scores. No new method or novelty claim is established here.

# Experimental setup

- **Inputs:** 30 numeric, no-missing binary variants from 10 source families; 9 Ecoli and 10 Yeast variants share related original data. Satellite class 4 versus rest and Abalone rings ≥18 versus rest are explicitly accepted approximations. Exact author preprocessing and folds are unverified.
- **Evaluation:** Five repeats of five stratified, coordinate-group-disjoint outer folds, seed 42. Within each outer training fold, approximately half of each class's rows remain labelled (`L`); the others form a training-only unlabelled pool (`U`). No outer-test row is used to fit preprocessing, resampling or a classifier.
- **Common comparison:** NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE, EC-EBO and SSHR use the same `L`, held-out fold and final classifier. Only SSHR may use `U`. Classifiers are logistic regression (`LogReg`) and Gaussian naive Bayes (`GNB`); their scores are reported separately.
- **SSHR ablation:** SSHR and NORES are also run without `U` on the same folds and `L`. The saved NORES outputs matched exactly for all 60 dataset/classifier conditions, supporting a controlled comparison.
- **Measures:** F1, G-mean, MCC and probability ROC-AUC are higher-is-better; FPR is lower-is-better. Each cell below is the unweighted average of 30 dataset-level means. Each dataset's six main methods use its same successful fold subset.

Source-family counts are Yeast 10, Ecoli 9, Glass 3, Wine Quality 2, and one each of Banana, Satellite, Abalone, New Thyroid, Page Blocks and Shuttle.

# Execution and coverage audit

The upload has 30 report folders, 1,800 canonical score rows (30 datasets × 2 classifiers × 6 methods × 5 metrics), and 300 SSHR ablation rows. The ZIP passed CRC and every included file matched its recorded SHA-256. The selection and source-registry hashes match the Windows checkout's CRLF line endings. The aggregate CSV matches the canonical rows in all 30 per-dataset score files exactly. This audit checks the **aggregate package**; individual saved job records and predictions were not uploaded.

| Run | Planned jobs | Successful | Failed |
| --- | ---: | ---: | ---: |
| Six-method main comparison | 9,000 | 8,994 | 6 |
| SSHR without-U ablation | 3,000 | 3,000 | 0 |

All six failures are standard SMOTE on `winequality_red_8_vs_6`: the same three folds fail under both classifiers because five labelled minority fitting rows are insufficient for its fixed five-neighbour setting (`n_neighbors = 6` requires at least six rows). The study did not change the sampler or redraw folds after seeing this. The six-way table uses **25 shared folds for 29 variants** and only **22 shared folds for this wine variant**. Its SSHR with/without-U ablation uses all 25 paired folds, so those two summaries have different fold scopes. The rarest test fold for this variant has one minority row, adding instability.

# Main comparison

| Classifier | Method | F1 ↑ | G-mean ↑ | MCC ↑ | ROC-AUC ↑ | FPR ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| LogReg | NORES | 0.426 | 0.487 | 0.431 | 0.866 | 0.008 |
| LogReg | SMOTE | 0.495 | 0.800 | 0.479 | 0.868 | 0.137 |
| LogReg | UGO-SMOTE | 0.475 | 0.582 | 0.462 | 0.862 | 0.047 |
| LogReg | BFCH-SMOTE | 0.489 | 0.778 | 0.466 | 0.858 | 0.126 |
| LogReg | EC-EBO | 0.492 | 0.794 | 0.473 | 0.868 | 0.125 |
| LogReg | SSHR | 0.421 | 0.689 | 0.406 | 0.856 | 0.119 |
| GNB | NORES | 0.487 | 0.653 | 0.471 | 0.846 | 0.061 |
| GNB | SMOTE | 0.482 | 0.771 | 0.465 | 0.838 | 0.153 |
| GNB | UGO-SMOTE | 0.505 | 0.685 | 0.486 | 0.847 | 0.074 |
| GNB | BFCH-SMOTE | 0.442 | 0.712 | 0.418 | 0.810 | 0.125 |
| GNB | EC-EBO | 0.439 | 0.762 | 0.421 | 0.832 | 0.167 |
| GNB | SSHR | 0.424 | 0.668 | 0.410 | 0.839 | 0.112 |

**Reading the tradeoff.** Under LogReg, SMOTE gains 0.069 F1 and 0.314 G-mean over NORES on average, but increases FPR by 0.128. UGO-SMOTE gains 0.048 F1 with a smaller 0.039 FPR increase; its G-mean gain is only 0.096. BFCH-SMOTE and EC-EBO have average LogReg F1 of 0.489 and 0.492, close to SMOTE's 0.495, with slightly lower FPR but no overall F1 advantage. Under GNB, UGO-SMOTE improves average F1 over NORES by 0.018 while SMOTE is lower by 0.005. EC-EBO and BFCH-SMOTE have lower average GNB F1 than both UGO-SMOTE and NORES.

These means conceal heterogeneity: UGO-SMOTE has the highest LogReg F1 on 13 of 30 variants even though SMOTE has the higher variant-average F1. A method's performance depends on the classifier and dataset. FPR should be read alongside recall-sensitive metrics; a very conservative model can achieve low FPR while missing minority examples.

# Questions addressed by the methods

**Boundary sampling and overlap (BFCH-SMOTE).** BFCH-SMOTE applies an observed-majority exclusion rule during its geometric generation. That does not guarantee low overlap on unseen data. The study's *training-only proxy* asks whether a generated minority point is nearer an observed majority row than an observed minority row. BFCH-SMOTE has a higher proxy rate than SMOTE on **26 of 30** variants, a lower rate on 3, and a tie on 1; its variant-average proxy rate is **0.141 versus 0.102** for SMOTE. These rates do not measure the true decision boundary or true class overlap. The current results therefore do **not** demonstrate the desired reduction in overlap.

**Intra-class imbalance (EC-EBO).** EC-EBO's entropy and clustering mechanism is implemented, but this comparison does not separately label minority subclusters or measure improvement by subcluster. Its mean LogReg G-mean is 0.794 versus SMOTE's 0.800; GNB G-mean is 0.762 versus 0.771. This study is insufficient to claim that it resolves intra-class imbalance, although dataset-specific improvements may warrant inspection.

**Unlabelled data (SSHR).** SSHR uses `U` to select and pseudo-label existing training observations and may generate new minority vectors through its SMOTE fallback. The ablation estimates what supplying `U` changed under the fixed protocol:

| Classifier | Metric | SSHR with U | SSHR without U | With − without |
| --- | --- | --- | --- | --- |
| LogReg | F1 | 0.421 | 0.438 | -0.016 |
| LogReg | G-mean | 0.689 | 0.676 | +0.013 |
| LogReg | MCC | 0.406 | 0.422 | -0.016 |
| LogReg | ROC-AUC | 0.856 | 0.861 | -0.005 |
| LogReg | FPR | 0.119 | 0.098 | +0.021 |
| GNB | F1 | 0.425 | 0.451 | -0.026 |
| GNB | G-mean | 0.669 | 0.657 | +0.012 |
| GNB | MCC | 0.411 | 0.437 | -0.026 |
| GNB | ROC-AUC | 0.840 | 0.839 | +0.001 |
| GNB | FPR | 0.112 | 0.089 | +0.024 |

Supplying `U` lowers variant-average F1 by 0.016 (LogReg) and 0.026 (GNB), while G-mean rises by about 0.013 and 0.012. FPR rises by about 0.022–0.024. F1 declines on 22/30 LogReg variants and 23/30 GNB variants, with one tie in each. Thus the present evidence does not support a general F1 benefit from `U`; it shows a recall/false-positive tradeoff. These ablation means use all 25 paired folds, including the three folds excluded from the six-method wine comparison.

# Interpretation limits and next discussion

The 30 entries are not 30 independent original datasets: Ecoli and Yeast alone contribute 19 entries. The averages are descriptive variant averages, not confidence intervals or independent-sample statistical tests. Repeated folds share observations, and the study uses only two classifiers, one labelled fraction, fixed method defaults and one seed. Paper-specific data preparation, hyperparameters, classifier sets and SSHR's original unlabelled-pool protocol are not established here. UGO-SMOTE remains a provisional, source-audited implementation; BFCH-SMOTE includes declared geometric repairs; EC-EBO is an audited interpretation; SSHR also contains explicit implementation choices. Consequently, differences from original paper tables are expected.

For a subsequent research method, define the primary target **before** tuning—for example, minority F1 subject to a maximum FPR—and evaluate boundary quality with a prespecified held-out or geometric overlap measure. The current BFCH proxy and SSHR ablation identify concrete limitations to discuss with the supervisor; they do not by themselves establish a new algorithm.

# Files and provenance

- [Three-page figure pack](figures/Imbalanced_Learning_30_Dataset_Figures.pdf) and [figure-reading guide](Imbalanced_Learning_30_Dataset_Figures.md): visual summaries of F1/FPR tradeoffs, per-variant consistency and the SSHR ablation.
- [30-dataset comparison table](Imbalanced_Learning_30_Dataset_Comparison.csv): 300 Excel-friendly rows, with all five metrics side by side for six methods, separately by classifier and dataset.
- [Detailed aggregate scores](Imbalanced_Learning_30_Dataset_Scores.csv): 1,800 rows with mean, sample SD, shared-fold count and paired difference from NORES.
- [Method coverage](Imbalanced_Learning_30_Dataset_Coverage.csv): planned, successful and failed fold events for every dataset, classifier and method.
- [SSHR unlabelled-pool ablation](Imbalanced_Learning_30_Dataset_SSHR_Ablation.csv): paired with/without-U effects for each dataset, classifier and metric.
- [Study protocol](../ALL_METHODS_30_LOCAL.md) and [fixed selection](../../data/manifests/all_methods_30.json).
- Uploaded review ZIP SHA-256: `05461d8e67ff55741982a0b8115fba80fcf6af0c7ff1ae8804e70699cca2b4be`. It is retained by the student as the original aggregate report bundle. Raw datasets and individual job outputs remain local.

Paper sources: Sağlam & Cengiz, UGO, *Knowledge-Based Systems* (2026), DOI 10.1016/j.knosys.2026.115886; Wen et al., BFCH-SMOTE, *Neurocomputing* (2026), DOI 10.1016/j.neucom.2026.134595; Tao et al., EC-EBO, *Knowledge-Based Systems* (2026), DOI 10.1016/j.knosys.2026.115544; Jiang et al., SSHR, *Expert Systems with Applications* (2023), DOI 10.1016/j.eswa.2023.119733. This report cites the studied methods; it does not assert exact author-code equivalence.
