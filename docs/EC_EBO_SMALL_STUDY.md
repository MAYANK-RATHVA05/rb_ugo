# Small paired EC-EBO study on two local datasets

This is a descriptive evaluation of the repository's **audited EC-EBO
interpretation**, not a reproduction of the article's benchmark table. It
uses the two locally available KEEL binary variants from the earlier
[BFCH small study](BFCH_SMALL_STUDY.md). Their names and class counts match
registered benchmark metadata, but exact author rows and preprocessing are
unverified. The raw inputs and per-fold job records remain local and are
ignored by Git.

| Input | Rows | Negative / positive | Features used | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `winequality_red_8_vs_6.dat` | 656 | 638 / 18 | All 11 numeric features | `6b556e909321b1ce3476506e188aabeaa3dea4a36cfb800d66b6f0a7375c4356` |
| `vowel0.dat` | 988 | 898 / 90 | `F0`–`F9` acoustic features | `478dab0fcc87f8932e9481af3dd66dfa99601ba8b72c49d2525cba04b7f6cf76` |

Vowel's `SpeakerNumber` defines nonoverlapping speaker groups across training
and test folds. `TT`, `SpeakerNumber`, and `Sex` are excluded from model
features; the resulting ten-feature variant differs from the source table's
13-feature input. Winequality uses shuffled stratified ten-fold splits; Vowel
uses shuffled stratified **group** ten-fold splits. Both use seed 42, and
preprocessing, logistic entropy fitting, oversampling, and final classifier
training use only fitting rows. These are declared local folds, not author
folds. The methods are NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE, and EC-EBO, with
the same folds, scaling, and classifiers (`LogReg`, `GNB`). No model or sampler
parameters were selected from test scores. There are 200 successful jobs and
no failed or missing jobs.

Each value below is the mean of the same ten held-out folds. FPR is
`FP / (TN + FP)`, using the declared negative/positive label order.
[EC_EBO_SMALL_STUDY_SCORES.csv](EC_EBO_SMALL_STUDY_SCORES.csv) contains the
sample standard deviations, shared-fold counts, run IDs, and source hashes.
The local per-dataset `comparison.csv` files additionally contain IAM and
paired differences from NORES.

| Dataset | Classifier | Method | F1 | G-mean | MCC | AUC | FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Wine | LogReg | NORES | 0.000 | 0.000 | -0.005 | 0.874 | 0.005 |
| Wine | LogReg | SMOTE | 0.223 | 0.661 | 0.254 | 0.890 | 0.139 |
| Wine | LogReg | UGO-SMOTE | 0.000 | 0.000 | -0.005 | 0.874 | 0.005 |
| Wine | LogReg | BFCH-SMOTE | 0.197 | 0.484 | 0.200 | 0.861 | 0.106 |
| Wine | LogReg | EC-EBO | 0.192 | 0.544 | 0.197 | 0.876 | 0.132 |
| Wine | GNB | NORES | 0.157 | 0.210 | 0.141 | 0.784 | 0.025 |
| Wine | GNB | SMOTE | 0.116 | 0.574 | 0.129 | 0.701 | 0.215 |
| Wine | GNB | UGO-SMOTE | 0.157 | 0.210 | 0.141 | 0.784 | 0.025 |
| Wine | GNB | BFCH-SMOTE | 0.138 | 0.474 | 0.148 | 0.621 | 0.114 |
| Wine | GNB | EC-EBO | 0.135 | 0.662 | 0.166 | 0.782 | 0.216 |
| Vowel | LogReg | NORES | 0.704 | 0.785 | 0.702 | 0.964 | 0.023 |
| Vowel | LogReg | SMOTE | 0.585 | 0.802 | 0.572 | 0.958 | 0.088 |
| Vowel | LogReg | UGO-SMOTE | 0.608 | 0.807 | 0.596 | 0.957 | 0.081 |
| Vowel | LogReg | BFCH-SMOTE | 0.632 | 0.827 | 0.617 | 0.957 | 0.094 |
| Vowel | LogReg | EC-EBO | 0.572 | 0.801 | 0.556 | 0.950 | 0.106 |
| Vowel | GNB | NORES | 0.638 | 0.822 | 0.633 | 0.955 | 0.063 |
| Vowel | GNB | SMOTE | 0.548 | 0.823 | 0.549 | 0.954 | 0.125 |
| Vowel | GNB | UGO-SMOTE | 0.607 | 0.816 | 0.600 | 0.964 | 0.086 |
| Vowel | GNB | BFCH-SMOTE | 0.524 | 0.798 | 0.508 | 0.957 | 0.135 |
| Vowel | GNB | EC-EBO | 0.513 | 0.804 | 0.509 | 0.951 | 0.139 |

EC-EBO's cluster and generation diagnostics are identical across the two
classifiers because its entropy model and random seed do not depend on the
downstream classifier. The totals below therefore count **one classifier's
ten-fold run**, not both copies combined. An exact duplicate is a generated
row equal to an original fitting row or an earlier generated row.

| Dataset | Mean clusters/fold | Original minority cluster sizes | Quota | Generated | Shortfall | Exact duplicates | Nearest-majority proxy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Wine | 14.6 | 1–2 | 5,580 | 5,580 | 0 | 251 | 80 / 5,580 (1.43%) |
| Vowel | 13.2 | 1–12 | 7,272 | 7,222 | 50 | 157 | 0 / 7,222 (0%) |

On Vowel, 18 cluster-fold combinations missed a quota; all 50 missed samples
had no positive-side seed for their selected majority reference. Most affected
clusters had one or two original minority rows. Their quotas were **not**
transferred to other clusters. The proxy counts synthetic rows whose nearest
*original fitting-set* majority point is strictly closer than the nearest
original minority point. It uses no held-out rows and cannot prove that
majority overlap is absent. It also shows that a positive Eq. 8 score did not
prevent every Winequality sample from lying closer to an original majority
row. Per-fold sizes, mean entropies, quotas, generated counts, and shortfall
reasons are in local `ec_ebo_clusters.csv` files.

EC-EBO did not improve all outcomes. On Winequality with GNB, its G-mean rose
from 0.210 (NORES) to 0.662, while FPR rose from 0.025 to 0.216. On Vowel,
EC-EBO's F1 was below NORES for both classifiers. Winequality has just 18
positive rows, leaving only one or two positives in each test fold; its
fold-level scores are unstable. Two datasets and two classifiers cannot
establish general superiority, boundary safety, or exact paper fidelity.

To regenerate this run from the repository root with the registered local
files present:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_ec_ebo_small_study --output results\ec-ebo-small-study
```

The script checks source hashes, dimensions, and labels. It writes
`results/ec-ebo-small-study/overview.json` and, under each dataset directory,
`manifest.json`, per-job JSON, `comparison.csv`, `comparison.json`,
`fpr.csv`, `ec_ebo_clusters.csv`, and `ec_ebo_diagnostics.csv`. Results under
`results/` are ignored by Git. The Wine and Vowel run IDs are respectively
`24127f955f0ecef0528fd12338993b97546a0bed6122a28761b965f3f8e8a563`
and `d2d9abab9f64cfa99fa58e2fea5fde22db4d2ece567a3818a2436430f1043a24`.
The four preexisting methods reproduce all 80 earlier per-metric means exactly.
