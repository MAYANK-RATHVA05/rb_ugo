# Small BFCH–UGO comparison on two local datasets

This is a **descriptive software study**, not a reproduction of either paper's
benchmark tables. It uses two locally available KEEL binary variants whose names
and class counts match the cited benchmark metadata. Exact author rows and
preprocessing are unverified. The registered SHA-256 values below identify the
inputs actually used; raw inputs and per-fold job records remain local and
ignored by Git.

| Input | Rows | Negative / positive | Features used | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `winequality_red_8_vs_6.dat` | 656 | 638 / 18 | All 11 numeric features | `6b556e909321b1ce3476506e188aabeaa3dea4a36cfb800d66b6f0a7375c4356` |
| `vowel0.dat` | 988 | 898 / 90 | `F0`–`F9` acoustic features | `478dab0fcc87f8932e9481af3dd66dfa99601ba8b72c49d2525cba04b7f6cf76` |

Vowel's `SpeakerNumber` is used only to keep speakers separate across train and
test folds. `TT` and `Sex` are also excluded so the samplers do not interpolate
these integer identifiers or categories. This ten-feature study variant differs
from the 13-feature source table. Winequality uses shuffled stratified ten-fold
evaluation; Vowel uses shuffled stratified **group** ten-fold evaluation by
speaker. Both use seed 42. These are declared local folds, not author folds.

Each dataset has the same folds, training-only preprocessing, classifiers
(`LogReg`, `GNB`), and positive label (`positive`) for NORES, SMOTE,
UGO-SMOTE, and BFCH-SMOTE. UGO-SMOTE remains source-audited with unresolved
paper conventions; BFCH uses the documented geometric corrections and declared
defaults in [BFCH_METHOD.md](BFCH_METHOD.md). The run contains 160 successful
jobs: 2 datasets × 2 classifiers × 4 methods × 10 folds, with no failures or
missing folds. Each score below is a mean ± sample standard deviation over
the same ten test folds. Full IAM, MCC, F1, G-mean, and AUC aggregates and
paired differences from NORES are in [BFCH_SMALL_STUDY_SCORES.csv](BFCH_SMALL_STUDY_SCORES.csv).

| Dataset | Classifier | Method | F1 | G-mean |
| --- | --- | --- | ---: | ---: |
| Winequality | LogReg | NORES | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Winequality | LogReg | SMOTE | 0.223 ± 0.161 | 0.661 ± 0.373 |
| Winequality | LogReg | UGO-SMOTE | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Winequality | LogReg | BFCH-SMOTE | 0.197 ± 0.204 | 0.484 ± 0.431 |
| Winequality | GNB | NORES | 0.157 ± 0.260 | 0.211 ± 0.339 |
| Winequality | GNB | SMOTE | 0.116 ± 0.072 | 0.574 ± 0.324 |
| Winequality | GNB | UGO-SMOTE | 0.157 ± 0.260 | 0.211 ± 0.339 |
| Winequality | GNB | BFCH-SMOTE | 0.138 ± 0.125 | 0.474 ± 0.421 |
| Vowel acoustic | LogReg | NORES | 0.704 ± 0.297 | 0.785 ± 0.299 |
| Vowel acoustic | LogReg | SMOTE | 0.585 ± 0.258 | 0.802 ± 0.296 |
| Vowel acoustic | LogReg | UGO-SMOTE | 0.608 ± 0.264 | 0.807 ± 0.298 |
| Vowel acoustic | LogReg | BFCH-SMOTE | 0.632 ± 0.276 | 0.827 ± 0.303 |
| Vowel acoustic | GNB | NORES | 0.638 ± 0.266 | 0.822 ± 0.304 |
| Vowel acoustic | GNB | SMOTE | 0.548 ± 0.249 | 0.823 ± 0.293 |
| Vowel acoustic | GNB | UGO-SMOTE | 0.607 ± 0.274 | 0.816 ± 0.300 |
| Vowel acoustic | GNB | BFCH-SMOTE | 0.524 ± 0.209 | 0.798 ± 0.291 |

On Winequality, UGO-SMOTE generated **zero samples on all 20 jobs**: its
pre-generation noise detection left no eligible minority seeds. The UGO scores
therefore equal NORES. BFCH generated samples in all 40 BFCH jobs. On Vowel,
UGO generated samples in 19 of 20 jobs; the one zero-generation job was GNB.
There is no consistent improvement over NORES or SMOTE across classifiers and
metrics. Winequality has only 18 positive rows, yielding one or two positives
per test fold and unstable fold metrics. These observations do not establish
superiority, author-result fidelity, or absence of class overlap.

To regenerate the local run from the repository root after obtaining files at
the registered paths in `data/manifests/ugo_74_approximate.json`:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_bfch_small_study --output results\bfch-small-study
```

The script verifies input hashes, dimensions, and labels before evaluation.
It writes `results/bfch-small-study/overview.json` plus each dataset's
`manifest.json`, job records, `comparison.csv`, and `comparison.json`. Results
under `results/` are ignored by Git; this document and its aggregate CSV are
the shareable snapshot. The Winequality and Vowel run IDs are respectively
`96ec097b728a85171ac297f99fa527f740eab68f48f091bd6fb3561be59622a7`
and `706b86d45aba59da9b7d41e459d19b1efde8d7a7ffbaf3f0856ba2b8c0b2e6ca`.
