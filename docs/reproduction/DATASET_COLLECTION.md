# Dataset collection and accepted approximations

This is the historical preparation/verification record, not a live file
inventory. The 74-input checks below describe that collection stage. Raw inputs
are not committed, and their availability depends on the checkout; the latest
SSHR cloud checkout does not contain `data/raw/ugo74_approx/`. See the current
[dataset guide](../../DATASETS.md) before running a registered experiment.

All 74 entries in the paper's final dataset list have prepared local inputs.
The user accepted using the collected candidate variants even when their exact
preparation differs from the paper. This is an independent reproduction input
collection, not a claim of exact author-data reconstruction.

The [portable manifest](../../data/manifests/ugo_74_approximate.json) identifies
both the reported paper name and actual dataset ID, input SHA-256, target,
positive class, categorical columns, source provenance, and preparation choices.
Paths are relative to the repository root. Data are under
`data/raw/ugo74_approx/`, ignored by Git and not included in commits.

| Check | Count |
| --- | ---: |
| Inputs prepared and loaded | 74 |
| Source names/aliases and class counts matched | 56 |
| Explicitly accepted approximate variants | 18 |
| Inputs whose raw feature count differs from the supplement | 16 |

All 74 selected inputs match the supplement's sample and majority/minority counts.
Matching these statistics is not proof of identical feature values, class
definitions, author preprocessing, or experiment folds. The 16 raw feature-count
differences comprise 13 earlier KEEL sources and the two Fishing variants and
thyroid source. Fold-local encoded/filtering dimensions may also vary.

## Accepted variant definitions

| Paper entry | Actual independent input |
| --- | --- |
| Satellite2 | Landsat original class 4 versus rest, 36 features |
| Satellite3 | Landsat original class 5 versus rest, 36 features |
| abalone2 | Rings >= 18 versus rest; categorical Type omitted, seven features |
| Vowel1..Vowel10 | UCI nominal class indices 1..10 versus rest; nine numeric mlbench features |
| Fishing1 | Beach versus rest; all nine mlogit predictors retained |
| Fishing2 | Pier versus rest; all nine mlogit predictors retained |
| thyroid_disease | Exclude the single missing-age row; retain all 29 predictors |
| yeast_2_vs_8 | Explicit substitution with KEEL yeast-2_vs_4 |
| winequality_red_3_9_vs_5 | Explicit substitution with KEEL winequality-white-3-9_vs_5 |

Vowel numbering uses the ARFF nominal declaration order, not the alphabetical
R factor order:

| Paper entry | Proposed minority phonetic label |
| --- | --- |
| Vowel1 | hId |
| Vowel2 | hEd |
| Vowel3 | hAd |
| Vowel4 | hYd |
| Vowel5 | had |
| Vowel6 | hOd |
| Vowel7 | hod |
| Vowel8 | hUd |
| Vowel9 | hud |
| Vowel10 | hed |

Class index 0 (`hid`) is omitted from this ten-variant suite. The separately
collected KEEL vowel0 remains its own source dataset. The nine mlbench numeric
features correspond to UCI Features 0 and 2 through 9; Feature 1 is omitted in
that source version, and the speaker identifier is omitted from these candidates.
This complete mapping is a documented assumption rather than an author rule.

The two KEEL substitutions use actual dataset IDs in the manifest. Outputs must
retain those IDs and their reported-paper-name mapping; white-wine results must
not be presented as observations from the reported red-wine source.

## Sources

The uploaded publisher supplement contains the exact final 74-entry metadata and
published results, including a larger 84-dataset detailed-results table. Only the
final 74 entries are used here. Raw inputs come from uploaded KEEL bundles,
uploaded HTRU2 and Landsat sources, public CRAN mirrors, and a public ARFF
collection. Public source repositories are pinned by commit and source-file hash
where available; they are not claimed to be an official author-code release.

- KEEL: original dataset files, with documented header-only compatibility copies
  where singular `@input`/`@output` or empty input tokens need handling.
- HTRU2: CSV and ARFF rows agree; CSV headers added using ARFF attribute names.
- Banana: `cran/imbalance`, source object `banana`.
- Churn: `cran/modeldata`, source object `mlc_churn`, all 19 predictors retained.
- Vowel: `cran/mlbench`, checked against UCI-style vowel ARFF.
- Abalone: `cran/AppliedPredictiveModeling`.
- Fishing: `cran/mlogit`, with matching mode labels checked against `cran/Ecdat`.
- Thyroid: `renatopp/arff-datasets`, `classification/sick.arff`.

Original sources and detailed audits remain outside the repository; the source
register and dataset manifest preserve the evidence needed for review.

## Validation and evaluation limits

Each input was parsed with the repository loader and checked for a binary target,
sample and class counts, SHA-256, feature metadata, and at least ten minority
examples. The original source audits checked numeric CSV serialization, labels,
and candidate feature values. All data paths were checked against Git ignore
rules. No preprocessing was fitted, folds generated, synthetic samples produced,
models trained, or real benchmark scores calculated during collection.

An independent reload verified all 74 selected inputs against the saved manifest,
including hashes, explicit positive labels, sample/class counts, ignored data
paths, vowel assignments, and substitution provenance. The focused loader and
imbalance-statistics regression checks passed: **47 tests**.

Later experiments must fit imputation, categorical encoding, feature filtering,
and scaling on training folds only and oversample only training data. The
existing FoldPreprocessor policy remains an engineering convention. Independent
stratified ten-fold partitions must be recorded, rather than called author folds.
Original Landsat source partitions and row lineage remain preserved in its audit.

The simulation pilot is already complete, but the full real-data benchmark and
final statistical reports remain subsequent work. All fourteen classifier
interfaces now exist and remain declared approximations wherever author package
details are unknown. Dataset readiness does not close the remaining metric,
fold, and exact-source fidelity gaps.
