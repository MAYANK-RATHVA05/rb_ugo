# Dataset provenance

## Benchmarks and toy fixtures

Imbalanced benchmark datasets let researchers compare methods under a shared
problem definition. Class proportions, feature types, duplicate records, and
missing values can materially affect these comparisons. File-level imbalance
statistics describe data; they do not demonstrate model performance.

KEEL is a repository and software ecosystem for machine-learning datasets and
experimentation, including imbalanced classification. Its `.dat` files can
declare attribute types, input columns, and output columns. Consult the dataset's
official documentation for its provenance and intended use. This project does
not yet specify the original UGO paper's experimental datasets or configurations.

Some KEEL datasets have specialized binary variants, for example a selected
class versus other classes or selected groups of classes. The variant defines
the problem, not just a different filename. Record the precise variant and
class grouping instead of treating it as interchangeable with the source dataset.

`tests/fixtures/toy.csv` and `toy.dat` are deliberately tiny, synthetic software
fixtures. They test parser behavior and count calculations only. They are not
<<<<<<< HEAD
research benchmarks, experimental results, or evidence for UGO or the provisional
DRA-UGO research direction.

## Current proposed research plan

Eight actual user-supplied full KEEL-format datasets now parse and have verified
file-level statistics/checksums. See [DATASET_VERIFICATION.md](docs/research/DATASET_VERIFICATION.md)
and [the machine-readable manifest](data/manifests/keel_dataset_manifest.csv).
None of the supplied archives includes official folds; real-fold UGO smoke checks
are blocked. Provider authentication, domain class grouping/license and record
identity remain unresolved. No predictive benchmark has run.

The provisional DRA-UGO simulations, label-budget feasibility and independent
A/B/C evaluation are in [DATASET_PLAN.md](docs/research/DATASET_PLAN.md).
They remain research plans, not completed experiments. Use the offline import
CLI in the verification document for Windows/Linux. Source ZIPs and extracted
raw files/PDFs stay ignored by Git; do not regenerate official folds or expose
outer-test rows to an unlabelled fitting pool.
=======
research benchmarks, experimental results, or evidence for UGO/RB-UGO.
>>>>>>> origin/main

## Adding external data manually

1. Obtain the dataset from a verified authoritative source and check its license
   and access conditions. No real data are downloaded automatically in Stage 2.
2. Place the original file under `data/raw/` without editing it to make the
   loader succeed. Use a separate dataset folder if the source includes splits.
3. Record the source/provider, exact source location, version or release, access
   date, filename, checksum, license, and any transformations in a separate
   provenance note. Never guess a citation or URL.
4. Record the target column, original label meanings, binary class grouping,
   and units/types of features. A lower-frequency label is not necessarily the
   scientifically meaningful positive outcome.
5. Preserve and document existing train/test splits, fold identifiers, and
   grouping or time constraints. Do not combine an outer test split with data
   used for training or selection. A future stage must decide evaluation and
   fitting partitions before fitting preprocessing.

Raw data, processed data, and generated results are ignored by Git. Do not
force-add real, private, or licensed raw datasets. Commit non-sensitive provenance
documentation only. JSON inspection reports include a source path, labels, and
counts (no sample records), so review them before sharing outside the project.

Stage 2 performs inspection only. It does not split, impute, encode features,
scale, resample, train, or evaluate models.
