# Risk-Budgeted Uncertainty-Guided Oversampling (RB-UGO)

## Scope: original UGO implementation and first training pipeline

This portable Python 3.11 research repository provides configuration and directory
validation (Stage 1), plus binary CSV/KEEL loading and descriptive imbalance
analysis (Stage 2). Stage 3 introduced UGO-ROS/UGO-SMOTE and an artificial-data
sanity check. The supplied full paper has now been checked, the generation batch
has been corrected, and a working single-dataset training pipeline has been added.
It fits training-only preprocessing, generates training observations, trains fresh
final Logistic Regression classifiers, evaluates a shared untouched holdout, and
saves models and audit records. The paper's cap conflict and several execution
conventions remain explicit assumptions. The full 74-dataset benchmark and
Risk-Budgeted UGO are not implemented.

Start with the [training guide](docs/ugo_training.md). Place extracted KEEL
`.dat` files in `data/raw/`; dataset ZIPs are not accepted by the training CLI.

## Research motivation

In imbalanced classification, a minority class has fewer observations than the
majority class. Overall accuracy can hide poor minority detection; evaluation
must consider both classes and the costs of errors.

Basic oversampling can duplicate observations, encourage overfitting, or produce
synthetic points in ambiguous or noisy regions. More minority examples do not
necessarily improve generalization, and false positives can increase.

The base paper, **“Uncertainty guided oversampling: A classifier-specific approach
for effective handling of class imbalance in machine learning,”** motivates using
classifier uncertainty to guide oversampling for a particular classifier. A
faithful implementation requires examining the original algorithm and protocol.
The supplied full text is documented in [the reproduction audit](docs/ugo_reproduction.md).

The current priority is to implement and evaluate original UGO before considering
extensions. This pipeline compares the same classifier with no resampling,
UGO-ROS, and UGO-SMOTE. It makes no claim of methodological novelty or general
superiority and uses no test observations to select thresholds or generation.

## Layout

```text
rb_ugo/                     # Repository root, not a second nested directory
├── data/
│   ├── raw/.gitkeep         # Local original .dat/.csv inputs; ignored by Git
│   └── processed/.gitkeep   # Local derived data
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths and RANDOM_STATE = 42
│   ├── data_loader.py      # Strict CSV/KEEL binary loader
│   ├── imbalance_stats.py  # Descriptive counts and readable summary
│   ├── _validation.py     # Numerical input checks
│   ├── uncertainty.py     # Signed MMA
│   ├── noise_detection.py # Neighbor votes and relocation
│   ├── stopping.py        # Welch decision and provisional cap
│   ├── ugo.py             # Paper-checked UGO variants; explicit assumptions
│   └── training.py        # Train-only preprocessing, final models and metrics
├── notebooks/.gitkeep      # Future exploratory notebooks
├── results/.gitkeep        # Future generated research outputs
├── tests/                 # Stage 1/2/3 tests and synthetic fixtures
├── main_stage1.py          # Directory checks and environment display
├── main_stage2.py          # Dataset inspection CLI
├── main_stage3.py          # Provisional UGO software sanity check
├── main_train.py           # Real-data training CLI and artifact export
├── docs/ugo_reproduction.md # Equations, assumptions, evidence, unresolved fidelity
├── docs/ugo_training.md    # Windows/Linux training and model loading
├── requirements.txt       # Pinned research dependencies
├── requirements-dev.txt   # Runtime dependencies plus pytest
├── README.md
├── DATASETS.md             # Benchmark provenance and manual data handling
├── AGENTS.md               # Permanent research rules
└── .gitignore
```

Data and generated results are ignored by Git; directory markers are retained.
Paths are resolved from the source location, independently of the shell's current
directory. Initialization checks directories and never creates missing ones.

## Antigravity setup on Windows (PowerShell)

Codex Cloud and Google Antigravity are separate environments. First synchronize
the project through Git; cloud edits are not automatically visible in your IDE.
For a new checkout of the training branch:

```powershell
git clone --branch ugo-working-training-pipeline https://github.com/MAYANK-RATHVA05/rb_ugo.git
Set-Location rb_ugo
# For an existing clone: git fetch origin; git switch ugo-working-training-pipeline;
# then git pull --ff-only. Preserve any existing local changes before switching.
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe main_stage1.py
.\.venv\Scripts\python.exe -m pytest -q
```

Install Python 3.11 if `py -3.11` is unavailable. In Antigravity, choose **Open
Folder** and select this repository root (the folder containing this README).
Select `.venv\Scripts\python.exe` as the Python interpreter when prompted.
Run the commands above in its PowerShell terminal. Explicit interpreter paths
avoid activation and PowerShell execution-policy changes.

To check the installed dependencies:

```powershell
.\.venv\Scripts\python.exe -c "import numpy, pandas, sklearn, imblearn, scipy, matplotlib, seaborn, tqdm, pytest; print('Dependency imports successful.')"
```

## Cloud / Linux setup

With Python 3.11 available, from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python main_stage1.py
.venv/bin/python -m pytest -q
```

No credentials or external services are needed for Stage 1 beyond package
installation. The initialization script prints the Python version and configured
paths, then reports success only if all required directories exist. Tests cover
path configuration, successful initialization, and missing or invalid directories.

## Stage 2: inspect a dataset

From the repository root, CSV needs an explicit target (the column containing
class labels). KEEL normally identifies it through a single `@outputs` entry:

```bash
.venv/bin/python main_stage2.py --help
.venv/bin/python main_stage2.py --data tests/fixtures/toy.csv --target class
.venv/bin/python main_stage2.py --data tests/fixtures/toy.dat
.venv/bin/python main_stage2.py --data tests/fixtures/toy.csv --target class --save-json csv_report.json
.venv/bin/python main_stage2.py --data tests/fixtures/toy.dat --save-json keel_report.json
```

For your own manually downloaded files, substitute `data/raw/example.csv` or
`data/raw/example.dat`. Nothing downloads real datasets automatically. Read
[DATASETS.md](DATASETS.md) before adding benchmark data.

`--save-json` optionally takes a simple `.json` filename. Without a filename it
uses `<dataset-stem>_report.json`. Reports always go into `results/stage2/`, include
metadata and counts but no sample rows, and never overwrite existing reports.
Use a new filename or deliberately remove an old local report to rerun. Identical
CSV and DAT stems otherwise share the default output name. Reports are ignored
by Git. Errors print a readable message and return a nonzero exit code.

### What the numbers mean

The majority class is the more frequent class, encoded as **0**. The minority
class is the less frequent class, encoded as **1**. Original labels are retained,
and the mapping is reported. A tie has no automatic minority: use
`--minority-label "rare"` to designate a class. An explicit label must exist and
cannot select a class with a greater count than the other class. These are
file-level inspection labels; later evaluation must define the positive outcome
and class roles from fitting data or an explicit research definition, never from
outer-test data to guide training or selection.

The **imbalance ratio** is majority count divided by minority count. For example,
90 majority and 10 minority records give 9.0. Our six-record toy fixtures give
5.0; this is a software-test example, not a research result.

The summary also counts samples, features, class percentages, numerical and
categorical columns, missing feature entries, and rows with any missing features.
Duplicate feature rows compare features alone; duplicate complete rows compare
features **and the target**. Both count additional occurrences after the first,
so three identical rows yield two duplicates. Matching missing values are treated
as equal in these comparisons. Different labels with identical features increase
feature duplicates but do not necessarily increase complete-row duplicates.
No missing values or duplicates are removed.

### Python interface and file support

```python
from src.data_loader import load_dataset
from src.imbalance_stats import calculate_statistics

dataset = load_dataset("tests/fixtures/toy.csv", target_column="class")
# dataset.X: feature DataFrame; dataset.y: aligned integer target Series
# dataset.original_y: original text labels; dataset.label_mapping: label -> 0/1
stats = calculate_statistics(dataset)
print(stats.to_dict())
```

`LoadedDataset` documents feature names/types, majority/minority labels/counts,
target name, source filename, and format. `ImbalanceStats` documents all 13
statistics. Loading does not mutate the file, reorder or remove rows, or fit
anything.

- Files must be UTF-8 (a UTF-8 BOM is supported), with `.csv` or `.dat` suffixes.
- CSV uses comma separators, a unique nonempty header, and standard double-quoted
  fields, including embedded commas, doubled double quotes, and multiline fields.
  A feature is numerical if every nonmissing value parses numerically; otherwise
  it is categorical. All-missing CSV columns are categorical because type cannot
  be inferred. Numeric-looking category codes/identifiers may therefore need a
  future explicit schema; CSV does not carry declared feature types.
- KEEL accepts `@relation`, `@attribute`, `@inputs`, `@outputs`, and `@data`,
  with case-insensitive directives; real/numeric/integer and declared nominal
  attributes; and optional numeric range metadata. Declared ranges are parsed
  but not used to filter or clip values. Integers and finite numerical values are
  validated; undeclared nominal values fail.
- KEEL supports full-line `%` comments, comma-delimited records, single/double
  quoted names/values, and doubled matching quotes. Sparse records, multiline
  records, inline comments, and backslash escapes are unsupported. This is a
  documented KEEL subset, not a general ARFF parser.
- KEEL respects `@inputs` when present; otherwise non-target attributes become
  features. `--target` overrides KEEL output selection and makes all other
  declared attributes features. Multiple declared outputs remain unsupported,
  including with an override. Missing output metadata requires an override.
- Empty fields and `?` represent missing values. Text such as `NA` is preserved
  as text, not silently treated as missing. Numeric infinity is rejected.
- Target labels remain strings, so `01` and `1` stay distinct. Missing targets,
  empty data, one-class or multiclass targets, inconsistent record widths,
  malformed declarations, and target-only datasets are rejected. Columns with
  missing features remain available without imputation or feature encoding.
- The loader reads the whole file into memory. No schema override, streaming,
  multiclass support, or original train/test split interpretation is implemented.

### Run Stage 2 in Antigravity on Windows

Synchronize the Stage 2 feature branch first (or pull `main` after its PR merges):

```powershell
git fetch origin
git switch stage2-data-loading
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe main_stage1.py
.\.venv\Scripts\python.exe main_stage2.py --help
.\.venv\Scripts\python.exe main_stage2.py --data tests/fixtures/toy.csv --target class
.\.venv\Scripts\python.exe main_stage2.py --data tests/fixtures/toy.dat --save-json keel_report.json
.\.venv\Scripts\python.exe -m pytest -q
```

If this is your first local setup, use the Python 3.11 environment-creation steps
above before these commands. Open this repository root in Antigravity and select
`.venv\Scripts\python.exe`. Linux cloud validation does not establish that the
Windows environment has already been tested or synchronized.

## Stage 3: provisional original-UGO implementation

Read [docs/ugo_reproduction.md](docs/ugo_reproduction.md) first. The implementation
follows the supplied signed MMA equation and highest-mean-uncertainty class rule,
with kNN noise detection, relocation, and Welch stopping. It does not silently
force minority-only generation or replace uncertainty selection with ordinary
SMOTE. The full paper has now been inspected; selection pools, rounding, and
neighborhood conventions remain explicit assumptions. Choose the conflicting
published cap through `cap_policy="section_3_3"` or `"table_4"`.

```python
from sklearn.linear_model import LogisticRegression
from src.ugo import UGO

# X_fit/y_fit must be fitting data only: finite numerical features, two classes.
sampler = UGO(LogisticRegression(max_iter=1000, random_state=42),
              variant="smote", allow_provisional=True)
# X_augmented, y_augmented = sampler.fit_resample(X_fit, y_fit)
# print(sampler.diagnostics_)
```

Run the included artificial-data sanity check on Windows after synchronizing:

```powershell
git fetch origin
git switch ugo-working-training-pipeline
git pull --ff-only
.\.venv\Scripts\python.exe main_stage1.py
.\.venv\Scripts\python.exe main_stage3.py
.\.venv\Scripts\python.exe -m pytest -q
```

Linux uses `.venv/bin/python main_stage3.py`. The demo's counts and stopping
diagnostics are software evidence only; it reports no accuracy or benchmark
results. `main_train.py` supplies training-only preprocessing outside UGO.
Already-converted numeric category codes cannot be detected by
dtype alone; callers must ensure numerical distance is meaningful. Stage 1/2
entry points continue to work unchanged. Only the original-UGO training pilot is
authorized; further paper-reproduction stages and extensions are separate work.

## Train on a real dataset

Extract your dataset archive and copy `ecoli2.dat` into `data/raw/`. On Windows:

```powershell
.\.venv\Scripts\python.exe main_train.py --data data/raw/ecoli2.dat
```

On Linux:

```bash
.venv/bin/python main_train.py --data data/raw/ecoli2.dat
```

The command reports IAM, MCC, F1 and minority recall for all three methods. It
saves prediction pipelines, metrics, holdout predictions, split indices and UGO
diagnostics in a new ignored directory under `results/ugo_training/`. See
[the training guide](docs/ugo_training.md) for reproducibility and inference.

The first [Ecoli2 pilot report](docs/experiments/ecoli2_pilot.md) records actual
training results. On its fixed holdout UGO increased minority recall while
increasing false positives and reducing F1, MCC and IAM. These are pilot results,
not a claim of overall superiority or a complete paper reproduction.
