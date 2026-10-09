# Risk-Budgeted Uncertainty-Guided Oversampling (RB-UGO)

## Scope: Stages 1 and 2

This portable Python 3.11 research repository provides configuration and directory
validation (Stage 1), plus binary CSV/KEEL loading and descriptive imbalance
analysis (Stage 2). It does not preprocess, split, resample, train, calculate model
performance, or implement UGO/RB-UGO. Further stages require explicit approval.

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
faithful implementation requires examining the original algorithm and protocol;
this scaffold does not claim to reproduce it yet. Bibliographic metadata and
algorithm details must be verified before adding citations or implementation.

Our proposed research question is whether accepting or rejecting uncertainty-guided
synthetic minority batches according to held-out balanced risk and majority-class
false-positive constraints improves reliability. This is a hypothesis, not a
claim of novelty or superiority. Literature review and controlled experiments,
including the original no-resampling baseline and faithfully reproduced UGO,
are required. Selection data must remain separate from the outer test data.

## Layout

```text
rb_ugo/                     # Repository root, not a second nested directory
├── data/
│   ├── raw/.gitkeep         # Original inputs (future stages)
│   └── processed/.gitkeep   # Derived data (future stages)
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths and RANDOM_STATE = 42
│   ├── data_loader.py      # Strict CSV/KEEL binary loader
│   └── imbalance_stats.py  # Descriptive counts and readable summary
├── notebooks/.gitkeep      # Future exploratory notebooks
├── results/.gitkeep        # Future generated research outputs
├── tests/                 # Stage 1/2 tests and synthetic fixtures
├── main_stage1.py          # Directory checks and environment display
├── main_stage2.py          # Dataset inspection CLI
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
Once the Stage 1 files have been committed and pushed to GitHub:

```powershell
git clone https://github.com/MAYANK-RATHVA05/rb_ugo.git
Set-Location rb_ugo
# For an existing clone, use git pull on the branch containing Stage 1 instead.
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
