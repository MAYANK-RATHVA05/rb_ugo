# Data-Reducibility-Aware Uncertainty-Guided Oversampling for Imbalanced Classification

## 1. Research title

**DRA-UGO** is the provisional working name. The earlier **Risk-Budgeted UGO
(RB-UGO)** title is historical; neither is a validated new algorithm. Repository
`MAYANK-RATHVA05/rb_ugo` and existing `src.*` Python import paths stay unchanged.

## 2. Project overview

This B.Tech AI research project investigates when classifier-specific minority
augmentation can help, when threshold changes suffice, and when synthesis may
reinforce overlap or noise. Software Stages 1–3 are implemented; Stage 4A updates
research scope, mathematical analysis and planning. No DRA-UGO mechanism has been
finalized, implemented or benchmarked. The current audit concludes **no defensible
methodological novelty established** with the available evidence.

See [project scope](docs/research/PROJECT_SCOPE.md),
[novelty decision](docs/research/NOVELTY_DECISION.md) and
[reference status](docs/research/REFERENCE_STATUS.md).

## 3. Why class imbalance is difficult

Overall accuracy can hide missed minority observations. Rare minority subgroups
can be neglected even when aggregate recall looks adequate. Interpolation can
place minority-labelled points in ambiguous regions, while duplication can act
like weighting rather than supply independent information. A default decision
threshold can also cause avoidable minority errors without poor ranking.

These effects require separate tests. Threshold/ranking separation is documented
in scikit-learn [S1]; ordinary oversampling caveats in imbalanced-learn [S5];
model-dependent data value in Data Shapley [D]. Citations resolve to the
[verified source register](docs/research/PRIOR_ART_MATRIX.md#verified-reference-register-and-audit-limits).
The mathematical counterexamples are audit derivations, not benchmark findings.

## 4. Official problem statement: Questions A, B and C

- **A:** How can we generate minority samples close to the decision boundary
  without increasing harmful class overlap?
- **B:** How can we handle intra-class imbalance, where different regions or
  subgroups of the minority class contain unequal numbers of observations?
- **C:** Can unlabelled data be used to support synthetic minority sample generation?

Each question requires a specific methodology and independent evidence. Combining
three ordinary techniques does not establish a novel contribution.

## 5. Original UGO baseline

The local baseline provides provisional UGO-ROS/UGO-SMOTE with internal classifier
scoring, signed label-aware MMA, noise handling and stopping diagnostics. Its
cap, pools, rounding and neighborhood conventions remain prompt-based assumptions;
the full original paper was inaccessible. **Software tests do not establish
paper fidelity or published-result reproduction.** Preserve the baseline unchanged.

Read [the reproduction record](docs/ugo_reproduction.md). Runtime requires
`allow_provisional=True`. Original class selection can choose the majority class;
do not silently force minority-only generation or ordinary SMOTE behavior.

## 6. Updated research hypothesis

> Can a classifier-specific oversampling method identify local minority regions
> where additional synthetic training information improves genuine discrimination,
> distinguish those regions from threshold-recoverable or irreducibly overlapping
> regions, and use unlabelled feature-space information to improve these decisions?

This is a hypothesis. High uncertainty or low density alone does not prove data
limitation. Finite intervention estimates depend on learner, generator and data;
no benefit is not proof of irreducibility. Synthetic points derived from existing
inputs do not provide new independent labels. Unlabelled X describes structure,
not verified class membership. MMA is not a formal epistemic/aleatoric decomposition.
See [the mathematical analysis](docs/research/MATHEMATICAL_GAP_ANALYSIS.md).

## 7. Proposed methodology — explicitly provisional

Two mathematical candidates are assessed in
[CANDIDATE_HYPOTHESES.md](docs/research/CANDIDATE_HYPOTHESES.md):

1. Control-adjusted finite regional augmentation contrasts: independent risk,
   ranking and false-positive comparisons beyond threshold/weight/count controls,
   with explicit inconclusive outcomes. Retained as a diagnostic research question,
   not a demonstrated novel oversampler.
2. Anchor-compatible graph sensitivity intervals: evaluate structural consistency
   without treating pseudo-labels as truth. Rejected as a standalone novelty claim;
   graph machinery and unverified smoothness assumptions remain strong objections.

No arbitrary entropy/density/confidence product is introduced. A UGO-plus-clustering,
pseudo-label, boundary-filter or count-tuning combination is insufficient novelty.
Implementation needs Stage 4B source/design review and subsequent explicit approval.

## 8. Datasets and simulations

Proposed KEEL shortlist: Ecoli2, Glass6, Yeast3, Page-blocks0, Ecoli4 and Yeast4.
Additional possibilities: Segment0, Glass2, Abalone19, and Pima or Ecoli1 for the
user's proposed lower-imbalance stress test. **Names/variants, targets, counts,
feature schemas, licenses and predefined folds remain unverified.** None were
downloaded or evaluated in this stage.

Proposed simulations vary (A) known class overlap, (B) unequal minority subgroup
representation, (C) fitting-label noise and minority count, and (D) hidden fitting
labels at 10%, 25%, 50% and 100%. No outer features/labels enter U. Hidden truth
is diagnostic-only, and H/A labels count in the total visible-label budget.
[DATASET_PLAN.md](docs/research/DATASET_PLAN.md) specifies distributions,
provenance gaps, question-specific metrics and safeguards.

## 9. Evaluation and baselines

Planned learners: Logistic Regression, Random Forest and RBF-SVM if feasible.
Controls include no resampling with default/tuned thresholds, class weighting,
random oversampling, SMOTE, Borderline-SMOTE, original UGO, validation-tuned
synthetic count, appropriate advanced methods and faithfully implementable SSL
resampling. The future method is included only after finalization and approval.

Report AP as the declared AUPRC implementation, MCC, balanced accuracy, minority
recall, majority FPR, F1, supplementary AUROC, generated counts and computational
cost. Question A adds boundary/overlap diagnostics; B adds subgroup recall and
coverage; C isolates U value across label fractions. Match label, U and search
budgets. Use fitting-only preprocessing/resampling, independent inner selection,
untouched outer tests, ablations and prespecified statistical comparisons.
No benchmark results exist here.

## 10. Research papers

The six requested sources are UGO, SLEO, SSHR, GDEO, *Don't Oversample the Boundary*
and *Synthetic Augmentation in Imbalanced Learning*. Their exact supplied titles,
DOI/arXiv links and unresolved claims are in
[REFERENCE_STATUS.md](docs/research/REFERENCE_STATUS.md). **Neither full text nor
abstract was verified** because the environment blocked their URLs. Title and
method topics supplied in the brief are not independently verified metadata.

Additional full texts verified: Ren et al. (2018), *Learning to Reweight Examples
for Robust Deep Learning*; Ghorbani & Zou (2019), *Data Shapley*; Koh & Liang (2017),
*Understanding Black-box Predictions via Influence Functions*. None were reviewed
from abstracts only. The [matrix](docs/research/PRIOR_ART_MATRIX.md) records actual
author equations, assumptions and limits; pinned library sources are distinguished
from original-paper verification.

## 11. Current implementation status

| Stage | Implemented software / research status |
| --- | --- |
| 1 | Configuration/path checks and portable initialization |
| 2 | Strict binary CSV/KEEL loading and descriptive imbalance reports |
| 3 | Provisional UGO variants and artificial-data sanity checks |
| 4A | Evidence-limited mathematical audit and updated research documentation |
| 4B onward | Not approved or implemented; planned stages only |

No automatic benchmark acquisition, evaluation splitting, preprocessing pipeline,
final-model scoring or new oversampler is implemented. Internal Stage 3 fits are
software checks, not research experiments. UGO fidelity and result reproduction
remain pending. Existing entry points and tests are preserved.

## 12. Updated development roadmap

[RESEARCH_ROADMAP.md](docs/research/RESEARCH_ROADMAP.md) defines the approval gates:

- 1–3: existing setup, data infrastructure and provisional original UGO.
- 4A: literature verification, mathematical audit and documentation update.
- 4B: final mathematical method selection and research-design review.
- 5: leakage-free benchmark infrastructure and faithful UGO evaluation.
- 6: approved new mechanism, conditional on Stage 4B approval.
- 7: unlabelled feature integration and labelled-only controls.
- 8: boundary, intra-class imbalance, U benefit and ablations.
- 9: statistics, robustness, complexity and research writing.

All stages after 4A are provisional. Development/pilot data must be distinct from
final outer evaluation. Stop after this update for review; a roadmap is not
permission to implement.

## 13. Project structure

```text
rb_ugo/                     # Repository root, not a second nested directory
├── data/
│   ├── raw/.gitkeep         # Original inputs (future stages)
│   └── processed/.gitkeep   # Derived data (future stages)
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths and RANDOM_STATE = 42
│   ├── data_loader.py      # Strict CSV/KEEL binary loader
│   ├── imbalance_stats.py  # Descriptive counts and readable summary
│   ├── _validation.py     # Numerical input checks
│   ├── uncertainty.py     # Signed MMA
│   ├── noise_detection.py # Neighbor votes and relocation
│   ├── stopping.py        # Welch decision and provisional cap
│   └── ugo.py             # Provisional UGO variants and diagnostics
├── notebooks/.gitkeep      # Future exploratory notebooks
├── results/.gitkeep        # Future generated research outputs
├── tests/                 # Stage 1/2/3 tests and synthetic fixtures
├── main_stage1.py          # Directory checks and environment display
├── main_stage2.py          # Dataset inspection CLI
├── main_stage3.py          # Provisional UGO software sanity check
├── docs/
│   ├── ugo_reproduction.md # Historical provisional UGO reproduction record
│   └── research/          # Scope, roadmap, evidence, mathematics, dataset plan
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

## 14. Installation and execution instructions

### Antigravity setup on Windows (PowerShell)

Codex Cloud and Google Antigravity are separate environments. First synchronize
the project through Git; cloud edits are not automatically visible in your IDE.
The documentation-review branch is `research-stage4a-novelty-audit`; after its
PR merges, use `main`. From a fresh clone:

```powershell
git clone https://github.com/MAYANK-RATHVA05/rb_ugo.git
Set-Location rb_ugo
# For an existing clone, start with git fetch below in its repository folder.
git fetch origin
git switch research-stage4a-novelty-audit
git pull --ff-only
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

### Cloud / Linux setup

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

### Stage 2: inspect a dataset

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

#### What the numbers mean

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

#### Python interface and file support

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

#### Run Stage 2 in Antigravity on Windows

Use the synchronized branch chosen above (or `main` after the documentation PR merges):

```powershell
git fetch origin
git switch research-stage4a-novelty-audit
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

### Stage 3: provisional original-UGO implementation

Read [docs/ugo_reproduction.md](docs/ugo_reproduction.md) first. The implementation
follows the supplied signed MMA equation and highest-mean-uncertainty class rule,
with kNN noise detection, relocation, and Welch stopping. It does not silently
force minority-only generation or replace uncertainty selection with ordinary
SMOTE. The cap formula, selection pools, rounding, and neighborhood rules are
explicitly provisional because the full paper could not be read.

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
git switch research-stage4a-novelty-audit
git pull --ff-only
.\.venv\Scripts\python.exe main_stage1.py
.\.venv\Scripts\python.exe main_stage3.py
.\.venv\Scripts\python.exe -m pytest -q
```

Linux uses `.venv/bin/python main_stage3.py`. The demo's counts and stopping
diagnostics are software evidence only; it reports no accuracy or benchmark
results. Categorical/missing features require future training-only preprocessing
outside UGO. Already-converted numeric category codes cannot be detected by
dtype alone; callers must ensure numerical distance is meaningful. Stage 1/2
entry points continue to work unchanged. These commands exercise implemented
software; they do not implement or evaluate the proposed method.

## 15. Research limitations

The six core papers and authoritative KEEL variant/fold metadata remain
inaccessible. The collection of CFS-SMOTE, BFCH-SMOTE, EC-EBO and related methods
needs exact bibliographic sources. No exhaustive novelty search, benchmark
superiority, intrinsic error decomposition or safety guarantee has been established.
Small subgroup audit counts may make proposed bounds uninformative. Graph support
can be misleading under overlap, poor anchors or shift; validation adaptation can
overfit. No proposed method is ready for implementation on present evidence.

## 16. Reproducibility and citation guidance

Use Python 3.11 and the pinned runtime/development requirements; default seed is
42. Record data checksums/licenses, fold/mask IDs, transformations, learner and
generator parameters, tuning budgets, hardware and actual run status. Raw data,
credentials, environments, caches and generated results remain ignored by Git.

Cite verified original sources for mathematical claims; local implementation
and library-source evidence must be labelled accordingly. DRA-UGO is a provisional
working title, not a published method to cite as validated. Negative/inconclusive
findings and failed source access must remain visible. Check
[NOVELTY_DECISION.md](docs/research/NOVELTY_DECISION.md) before proceeding; Stage 4B
requires user review and approval.
