# Train and save an original-UGO pilot

This pipeline trains Logistic Regression with **NORES**, **UGO-ROS**, and
**UGO-SMOTE** on a single dataset. It implements a working training/evaluation
workflow, rather than the complete published benchmark. The source paper is
Sağlam and Cengiz, *Knowledge-Based Systems* 342 (2026), 115886,
[doi:10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886).
The [core audit](ugo_reproduction.md) distinguishes verified equations from
unresolved execution choices.

## Synchronize Antigravity on Windows

From an existing repository checkout, preserve any local changes, then run:

```powershell
git fetch origin
git switch ugo-working-training-pipeline
git pull --ff-only
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Skip environment creation if the project already has a working Python 3.11
`.venv`. Select `.venv\Scripts\python.exe` as Antigravity's interpreter.
Cloud and Windows are separate machines: pulling code does not copy datasets or
trained models between them. The source code uses portable `pathlib` paths.
Windows commands are documented here; runtime verification was performed on
Linux, and does not establish a completed local Windows installation.

## Place the data

Extract each supplied ZIP and copy its original `.dat` member into `data/raw/`:
`ecoli2.dat`, `ecoli4.dat`, `glass6.dat`, `yeast3.dat`, `yeast4.dat`,
`segment0.dat`, `page-blocks0.dat`, and `abalone19.dat`. The parser reads KEEL
`@inputs`/`@outputs` and encodes the majority label as 0, minority as 1. CSV
requires `--target COLUMN`; use `--minority-label LABEL` for a tied class count.
The label mapping is saved in the experiment report.

These are user-supplied binary KEEL files. Exact membership/versions in the
paper's 74-dataset collection remain unverified because its complete manifest
is deferred to supplementary material (Section 4.1.2, p.8). No dataset download
is performed by the training CLI. `data/raw/`, `data/incoming/`, uploaded papers
and generated `results/` are excluded from Git. Do not force-add them.

## Run the experiment

Windows PowerShell, from the repository root:

```powershell
.\.venv\Scripts\python.exe main_train.py --data data/raw/ecoli2.dat
```

Linux/cloud:

```bash
.venv/bin/python main_train.py --data data/raw/ecoli2.dat
```

CSV example:

```powershell
.\.venv\Scripts\python.exe main_train.py --data data/raw/example.csv --target class
```

Defaults: seed 42, test fraction 0.25, LR `max_iter=1000`, and UGO scoring-loop
safeguard `max_iter=100`. The shared stratified split requires at least two
observations of each class in training and one in test. Input feature metadata
and original labels are retained; raw dataset files are not edited.

Each method uses the same preprocessing and test split. Train-only filtering
removes features constant within either training class, following the paper's
stated zero-variance removal (Section 4.4, p.10). Removing all-missing or
single-observed-value columns before imputation is an explicit implementation
convention. Numeric features receive median imputation and standard scaling;
nominal features receive most-frequent imputation and one-hot encoding with
unknown categories ignored. These preprocessing choices are not claimed as the
authors' exact implementation.

UGO receives only the transformed training data. A fresh final LR is fitted on
all returned augmented rows. The last internal UGO scoring estimator is not
used as the final model. The same unchanged default decision threshold is used
for every method, with no selection from test results.

Abalone19 has a nominal `Sex` feature. Encoding makes the numerical pipeline
executable, but UGO-SMOTE may interpolate its one-hot coordinates into fractional
values. This is not a verified categorical-data rule from the paper. The first
real-data pilot uses the numerical Ecoli2 dataset.

## Choose an explicit generation limit

The main paper conflicts between Section 3.3 and Table 4. The default
`--cap-policy section_3_3` uses `floor(phi*K*Nmax-Noriginal)`; the alternative
`--cap-policy table_4` uses `floor(phi*Nmax)`. Both use fixed original training
counts, and final batches are clipped to the remaining budget. The fixed batch
is `ceil(lambda_step*Noriginal)`, with `lambda_step=0.1`; its original-total
denominator is verified in Section 3.4/Table 4. Rounding, clipping, reference
pools and degenerate-test safeguards are declared conventions. Diagnostic status
remains `paper_checked_assumptions_unresolved`, and the CLI explicitly opts in.

To specify a policy and a new output directory:

```powershell
.\.venv\Scripts\python.exe main_train.py --data data/raw/ecoli2.dat --cap-policy section_3_3 --output-dir results/ugo_training/my_ecoli2_run
```

An existing output directory is refused. Running without `--output-dir` creates
a uniquely named directory under `results/ugo_training/`. Keep custom outputs
under `results/` so they remain ignored by Git. Do not select a policy, seed or
sample count by repeatedly looking at the same test results.

## Output and metrics

Each run saves:

- `run.json`: dataset mapping, configurations, preprocessing choices, per-method
  metrics/diagnostics, SHA256 of the exact source file, source commit, dirty-tree
  status, Python/package versions and UTC run time.
- `split.json`: zero-based source row positions for the shared train/test split.
- `metrics.csv` and `predictions.csv`: method metrics and held-out predictions.
- `nores.joblib`, `ugo_ros.joblib`, `ugo_smote.joblib`: fitted raw-feature
  prediction pipelines containing preprocessing and the final classifier.
- `ugo_ros_diagnostics.json`, `ugo_smote_diagnostics.json`: complete generation
  history, selected classes, noise/relocation counts, cap and stopping reasons.

IAM uses the actual paper's Section 4.3 equation (p.9). For a confusion matrix
`C` with true labels on rows and predictions on columns, it averages
`(Cii-max(row_errors_i,column_errors_i))/max(row_total_i,column_total_i)`.
It is distinct from balanced accuracy. MCC and positive-class F1 follow the
paper; the pilot additionally reports balanced accuracy, minority precision/
recall and non-interpolated average precision (`average_precision_score`,
reported as AUPRC). Class 1 is the positive minority class. Undefined precision
or F1 uses zero, and sklearn's constant-prediction MCC convention is zero.
Confusion matrices use label order `[0,1]`.

## Load a trained model

Use the same checkout and dependency versions recorded in `run.json`. Replace
the example run directory with the one printed by the CLI:

```python
from pathlib import Path
import joblib
from src.data_loader import load_dataset

run_dir = Path("results/ugo_training/my_ecoli2_run")
model = joblib.load(run_dir / "ugo_smote.joblib")
dataset = load_dataset("data/raw/ecoli2.dat")
predictions = model.predict(dataset.X.iloc[:5])
minority_probabilities = model.predict_proba(dataset.X.iloc[:5])[:, 1]
```

The pipeline accepts raw feature DataFrames with the original named columns.
Do not scale them a second time or include the target column. The example shows
inference; the stored test positions provide the independent evaluation set.

## Interpretation and remaining reproduction work

A single stratified holdout demonstrates the workflow and supplies descriptive
results; it does not establish general superiority. Do not choose a winning
configuration from these held-out scores and then reuse them as unbiased evidence.
If model/threshold selection is added later, reserve separate selection data or
use nested evaluation.

The published protocol includes 74 benchmarks, two principal simulations,
additional overlap experiments, 14 classifiers, more resampling conditions,
ten-fold evaluation and statistical comparisons (Sections 4–5). Their complete
reproduction still requires the supplementary manifest/results, missing methods
and classifiers, and clarification of execution ambiguities. No extension,
risk-budget mechanism or novelty claim is included in this pipeline.
