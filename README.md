# Imbalanced Learning Reproductions

Python 3.11 implementations, source audits and reproducible comparisons of
published resampling methods: **UGO, BFCH-SMOTE, EC-EBO and SSHR**. The current
work reproduces existing papers with explicit interpretations where their
operational details are ambiguous. The earlier risk-budget and CSG extension
proposals are inactive; this repository makes no methodological novelty claim.
See [project scope](docs/PROJECT_SCOPE.md) for the current boundaries.

The GitHub repository is currently
[`MAYANK-RATHVA05/rb_ugo`](https://github.com/MAYANK-RATHVA05/rb_ugo).
`imbalanced-learning-reproductions` is the proposed repository name; the rename
has not been applied. [Repository rename instructions](docs/REPOSITORY_RENAME.md)
record the remaining action and how to update existing clones afterward.

## Implementations and branch status

**Use `sshr-implementation` for all four methods and the latest pilot.**
`main` currently contains UGO, BFCH-SMOTE and EC-EBO code. SSHR code, its pilot
artifacts and the supervisor briefing remain on `sshr-implementation`; links
to those materials below explicitly select that branch.

| Method | Implemented work | Evidence and limits |
| --- | --- | --- |
| UGO | Classifier-guided UGO-ROS, UGO-SMOTE, UGO-RWO and UGO-MCSMOTE; noise handling, stopping, fresh final-model evaluation and resumable jobs | [Paper alignment](docs/ugo_reproduction.md), [generator decisions](docs/reproduction/GENERATOR_ALIGNMENT.md), [300-job simulation pilot](docs/reproduction/SIMULATION_PILOT.md). Paper ambiguities and classifier approximations prevent exact author-code/result equivalence claims. |
| BFCH-SMOTE | Boundary fusion and convex-hull sampling, explicit geometric corrections, opt-in runner integration | [Method audit](docs/BFCH_METHOD.md) and [160-job two-dataset comparison](docs/BFCH_SMALL_STUDY.md). Descriptive local results, not the paper's benchmark. |
| EC-EBO | Externally constrained minority clustering, entropy allocation, minority generation and opt-in runner integration | [Method decisions](docs/EC_EBO_METHOD_DECISIONS.md) and [200-job two-dataset comparison](docs/EC_EBO_SMALL_STUDY.md). Source ambiguities and quota shortfalls are retained. |
| SSHR | Semi-supervised hierarchical clustering, majority undersampling, unlabelled-point selection, SMOTE fallback and a three-input estimator | [Method decisions](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/SSHR_METHOD_DECISIONS.md), [integration](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/SSHR_INTEGRATION.md) and [800-job two-dataset pilot](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/SSHR_SMALL_STUDY.md). Author code and the original unlabelled-pool protocol remain unverified. |

**SSHR mainly selects existing unlabelled observations and assigns minority
pseudo-labels. Only its SMOTE fallback generates new feature vectors.**
Diagnostics distinguish retained rows, removed majority/minority rows, selected
pseudo-labelled observations, generated vectors and unmet quotas.

The original UGO algorithm and existing default method lists are preserved.
BFCH-SMOTE, EC-EBO and SSHR are selected explicitly. Source audits distinguish
verified paper rules, verified author behavior when available, and declared
implementation choices.

## Latest measured study

The [SSHR pilot report](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/SSHR_SMALL_STUDY.md)
compares NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE, EC-EBO and SSHR on **Ecoli4 and
Yeast4**, using LogReg/GNB, five repeats of five shared outer folds and the same
reduced labelled training set. Its 600 main-study jobs and 200 empty-U controls
all succeeded. The complete branch passed **738 tests** in the recorded Linux
environment with the optional classifier backends installed.

Adding the unlabelled pool reduced mean F1 in all four dataset/classifier
comparisons. Yeast4 with GNB showed higher mean probability ROC-AUC and G-mean
alongside higher false-positive rate. Retrospective pseudo-label accuracy was
**54.3% on Ecoli4** and **35.2% on Yeast4**, weighted over selection events within
each dataset/classifier condition. These events are not unique observations;
SSHR diagnostics match across classifier copies and must not be double-counted.
The two-dataset pilot establishes neither general superiority nor exact paper
result reproduction.

Read the [CSV/JSON artifact guide](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/sshr_pilot/README.md)
for metric definitions, coverage, counts, cluster records, source hashes and
ablation denominators. The
[supervisor briefing PDF](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/presentation/Imbalanced_Learning_Pilot_Results.pdf)
and [editable Word copy](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/presentation/Imbalanced_Learning_Pilot_Results.docx)
summarize the same pilot without adding experiments.

## UGO benchmark progress

The [dataset collection report](docs/reproduction/DATASET_COLLECTION.md) and
[portable manifest](data/manifests/ugo_74_approximate.json) document a historical
74-entry input collection: **56 source metadata matches and 18 accepted
approximate variants**. Count/name compatibility does not establish exact author
feature values, class definitions, preprocessing or evaluation folds. Raw files
are excluded from Git and are not automatically available in another checkout.

The [interrupted benchmark record](docs/reproduction/BENCHMARK_EXECUTION.md)
reports 28,919 saved jobs, including 18 complete datasets with 27,610 successful
jobs and 110 recorded failures. The full 113,960-job matrix was not completed.
The historical `data/raw/ugo74_approx/` and `results/stage4r7-full/` directories
are absent from the current cloud checkout; GitHub contains their documentation,
not the raw inputs or detailed job cache. No full-paper reproduction is claimed.
Read [DATASETS.md](DATASETS.md) before preparing or transferring data.

## Antigravity setup on Windows

Codex Cloud and Antigravity have separate files. Synchronize through Git before
opening the project; changing the GitHub repository name does not rename a local
folder. For a fresh clone in PowerShell:

```powershell
git clone --branch sshr-implementation https://github.com/MAYANK-RATHVA05/rb_ugo.git imbalanced-learning-reproductions
Set-Location imbalanced-learning-reproductions
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe main_stage1.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_sshr.py tests/test_sshr_clustering.py tests/test_sshr_resampling.py tests/test_sshr_evaluation.py tests/test_sshr_cli.py tests/test_sshr_small_study.py tests/test_sshr_study_report.py
```

For the existing clone at `C:\ugo\ugo_base`, inspect local changes before
switching branches. Preserve them if Git reports a conflict; do not use reset or
force to make the following commands succeed:

```powershell
Set-Location C:\ugo\ugo_base
git status --short --branch
git fetch origin
git switch sshr-implementation
git pull --ff-only
```

Open this repository root in Antigravity and choose
`.venv\Scripts\python.exe` as the interpreter. Install Python 3.11 if the `py`
command cannot locate it. The core requirements support the LogReg/GNB pilot.
The optional fourteen-classifier setup includes native C5.0 and was validated
on Linux; native Windows C5.0 execution has not been verified. Use the documented
Linux/WSL setup when reproducing the full optional backend environment.

## Cloud / Linux / WSL setup

From the repository root with Python 3.11, Git and a C compiler available:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt -r requirements-classifiers.txt
.venv/bin/python scripts/setup_c50.py
.venv/bin/python main_stage1.py
.venv/bin/python -m pytest -q
```

Dependency versions are pinned. The C5.0 setup verifies a pinned CRAN C50 source
commit and checksum before compiling into ignored `.runtime/c50`. The
[classifier completion record](docs/reproduction/CLASSIFIER_COMPLETION.md)
documents all fourteen interfaces and their deviations from unspecified author
package settings. Installation does not start benchmark training.

## Inspect data and run software checks

The strict binary loader supports CSV with an explicit target and a documented
subset of KEEL `.dat`. Inspection preserves original rows and does not train,
impute, encode or resample:

```bash
.venv/bin/python main_stage2.py --data tests/fixtures/toy.csv --target class
.venv/bin/python main_stage2.py --data tests/fixtures/toy.dat
.venv/bin/python main_stage3.py
.venv/bin/python -m scripts.run_sshr_study --software-check --output results/sshr-software-check
```

The last command requires `sshr-implementation` and runs all six methods on
artificial data. These checks are software validation, not research results.
On Windows, replace `.venv/bin/python` with `.\.venv\Scripts\python.exe`.

To regenerate the fixed SSHR study manually, first place the hash-verified
Ecoli4/Yeast4 files at the paths in [DATASETS.md](DATASETS.md), then run on
`sshr-implementation`:

```bash
python -m scripts.run_sshr_small_study --output results/sshr-small-study --report-dir docs/sshr_pilot
```

Use the selected environment's Python interpreter. The runner never downloads
datasets; it validates inputs before fitting and resumes only matching saved
identities. Run identity changes require a new output directory. The shared
aggregate report alone cannot resume training without the ignored job records.

## APIs and layout

UGO needs a probability classifier for internal uncertainty fitting and accepts
labelled fitting data. Its provisional flag records unresolved source choices:

```python
from sklearn.linear_model import LogisticRegression
from src.ugo import UGO

sampler = UGO(
    LogisticRegression(max_iter=1000, random_state=42),
    variant="smote", cap_interpretation="section_3_3", allow_provisional=True,
)
# X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
```

Fit a fresh downstream classifier on the returned rows; UGO's last internal
uncertainty model may predate its last generated batch. On `sshr-implementation`,
SSHR accepts a separate unlabelled training feature pool:

```python
from src.sshr import SSHR

sampler = SSHR(minority_label=1, random_state=42)
# X_resampled, y_resampled = sampler.fit_resample(X_labeled, y_labeled, X_unlabeled)
```

Both interfaces require meaningful finite numerical features. Neither interface
can identify test rows incorrectly supplied by a caller. The controlled SSHR
runner rejects unsupported categorical/one-hot source inputs and enforces its
training-pool boundaries.

| Path | Purpose |
| --- | --- |
| `src/` | Loaders, samplers, classifier adapters, preprocessing, evaluation and resumable results |
| `scripts/` | Reproducible study runners, report exporters and optional C5.0 setup |
| `docs/` | Paper decisions, implementation records and measured comparisons |
| `data/manifests/` | Portable source hashes and declared dataset preparation choices |
| `data/raw/`, `data/processed/` | Local inputs and derivatives, ignored by Git |
| `results/` | Local detailed experiment records and generated outputs, ignored by Git |
| `tests/` | Software checks and artificial fixtures |
| `main_stage*.py` | Existing inspection, UGO check and reproduction entry points |

## Evaluation rules

Preprocessing, clustering, pseudo-labelling and resampling use fitting data
only. Outer-test features and labels never enter the unlabelled pool. In a
controlled label-hiding study, all methods receive the same reduced labelled
training set; hidden labels are used only for retrospective auditing after
generation. Fold/group definitions, seeds, source hashes, failures and quota
shortfalls are recorded.

Compare against NORES on the same held-out folds. Report standard minority F1,
G-mean, MCC, probability ROC-AUC and false-positive rate with their denominators;
the SSHR paper's printed AUC is not assumed equivalent to probability ROC-AUC.
Repeated folds are dependent, and their sample standard deviations are
descriptive. Distance proxies do not establish true overlap or class membership.
Further experiments or extensions require a separately requested stage under
[AGENTS.md](AGENTS.md).
