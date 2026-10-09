# Risk-Budgeted Uncertainty-Guided Oversampling (RB-UGO)

## Scope: Stage 1

This portable Python 3.11 research repository currently provides configuration,
directory validation, and tests only. No datasets, oversampling algorithms,
training, or experiments are implemented. Stage 2 requires explicit approval.

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
│   ├── data_loader.py      # Documented placeholder
│   └── imbalance_stats.py  # Documented placeholder
├── notebooks/.gitkeep      # Future exploratory notebooks
├── results/.gitkeep        # Future generated research outputs
├── tests/test_stage1.py    # Configuration and initialization tests
├── main_stage1.py          # Directory checks and environment display
├── requirements.txt       # Pinned research dependencies
├── requirements-dev.txt   # Runtime dependencies plus pytest
├── README.md
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
