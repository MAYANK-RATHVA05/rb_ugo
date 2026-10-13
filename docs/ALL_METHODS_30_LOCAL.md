# Run the fixed 30-dataset comparison on Windows

This descriptive study extends the Ecoli4/Yeast4 controlled pilot. Its fixed
selection is [all_methods_30.json](../data/manifests/all_methods_30.json). It
compares NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE, EC-EBO and SSHR with LogReg and
GNB. UGO-ROS/RWO/MCSMOTE are not part of this six-condition table.

The 30 inputs are related binary variants, not 30 independent original data
sources. Ecoli4 and Yeast4 use the same source bytes as the earlier pilot. The
selection replaces Vowel0 with `ecoli_0_1_4_6_vs_5` so all 30 inputs use the
same coordinate-group split and label-mask procedure. Satellite class 4 and
Abalone rings >=18 are accepted approximate variants. Exact author data,
preprocessing, folds and SSHR's unlabelled pool are unverified.

## Prepare Antigravity PowerShell

From the repository root (adjust the drive/path if needed):

```powershell
cd C:\ugo\ugo_base
git status --short
git fetch origin
git switch --track -c all-methods-30-study origin/all-methods-30-study
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

If the local branch already exists, use `git switch all-methods-30-study` and
`git pull --ff-only origin all-methods-30-study` instead of the `git switch -c`
line. Preserve or commit any local changes reported by `git status --short`
before switching branches. Python 3.11 is the tested target; the existing
`.venv` may be reused. On a fresh clone, create it with `py -3.11 -m venv .venv`.

The raw data under `data/raw/ugo74_approx/` are deliberately ignored by Git.
A Git pull alone does not provide them. To rebuild the selected inputs locally,
put the four KEEL section ZIPs (`imb_IRlowerThan9.zip` and
`imb_IRhigherThan9p1.zip` through `imb_IRhigherThan9p3.zip`) and the UCI
`statlog+landsat+satellite.zip` in your Downloads folder. These are the
original archives collected for the UGO baseline. The preparation command
finds renamed copies such as `(1)` by their registered SHA-256. The banana and
abalone R objects are fetched from pinned public CRAN GitHub revisions unless
matching local copies are already in Downloads. Install the R-data reader and
run:

```powershell
.\.venv\Scripts\python.exe -m pip install pyreadr==0.5.7
.\.venv\Scripts\python.exe -m scripts.prepare_all_methods_30_inputs --source-dir "$env:USERPROFILE\Downloads"
```

Use another folder with `--source-dir` if the five original ZIPs are elsewhere.
The command verifies both source hashes and prepared-input hashes, leaves
existing mismatched files untouched, and never trains models. It writes only
under the ignored `data/raw/` folder. The target paths and SHA-256 values are
in [ugo_74_approximate.json](../data/manifests/ugo_74_approximate.json).

## Check sources without training

```powershell
.\.venv\Scripts\python.exe -m scripts.run_all_methods_30 --check-only
```

This verifies all 30 file hashes, numerical columns, class counts, duplicate
coordinate groups, 25 test-fold events and labelled/unlabelled masks. It prints
the first missing or inconsistent file and fits no model. The Ecoli4/Yeast4
source hashes were checked against the previous pilot registry. Winequality
red 8-vs-6 has as few as one minority example in a test fold, so its fold
scores are especially unstable.

## Pilot, then complete the study

Start with one small input. Its saved jobs can be reused by the full command:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_all_methods_30 --datasets ecoli4 --output results\all-methods-30
```

After inspecting runtime, failures and disk usage, run the full fixed study:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_all_methods_30 --output results\all-methods-30
```

The main comparison has 30 × 2 classifiers × 25 folds × 6 methods = **9,000
jobs**. The SSHR empty-U ablation adds 30 × 2 × 25 × 2 = **3,000 jobs**.
The driver processes datasets sequentially. A completed job is saved atomically;
rerunning the same command with the same code, data and dependencies reuses it.
After a stopped or failed invocation, inspect the reported error before
rerunning. Do not edit the software or source files within one result identity.
Failures remain visible in coverage and are not silently dropped or retried.

Every method receives the same reduced labelled fitting set `L` and the same
held-out folds. SSHR alone receives the training-only unlabelled feature pool
`U`. The gold labels of `U` are used only to construct the fixed experimental
mask and to audit results after fitting. Preprocessing, resampling and model
selection never use outer-test rows. The extra empty-U comparison isolates the
effect of supplying `U` under this declared protocol.

## Read the results

- `results/all-methods-30/all_methods_30_scores.csv`: combined canonical
  same-fold F1, G-mean, MCC, probability ROC-AUC and FPR scores. At complete
  coverage it has 1,800 data rows (30 × 2 × 6 × 5). Open it in Excel or use
  `Import-Csv` in PowerShell.
- `results/all-methods-30/<dataset>/reports/scores.csv`: per-dataset means,
  sample SD and shared-fold counts.
- `results/all-methods-30/<dataset>/reports/coverage.csv`: successes, failures
  and missing jobs. Also inspect `diagnostics.csv`, `ablation.csv` and
  `summary.json` for sample counts and SSHR with-U/without-U results.
- `results/all-methods-30/<dataset>/{with-u,without-u}/`: detailed local job
  records and run manifests. These and all raw inputs are ignored by Git.

The combined CSV contains only `all_methods_shared_success` rows. Empty means
or fewer folds must remain explicit; never rank a method on a different fold
subset without stating it. Use separate LogReg and GNB summaries, report all
five metrics plus coverage, and note correlated Ecoli/Yeast variants. The
earlier UGO-only benchmark used different folds and cannot be inserted into
this table.

## Package results for review

After all 30 dataset reports are written, create one small ZIP to upload for
analysis. It contains the combined table, `overview.json`, and each dataset's
six aggregate report files. Raw data, saved fold jobs and predictions stay on
your PC. The command checks report completeness and records any reduced or
failed fold coverage in `review_manifest.json` inside the ZIP:

```powershell
.\.venv\Scripts\python.exe -m scripts.package_all_methods_30_reports
```

Upload `results/all-methods-30-review.zip`. The packaging command does not
change the experiment or push results to GitHub.
