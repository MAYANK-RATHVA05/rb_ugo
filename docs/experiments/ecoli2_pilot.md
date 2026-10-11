# First real-data UGO training pilot: Ecoli2

## Scope and provenance

The working pipeline was trained and evaluated on the user-supplied binary
KEEL `ecoli2.dat` file on **2026-10-11**. This is a single fixed holdout pilot,
not a reproduction of every table or figure in the UGO paper. No test-guided
parameter, threshold, seed or method selection was performed.

- Implementation commit: `a91a369458a73fb05dae5cd766798291398f1d5f`.
- Source worktree at training time: clean.
- Source SHA256: `0fd2a70626e5361c9676ce80088c501c865320195396434e175e1f0c555d044a`.
- Source: original `.dat` member extracted unchanged from the uploaded
  `ecoli2.zip`; parsed with the existing KEEL loader.
- Dataset: 336 observations, seven raw numerical features, labels
  `negative -> 0` (284) and `positive -> 1` (52).
- Exact membership/version in the paper's 74-benchmark suite remains unverified
  until its supplementary dataset manifest is obtained.

Source method: Sağlam and Cengiz, *Knowledge-Based Systems* 342 (2026), 115886,
[doi:10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886).
The [core audit](../ugo_reproduction.md) records verified equations and execution
ambiguities. Empirical values below come from this project's saved run artifacts,
not from the paper's published results.

## Fixed protocol

Seed 42; one stratified 75/25 split shared by all methods. Training contains
252 rows (213 majority, 39 minority); untouched evaluation contains 84 rows
(71 majority, 13 minority). Train/test row positions are disjoint and cover the
source file. This dataset has no duplicate complete feature-plus-target rows.

Training-only feature filtering removed `Lip` and `Chg`, which were constant
within the minority training class. The remaining five features were `Mcg`,
`Gvh`, `Aac`, `Alm1`, and `Alm2`. Median imputation and standard scaling were
fitted once using training data and shared by all methods. The paper states
within-class zero-variance removal (Section 4.4); the remaining preprocessing
choices are declared pipeline conventions.

All three final classifiers are independent
`LogisticRegression(max_iter=1000, random_state=42)` fits. Each UGO method also
uses this estimator configuration internally. Predictions use the unchanged
default threshold, with no threshold tuning. UGO receives training observations
only; each final classifier is fitted after generation finishes.

UGO settings: `lambda_select=0.5`, `lambda_step=0.1`, `phi=2.5`, `k_pre=5`,
`k_after=1`, `k_smote=5`, `alpha=0.05`, `max_iter=100`,
`cap_policy=section_3_3`. The fixed batch is `ceil(0.1*252)=26`; the selected
generated-row cap is 813. The Section 3.3/Table 4 conflict and remaining
pool/rounding conventions remain unresolved and explicit in diagnostics.

## Observed holdout results

| Method | Added training rows | IAM | MCC | F1 | Minority recall | Minority precision | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NORES | 0 | 0.7281 | 0.8096 | 0.8333 | 0.7692 | 0.9091 | 0.8920 |
| UGO-ROS | 78 | 0.5822 | 0.7391 | 0.7742 | 0.9231 | 0.6667 | 0.8958 |
| UGO-SMOTE | 78 | 0.5822 | 0.7391 | 0.7742 | 0.9231 | 0.6667 | 0.8958 |

IAM uses the actual equation in the UGO paper's Section 4.3, p.9. Average
precision is sklearn's non-interpolated `average_precision_score`; it is an
additional pilot metric. Balanced accuracy was 0.8776 for NORES and 0.9193 for
both UGO variants. Metric conventions are recorded in `run.json`.

Confusion matrices use true labels on rows and predictions on columns, with
label order `[0,1]`:

```text
NORES       [[70, 1], [3, 10]]
UGO-ROS     [[65, 6], [1, 12]]
UGO-SMOTE   [[65, 6], [1, 12]]
```

Each UGO variant generated three batches of 26 minority observations, reaching
330 training rows (213 majority, 117 minority). Both pre-detected 16 noisy
original training observations and relocated zero generated points. Four
scoring iterations include the final stopping check: Welch p-values were
approximately 0.2124 (ROS) and 0.1371 (SMOTE), leading to non-rejection at 0.05.
Neither reached its generation cap. Non-rejection is not proof that the class
uncertainty means are equal.

Both UGO variants produced the same hard predictions on this test split, with
different probability values. Their matching classification metrics do not mean
their generated data or fitted models are identical.

## Interpretation

The pipeline trained successfully. UGO recovered two additional minority test
observations while introducing five additional majority false positives.
Minority recall increased, while precision, F1, MCC and IAM decreased. These
results demonstrate a working training/evaluation pipeline; they do not establish
general superiority or validate all published claims. No follow-up configuration
was selected using these scores, and no significance test was performed on this
single split.

Only **Ecoli2** was trained in this milestone. The other seven supplied datasets
were extracted and parsed but were not used for model-performance experiments.
Several contain duplicate rows; broader evaluation must explicitly consider
duplicate/group handling and the authors' original folds before claiming
independent benchmark evidence.

## Software validation and artifacts

**188 software tests passed**. Tests cover train-only preprocessing, resampler
input isolation, the shared split, final fitting on every augmented row, known
IAM values, categorical inference, deterministic generation, CLI output handling
and model persistence.

Every actual Ecoli2 saved model was reloaded and checked against its recorded
test predictions and probabilities. The data checksum, split coverage and clean
source revision were also verified. Linux/Python 3.11.16 was executed; Windows
commands are documented but a Windows runtime was not tested here.

Recorded runtime packages: NumPy 1.26.4, pandas 2.2.3, SciPy 1.14.1,
scikit-learn 1.5.2, imbalanced-learn 0.12.4 and joblib 1.6.0.

Local artifacts are under `results/ugo_training/ecoli2-seed42-section3_3/`:
three fitted `.joblib` pipelines, `run.json`, `split.json`, `metrics.csv`,
`predictions.csv` and both UGO diagnostic reports. Raw data and generated model/
result files remain ignored by Git. This aggregate documentation is committed.

The recorded command was:

```bash
.venv/bin/python main_train.py --data data/raw/ecoli2.dat --seed 42 --test-size 0.25 --cap-policy section_3_3 --output-dir results/ugo_training/ecoli2-seed42-section3_3
```

To reproduce locally, use the [training guide](../ugo_training.md) and a new
output directory, or omit `--output-dir` to obtain a unique directory. The CLI
never overwrites an existing run.
