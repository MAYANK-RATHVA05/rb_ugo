# Stage 4R-5: paired evaluation and resumable results

The evaluation software is implemented. Verification uses artificial fixtures;
no paper simulation pilot or benchmark result is claimed at this stage. The
source remains [UGO §§4.3–4.4](SPECIFICATION.md), with the available classifier
implementations and source gaps described in [EXPERIMENT_SETUP.md](EXPERIMENT_SETUP.md).

## Training and test boundaries

`src.evaluation.evaluate_ten_fold` accepts a dataset identity, provenance, an
explicit source/domain positive label, a fixed configuration and a dedicated
output directory. NORES is mandatory. Eleven conditions are supported: NORES,
RUS, ROS, SMOTE, RWO, MCSMOTE, LEAO and the four UGO variants. RUS is included
from §4.2 although it is absent from the paper's main result tables.

All methods/classifiers use the same ten recorded partitions. With no supplied
folds, `StratifiedKFold(10, shuffle=True, random_state=42)` is a **declared
approximate protocol**, not the author folds. Each class must have at least ten
rows. Supplied folds need `fold_source` provenance, exactly ten complementary
train/test pairs, both labels in each partition, and exactly one test appearance
per input row. Grouped/time-dependent sources must supply compliant partitions;
set `requires_group_or_time_split=True` to prevent ordinary stratification.
The caller must verify provider constraints in those supplied partitions; the
runner does not infer groups or timestamps. Infeasible inputs fail explicitly.

Each job fits a fresh `FoldPreprocessor` on its training rows and labels. It
transforms training and test features with that saved state, resamples only the
transformed training rows, and fits a **fresh cloned final classifier** on the
returned resampled rows. It never uses `UGO.estimator_` for final predictions,
because that internal uncertainty model may predate the last generated batch.
Final prediction/probability shape, class order, finiteness and normalization
are checked before scores are saved. There is no test-driven tuning or method
substitution. SVC's existing probability calibration uses its training input.

The split seed is the configuration seed; job seed is `(seed + fold) % 2**32`,
identical across methods/classifiers in that fold. Configurations are fixed
before running. No Bayesian calibration/posterior averaging is implemented for
the five unavailable classifiers.

## Declared preprocessing choices

The paper says to remove features with zero variance within any class but does
not fully identify transformation order, missing-value handling, scaling or
category representation. Our fixed convention is:

1. Numeric training medians impute missing values; a completely missing training
   column is filled with zero. Categorical values use string representation;
   missing values use the reserved `__ugo_missing__` token.
2. Fit dense one-hot encoding on training categories. Unknown test categories
   become all-zero encoded values. Numeric columns precede encoded columns.
3. Remove any imputed/encoded feature constant within **either** training class,
   checking exact constancy before scaling. If no features remain, fail the job.
4. Fit StandardScaler on the retained training columns, including one-hot
   features. `scale=False` disables this step and changes the run identity.

Training labels influence only the stated variance filter; test labels never
influence features. Class-exclusive categorical indicators can be removed under
this literal filter. Oversamplers operate on continuous encoded coordinates;
this does not implement categorical RWO roulette or category-aware SMOTE.
These choices are approximations to unspecified author preprocessing. Column
names/order must match; infinities and complex numerical features fail. Input
frames and labels are copied rather than modified.

## Metrics

`src.metrics` uses true rows and predicted columns with label order
`[negative, positive]`, i.e. `[[TN, FP], [FN, TP]]`. Table 6's swapped FP/FN
labels are not propagated. F1 uses the caller's fixed positive outcome even
when it is label 0 or the majority class. The CLI requires the original text
label instead of inferring that outcome from test frequencies.

IAM is the mean of `(diagonal - max(off-diagonal row sum, off-diagonal column
sum)) / max(row total, column total)` from §4.3 p9. A class absent from both
truth and predictions makes IAM undefined (`null`). MCC and F1 with zero
denominators return zero and set an explicit undefined flag. These are declared
engineering policies. Valid ten-fold partitions have both true classes, so IAM
has nonzero denominators even for single-class predictions. Independent count
examples, perfect/inverted classifications, class permutations and sklearn
MCC/F1 comparisons verify the formulas.

## Result schema and resume

The directory contains `manifest.json` and one atomically replaced JSON file per
`(fold, classifier, method)` job. This is a sequential, single-writer store.
Interrupted unfinished jobs have no terminal record and run on resume; already
written jobs remain reusable. The manifest identifies:

- Dataset feature/label hash, identity, provenance, explicit positive outcome
  and exact fold indices/hash.
- Fixed method configurations, preprocessing/metric policies, source classifier
  settings and approximation/unavailability status.
- Python/dependency versions and a SHA-256 fingerprint of all `src/*.py` and
  `requirements.txt` bytes. Changed algorithm code invalidates reuse.

Successful records include truth, predictions, probabilities and column label
order; IAM/MCC/F1 and confusion counts; training/test/resampled class counts;
added/removed row counts; fitted preprocessing statistics/categories and
retained/dropped feature names;
resolved classifier/resampler parameters; UGO/generator diagnostics; warnings,
seeds and phase runtimes. Array/nonfinite diagnostic values become JSON lists
or explicit nulls; undefined scores carry their metric flags.

Failed jobs have `metrics`, `predictions` and `probabilities` set to `null` and
record the error stage/type/message. Unavailable classifiers have a separate
`unavailable` status. Neither becomes a successful result. Terminal failures
are reused by default; `retry_failures=True` retries failed jobs with the same
identity. Successful and unavailable jobs stay cached. Code/config/data changes
require a new directory. Hash verification rejects corrupted records rather
than trusting or silently replacing them.

Returned coverage reports success/failure/unavailability counts and the exact
successful fold intersection with NORES. There is no silent averaging across
different paired coverage and no significance analysis at this stage. Actual
performance aggregation and dataset-based statistical comparisons belong to
the later experiment/report stages.

## Commands and validation

From the repository root, run the artificial integration fixture and repeat the
same command to verify resume:

    .venv/bin/python main_stage4r5.py --software-check --output results/stage4r5-software-check

This verifies 20 jobs (LogReg × NORES/ROS × ten folds), not paper performance.
For a local input with a known positive source label:

    .venv/bin/python main_stage4r5.py --dataset data/raw/example.csv --target outcome --positive-label case --output results/example-run --classifiers LogReg --methods nores ros smote ugo_ros ugo_smote

`example.csv` is a placeholder, not an available or paper-confirmed dataset.
`--method-parameters` reads a JSON mapping of selected methods to fixed
parameters; `--folds` reads ten train/test index objects and requires
`--fold-source`. All requested jobs finish despite individual failures. The CLI
returns exit code 1 if any job failed or was unavailable, preserving the records.

Tests exercise all eleven real samplers through the runner on a small artificial
fixture; spy checks prove training-only preprocessing/resampling and fresh final
fitting on added rows. Other checks cover interruption/resume, identity changes,
corruption, infeasible folds, failure/unavailability, invalid probabilities,
categorical/missing CSV input and an explicitly majority-positive outcome.

On Python 3.11, the full suite passed **313 tests** with no failures or skips.
The documented software command completed all 20 jobs; repeating it reused
all 20 saved records. The CLI tests also verify nonzero exit status and ten
empty-score records for an unavailable classifier.

The supplement's 74 dataset identities/folds and five classifier implementations
remain unavailable. The subsequently authorized **4R-6** simulation pilot is
documented in [SIMULATION_PILOT.md](SIMULATION_PILOT.md), including cap stops
and the unresolved paper F1 averaging discrepancy.
Stages **4R-7** and **4R-8** cover the full declared matrix/additional analyses
and final tables/figures/report. This stage completes the evaluation engine,
not the whole paper reproduction.
