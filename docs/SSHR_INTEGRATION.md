# SSHR estimator and controlled runner — Stage 3

Stage 3 provides [src/sshr.py](../src/sshr.py), a complete resampling estimator,
and an opt-in [controlled evaluation runner](../src/sshr_evaluation.py).
The command-line entry point is [scripts/run_sshr_study.py](../scripts/run_sshr_study.py).
Existing UGO, BFCH-SMOTE, EC-EBO, comparator registries, evaluation defaults,
preprocessing and probability-AUC implementation are unchanged.

This document retains Stage 3's API and historical software validation. The
subsequently authorized [Stage 4 study](SSHR_SMALL_STUDY.md) is complete;
its real-data outcomes are recorded separately from the artificial checks.

**SSHR selects EXISTING unlabelled observations with minority pseudo-labels.
Only its within-cluster SMOTE fallback generates synthetic feature vectors.**
The output retains separate counts and provenance for these operations; neither
cluster assignment nor confidence establishes the true class of an unlabelled
observation.

The source is Jiang et al. (2023), *A semi-supervised resampling method for
class-imbalanced learning*, [DOI:10.1016/j.eswa.2023.119733](https://doi.org/10.1016/j.eswa.2023.119733),
Algorithms 1–2, Eqs.1–8 and 12, and Sections 3–4.
The full-paper [method decisions](SSHR_METHOD_DECISIONS.md),
[SSHC record](SSHC_IMPLEMENTATION.md) and [resampling record](SSHR_RESAMPLING.md)
preserve verified rules and declared interpretations. **Author-code behavior
and the paper's unlabelled-pool construction remain UNVERIFIED.** This software
does not establish exact author-code equivalence or reproduced paper results.

## Estimator API

```python
from src.sshr import SSHR

sampler = SSHR(minority_label=1, random_state=42)
X_resampled, y_resampled = sampler.fit_resample(
    X_labeled, y_labeled, X_unlabeled,
)
```

Every call creates a fresh SSHC model from exactly the supplied pools, then
executes the audited Stage 2 core. The estimator fits no preprocessing or
downstream classifier. Its three-input interface needs an explicit call;
ordinary two-input imbalanced-learn pipelines do not automatically provide a
separate unlabelled pool. `X_unlabeled=None` or an empty correctly shaped array
supports an ablation. Inputs must be finite, real numerical fitting matrices
with two observed classes, following the earlier validation restrictions.

Parameters expose all earlier choices: source minority identity, `mu`, printed
versus conventional objective, pruning scope, centroid tolerance and iteration
guards, `delta`, `removal_fraction`, neighbour count, seed and SMOTE pool policy.
An explicit minority identity remains fixed even if reduced observed counts tie
or invert. The printed objective remains the default; it is not silently changed
to standard F1. Parameters support sklearn cloning and inspection.

Successful fitted attributes include `clustering_`, `result_`, `classes_`,
`minority_label_`, `majority_label_`, `n_features_in_`,
`retained_labeled_indices_`, `selected_unlabeled_indices_`,
`synthetic_samples_`, `row_lineage_` and `diagnostics_`.
Returned arrays and exposed records are copied independently. Every refit clears
previous fitted state. A diagnosed cleaning failure re-raises
`SSHRResamplingError` and retains only fresh `diagnostics_`; other invalid fits
leave no previous learned geometry or generation state.

The standalone caller must establish original feature semantics and training/test
separation. The estimator cannot determine whether an arbitrary row came from
an outer test set. The controlled runner below makes the pool boundary explicit.
There is no hidden-label input or pseudo-label accuracy attribute in the sampler.

## Controlled runner

```python
from src.sshr_evaluation import ControlledEvaluationConfig, evaluate_controlled

configuration = ControlledEvaluationConfig(
    classifiers=("LogReg",),
    methods=("nores", "smote", "ugo_smote", "bfch_smote", "ec_ebo", "sshr"),
    n_splits=5,
    n_repeats=5,
    labeled_fraction=0.5,
    min_labeled_per_class=2,
    use_unlabeled=True,
    seed=42,
)
result = evaluate_controlled(
    X, y,
    dataset_id="explicit_source_identity",
    positive_label=1,
    provenance={
        "kind": "caller-documented numeric source",
        "feature_types": {column: "continuous" for column in X.columns},
        # Also preserve original source hashes and preparation choices here.
    },
    output_dir="results/sshr-controlled-run",
    config=configuration,
)
```

This example describes the later study interface; **no real-data study was run
in Stage 3**. The new configuration defaults to NORES and SSHR, not all existing
methods. Calling this separate runner opts into the controlled protocol. Its
supported methods are the six names above, and NORES is mandatory. Existing
`EvaluationConfig`, `METHODS`, `AVAILABLE_METHODS`, `evaluate_ten_fold` and
comparison scripts retain their previous behavior.

Five folds and five repeats match the paper's reported counts (Section 4.3,
p.7), but the row assignments, random seeds, label mask and preprocessing are
our declared protocol. They are not recovered author folds. Classifier specs
and actual parameters are saved; repository classifier defaults must not be
presented as the paper's RBF SVM settings.

Before fitting, the runner copies configuration/provenance, freezes all outer
folds and label masks, validates them and writes the run manifest. For each
repeat, every row occurs in exactly one test fold. Each training fold is the
exact complement of its test fold, with both fixed classes in each partition.
The seed derivation is `SeedSequence([root_seed, repeat, fold, purpose])`, with
purpose 1 for outer splits, 2 for masks and 3 for classifier/resampler jobs.
Methods use the same job seed and classifier configuration within a fold.

## Label hiding and leakage boundary

This is a controlled study of fully labelled files. For a training class with
`n_c` observations, default masking retains:

```text
keep_c = min(n_c, max(min_labeled_per_class, floor(labeled_fraction * n_c)))
```

A class with fewer rows than the required minimum fails before evaluation;
folds are not silently reduced. Sampling within each training class is seeded
and without replacement. The manifest records exact global labelled/unlabelled
indices, requested fraction, actual fraction, per-class counts and any minimum
clamp. The ordered `L` and `U` pools partition the outer training rows only.
No mask is redrawn to rescue a method-specific failure.

Stratification uses known training labels to construct this simulation design
**before** hiding them. That is an experiment-design choice, not a way to
stratify genuinely unlabelled observations whose classes are unknown. Once the
mask is fixed, withheld labels cannot influence learned operations.

| Operation | Features supplied | Labels supplied |
| --- | --- | --- |
| Preprocessing fit, including imputation, feature filtering and scaling | Original `X_L` | Observed `y_L` only |
| Baseline resampling and any internal UGO classifier | Transformed `X_L` | Observed `y_L` only |
| SSHC and SSHR | Transformed `X_L` plus training-only `X_U` | Observed `y_L` only |
| Fresh final classifier fit | The method's returned training rows | Observed or assigned training labels |
| Outer-test transformation/prediction | Test features transformed using the fitted `L` preprocessor | No labels supplied to transformation/prediction |
| Metrics and retrospective pseudo-label audit | Saved predictions/selected source indices | Evaluator-only test truth or withheld training truth |

NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE and EC-EBO receive the **identical reduced
labelled set**, not all outer training labels. A fresh clone of the same
classifier template is fit on each method's output; the runner does not score
UGO's internal classifier. Held-out features are transformed only after
resampling and final fitting. An observation may serve different roles across
CV folds, but within each fold `L`, `U` and the outer test set are disjoint.

`use_unlabeled=False` preserves the exact folds, masks and `L`, while supplying
an empty algorithmic `U` to SSHR. This tests the value of access to the withheld
feature pool without changing the baseline label budget. `labeled_fraction=1`
instead changes the label budget and naturally produces empty `U`; it is a
different control and must be reported separately.

The evaluator computes pseudo-label accuracy only after resampling finishes,
using selected pool positions to retrieve withheld training truth. It records
correct/incorrect counts and global source indices. Empty selection gives
`accuracy=None`, not perfect accuracy. This diagnostic never selects parameters,
changes quotas, rescues failures or supplies labels to a sampler/final model.
In a genuinely unlabelled application, accuracy is unavailable until labels are
obtained independently.

## Source restrictions, supplied partitions and resume

`provenance.feature_types` must identify every original feature column as
`numeric` or `continuous`. Missing, extra or categorical/boolean/one-hot source
declarations fail before results are written. Actual categorical, object,
boolean and complex feature dtypes also fail. Metadata is a caller declaration,
not proof of continuous semantics; one-hot coordinates must not be falsely
declared continuous. Numerical missing values use the existing `L`-only imputer;
infinite values are rejected.

The default path rejects duplicate feature-coordinate rows. Stage 4 adds an
explicit opt-in: provenance may supply integer `coordinate_groups`, one per
raw input row, together with compliant supplied folds and masks plus their
authorities. Identical raw feature vectors must have the same group, and a
group cannot cross outer train/test or observed L/hidden U boundaries. All rows
remain intact; no silent deduplication occurs. See the documented Stage 4
preparation recipe. Other entity/time constraints still require caller-supplied
compliant partitions and cannot be inferred from these files.

Supplied partitions are lists of records:

```python
supplied_folds = [{"repeat": 0, "fold": 0,
                   "train_indices": [...], "test_indices": [...]}, ...]
supplied_masks = [{"repeat": 0, "fold": 0,
                   "labeled_indices": [...], "unlabeled_indices": [...]}, ...]
```

All indices are global input-row positions, not DataFrame index labels.
There must be one fold/mask record for every configured repeat/fold, with unique
identifiers and complete, disjoint partitions. Supplied masks need both observed
classes at the configured minimum. Supplied folds require `fold_source` and
supplied masks require `mask_source` provenance. A supplied mask has no asserted
executed mask seed: `mask_seed=None`.

Atomic `ResultStore` manifests and records preserve input hashes, source
provenance, folds/masks and their hashes, configuration/policies, classifier
specifications, source/package identities and the CLI script hash. A changed
identity requires a new output directory. Provenance must be finite JSON data;
invalid NaN/infinite claims are rejected rather than normalized to a legitimate
null value in the identity. Successful/unavailable jobs are reused;
failed jobs remain terminal unless `retry_failures=True`. Callbacks receive saved
records and cannot change the copied configuration through the caller's original
mutable parameter dictionaries. Coverage lists failures and the exact successful
repeat/fold pairs shared with NORES. Later summaries must use explicitly shared
folds and retain coverage, not silently compare different success subsets.

## Metrics, lineage and executable software check

Held-out records contain standard minority F1, G-mean, MCC, confusion counts,
false-positive rate `FP/(TN+FP)` and **probability ROC-AUC** from the fixed
positive class's probabilities. `probability_auc` explicitly aliases the existing
`auc` field. These metrics are not an assertion that the paper's printed AUC is
probability ROC-AUC; its source conflicts remain in the audit.

SSHR records retain separate removed-majority/minority, selected-existing and
SMOTE-generated counts, shortfalls, cluster diagnostics and row lineage. The
runner adds global source positions to retained/pseudo rows and to SMOTE
anchor/neighbour positions while preserving their algorithm-relative indices.
For SSHR additions it records a training-only proxy: distance to the nearest
original observed majority row is strictly less than distance to the nearest
original observed minority row. Ties are separate. The reference remains the
original reduced `L` before method-specific deletion; hidden truth and test rows
are absent. This is a local geometric proxy, not irreducible class overlap.
Stage 4 adds equivalent baseline exports, with generated minority/majority
counts separated, an append-only observed-prefix check, and integrity-checked
paired descriptive summaries. Existing supervised algorithms are unchanged.

From the repository root, run the artificial check with:

```bash
.venv/bin/python -m scripts.run_sshr_study --software-check --output results/sshr-software-check
```

Repeat the exact command to verify resume. On Windows use
`.venv\Scripts\python.exe -m scripts.run_sshr_study` with the same options.
The fixture creates 120 unique two-feature Gaussian observations, uses one
repeat/two folds and all six methods with LogReg, and records explicit small
UGO/SMOTE software-check parameters. It is a functional test, not a paper
simulation, dataset benchmark or method-superiority experiment.

The CLI also prepares a single local-file path for the later authorized study:
`--data` requires the expected 64-character `--sha256`; a mismatch fails before
loading/training/output. The hash is checked again after loading. CSV additionally
needs `--target` and `--numeric-source` to declare original numerical semantics;
KEEL uses original attribute declarations and still rejects nominal inputs.
The script never downloads data or automatically runs a 44/74-dataset benchmark.

## Validation and remaining work

Artificial tests exercise the actual estimator and all six runner constructors,
training-only preprocessing/sampler spies, fresh final fitting, hidden-label and
test-data perturbations under frozen masks, identical-label-budget ablations,
global lineage, metric orientation, diagnosed/unavailable failures,
configuration/mask/fold identity, strict resume and interruption recovery,
categorical/source restrictions, source hashes and the CLI check.

| Validation in the cloud checkout | Result |
| --- | --- |
| Estimator tests | 38 passed |
| Controlled runner tests, including final provenance/lineage checks | 56 passed |
| CLI tests | 11 passed |
| Combined new Stage 3 tests: `.venv/bin/python -m pytest -q tests/test_sshr.py tests/test_sshr_evaluation.py tests/test_sshr_cli.py` | 105 passed in 3.05 s |
| Full suite: `.venv/bin/python -m pytest -q` | 650 passed in 16.25 s; no failures or skips |
| Artificial all-six-method CLI check, LogReg, two folds | 12/12 jobs succeeded; no failed/unavailable jobs |
| Exact repeat of that CLI invocation | 12/12 jobs reused, with the same run ID and records |

The final CLI check used `/tmp/sshr-stage3-final-software` and run ID
`bea4a62b8c1537a8e3b8a9f4a8690eecfdc2d32ba563400fd07535698673fe3e`.
Generated records stay outside Git. The check executes fresh final classifiers
on artificial data, and the tests inspect preprocessing/sampling boundaries and
frozen-mask perturbation behavior; these checks are not real-data performance
evidence. No new dependency or existing algorithm/test-file change was required.

Stage 3 stopped after software validation, documentation, commit and push.
The user's subsequent **“next”** authorized Stage 4: the hash-verified Ecoli4/
Yeast4 study with paired classifiers/folds/label budgets, baseline proxies and
the unlabelled-data ablation. The [measured report](SSHR_SMALL_STUDY.md) preserves
coverage, diagnostics and limitations; these outcomes do not alter the historical
Stage 3 checks above or establish paper-result fidelity, superiority or novelty.
**Stop after Stage 4 and wait for the user's review.**
