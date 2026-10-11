# Stage 4R-4: simulation, dataset and classifier setup

This stage prepares paper experiment inputs. It does not run cross-validation,
resampling comparisons, predictive metrics or significance tests. Source
equations and original method papers are recorded in [SPECIFICATION.md](SPECIFICATION.md)
and [SOURCE_REGISTER.json](SOURCE_REGISTER.json).

## Two declared simulations

`src.simulations` constructs the linear and nonlinear two-feature designs in
UGO §4.1.1 (pp7–8). Both start from seed 42 and independently sampled normal
feature coordinates. Noise relabels a selected number of original **majority**
rows to the minority class. The number relabeled is `0.3` times the **pre-noise
minority** count, rounded by `floor(x+0.5)`. The original labels, observed labels
and selected indices remain separate so checks can verify the direction.

The linear default draws 1000 rows from `N(0,3)` per coordinate and 50 from
`N(6,2)`, interpreting the second parameter as standard deviation. Relabeling
15 majority rows yields 985/65, the counts illustrated in Figure 7(a). The
1000/50 starting counts are inferred from the figure, not specified explicitly
in §4.1.1; the figure does not identify actual random seed or coordinates.
`normal_second_parameter="variance"` is supported for sensitivity checks.
Independence between the two features is also an explicit convention.

The nonlinear design samples 1000 `(X1,X2)` pairs from `N(0,1)` and computes
`Z=sin(X1)^2+cos(X2)+sin(X1)*cos(X2)` and `Y=round(sigmoid(Z))`. Here the
paper says label 1 is majority and label 0 is minority. With seed 42 this gives
942/58 before noise, and relabeling 17 majority rows gives 925/75. Figure
7(b) displays 915/85; the paper does not provide the seed or a way to
reconstruct those exact coordinates. Our output is a declared reconstruction,
not that figure's original data. The twenty-dataset center-distance sweep in
§5.1.1 needs further movement and noise details and is a later experiment task.

## Benchmark registry

The UGO article says there are 74 UCI/KEEL binary variants with imbalance
ratio above five. Their actual names, class definitions and dataset-level
results are in a supplementary package that is still unavailable. The
[candidate registry](../../data/manifests/benchmark_candidates.json) records
eight historically inspected KEEL-format files from commit
`0d78c2d9db22f92cee912c94671705d4cd5ef8b9`. It records original `.dat`
and `.zip` hashes, counts, feature counts and known categorical columns.
**None of these eight is currently claimed as one of the paper's 74.** Their
provider URLs and licenses were not independently authenticated in that
historical audit, and no original fold pairs were supplied.

`src.dataset_registry` checks that a placed file under `data/raw/` has the
historical `.dat` hash before loading it, then compares rows, labels and feature
types. A successful match verifies historical bytes only; it does not confirm
paper membership or author folds. All eight files are absent from the current
workspace. Raw files remain ignored by Git. Missing candidates do not become
stand-in paper datasets or successful benchmark results.

## Fourteen classifier identities

`src.classifier_registry` records every classifier and group in UGO Tables 7–8,
including paper settings and current implementation status. Nine methods have
local sklearn constructors for **training-only probability smoke checks**:
CART, GNB, KNN, LDA, LogReg, QDA, RF, SVMLinear and SVMRadial. All are marked
`python_approximation`. This matters for CART's `cp` versus sklearn
`ccp_alpha`, SVM radial `sigma` versus `gamma`, default penalties, calibration
and unknown author package versions. These models must not be reported as
verified paper configurations.

BART, BNNET, C5.0, GKNB and NNET have explicit `unavailable` status. The stated
dropout/RMSprop architecture, Bayesian posterior sampling, C5.0 behavior and
unbiased-CV kernel bandwidth cannot be replaced by similarly named sklearn
models without changing the method. These five names fail clearly if requested.
The paper's 100-posterior-sample averaging for Bayesian methods is not yet
available. Within-class zero-variance filtering, train-only imputation/encoding
and scaling belong to the preprocessing runner in Stage 4R-5.

`smoke_classifier_on_fitting_data` clones each supported constructor, fits it
on supplied fitting rows and checks two-class finite normalized probabilities
on those same rows. It accepts no test inputs and computes no performance
metrics. The exercise establishes software compatibility only.

## Run and verify

From the repository root:

    .venv/bin/python main_stage4r4.py
    .venv/bin/python main_stage4r4.py --smoke
    .venv/bin/python -m pytest -q

The command prints counts and source status, with no sample records or scores.
On Python 3.11, `pytest -q` passed **268** checks. Both command forms ran; the
smoke form reported nine supported probability interfaces without test rows or
predictive-performance values. The eight-entry registry was independently
compared field by field with the historical committed manifest; hashes, labels,
counts and feature types matched. The linear default reports 985/65 observed
labels, and the nonlinear seed-42 default reports 925/75.

Use the actual UGO supplement to confirm the 74 variants and compare published
data; obtain exact classifier implementations/configuration where possible.
Stage 4R-5 has implemented the [evaluation engine](EVALUATION.md), verified on
artificial software fixtures. The measured simulation/dataset pilot belongs to
Stage 4R-6; unresolved methods remain excluded from exact-result claims.
