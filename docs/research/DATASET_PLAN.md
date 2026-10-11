# Dataset, simulation and evaluation plan

Updated after Stage 4A-R verification, 2026-10-11. Eight supplied **full files**
are imported and parsed; no official folds are supplied, no controlled simulation
has been executed and no predictive benchmark result exists. The proposed study
below remains unimplemented. See [actual verification](DATASET_VERIFICATION.md),
[reference evidence](REFERENCE_STATUS.md) and [question coverage](PROBLEM_STATEMENT_COVERAGE.md).

## Verified files and remaining provenance gaps

| Relation / filename | Rows | Features (numeric/categorical) | Majority / minority | IR | Missing | Feature / complete duplicates | Fold pairs |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: |
| ecoli2 / ecoli2.dat | 336 | 7 (7/0) | 284 / 52 | 5.461538 | 0 | 0 / 0 | 0 |
| glass6 / glass6.dat | 214 | 9 (9/0) | 185 / 29 | 6.379310 | 0 | 1 / 1 | 0 |
| yeast3 / yeast3.dat | 1484 | 8 (8/0) | 1321 / 163 | 8.104294 | 0 | 31 / 31 | 0 |
| page-blocks0 / page-blocks0.dat | 5472 | 10 (10/0) | 4913 / 559 | 8.788909 | 0 | 79 / 69 | 0 |
| ecoli4 / ecoli4.dat | 336 | 7 (7/0) | 316 / 20 | 15.800000 | 0 | 0 / 0 | 0 |
| yeast4 / yeast4.dat | 1484 | 8 (8/0) | 1433 / 51 | 28.098039 | 0 | 31 / 31 | 0 |
| segment0 / segment0.dat | 2308 | 19 (19/0) | 1979 / 329 | 6.015198 | 0 | 224 / 224 | 0 |
| abalone19 / abalone19.dat | 4174 | 8 (7/1) | 4142 / 32 | 129.437500 | 0 | 0 / 0 | 0 |

All target labels are negative/positive (minority positive), missingness zero.
Exact feature names/types/hashes are in the tracked manifest. Numeric distances
are unsuitable for Abalone19's Sex without future training-only preprocessing.
Provider URL/license, original class grouping, record IDs and official fold protocol
remain unverified. Glass2, Pima and Ecoli1 are **not supplied** and no statistics
are asserted. Keep raw files and existing source folds unchanged; do not regenerate
folds to fill this task's missing-official-fold requirement.

Future benchmark selection needs authenticated variant metadata/license, split
provenance and an independent outer protocol. Test mapping must follow training
or a documented scientific positive outcome, not test frequencies. Related binary
variants and their fold files are not independent extra datasets. No automatic
acquisition or full experiment is authorized by this plan.

## Preliminary label-budget feasibility

The table gives **floor(fraction × full-file minority count)** as a transparent
planning calculation, not actual masked fitting counts or permission to train on
full data. An outer training fold and F/H/A split can only reduce usable anchors.
Actual feasibility must be checked for each future fitting partition/mask.

| Dataset | Full minority | 10% | 25% | 50% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ecoli2 | 52 | 5 | 13 | 26 | 52 |
| glass6 | 29 | 2 | 7 | 14 | 29 |
| yeast3 | 163 | 16 | 40 | 81 | 163 |
| page-blocks0 | 559 | 55 | 139 | 279 | 559 |
| ecoli4 | 20 | 2 | 5 | 10 | 20 |
| yeast4 | 51 | 5 | 12 | 25 | 51 |
| segment0 | 329 | 32 | 82 | 164 | 329 |
| abalone19 | 32 | 3 | 8 | 16 | 32 |

Declare the exact mask rounding before testing, with deterministic stratified
allocation when feasible. Do not retry masks using hidden outcomes. Require two
clean same-class anchors for any interpolation arm, and **six** if a fixed five
neighbor protocol is required; effective-k alternatives must be explicit separate
protocols. Under the floor planning convention, 10% conditions for Glass6,
Ecoli4, Yeast4 and Abalone19 already fail the fixed-k=5 six-anchor requirement
on the full file; mark that fixed-k condition **infeasible**. Yeast4 25% and
Abalone19 25% are only preliminary counts; independent splits/noise can still
make them infeasible. Do not silently inflate minority proportions, expose hidden
labels or relax k to rescue an arm. A documented clamped-k software smoke run is
not evidence of statistically meaningful benefit detection.

H/A labels count in the total visible-label budget. Audit n_y or n_1r=0 makes
comparisons undefined; tiny positive counts may yield uninformative intervals,
which must be marked inconclusive and fail the predeclared power criterion.
Minority anchor feasibility and audit power are distinct. No actual mask experiment
is reported because official folds are absent and the new mechanism is unapproved.

## Controlled simulations: proposed exact designs

The following are **our proposed distributions**, not reported paper settings or
executed experiments. Final sample sizes, parameter grids and repetitions need
Stage 4B power/compute review; they are intentionally not presented as results.
Use distinct seeded streams for fitting, selection, U and outer observations.

### A. Vary overlap with other factors controlled

For two-dimensional features and fixed minority prevalence pi, use:

$$
X\mid Y=0\sim N(-a e_1,I_2),\quad X\mid Y=1\sim N(a e_1,I_2),\quad a\ge0.
$$

Hold covariance, fitting sample counts, minority prevalence, label noise and
dimensions fixed while changing a. A direct density calculation gives
log(p_1/p_0)=2a x_1 and balanced Bayes error Phi(-a). At a=0 the class-conditionals
coincide and balanced error is 1/2; for larger a they separate. These are analytic
references, not an oracle accessible to the method. The Bayes-risk derivation
is in [gap analysis §4](MATHEMATICAL_GAP_ANALYSIS.md).

Predeclare a boundary band |log(p_1/p_0)|<=b for fixed b>0 and report its test
counts and minority recall. At a=0 there is no unique informative boundary; report
global ambiguity rather than comparing a misleading narrow-band metric.
Report expected synthetic-label mismatch 1-eta(z) for minority-assigned z,
where eta(z)=pi p_1(z)/[pi p_1(z)+(1-pi)p_0(z)], plus fraction in p_0(z)>p_1(z).
These are different diagnostics: natural-prior label ambiguity versus a
class-balanced decision region. They are not observed synthetic truth labels.
Data geometry is unchanged by augmentation; measure synthetic mass in ambiguous
regions and prediction harm, not a claim that generating points changes p_X.

### B. Unequal minority subgroup representation

Keep the evaluation distribution fixed with two minority components:

$$
p_1=\tfrac12N((-2,0),I_2)+\tfrac12N((2,0),I_2),\quad
p_0=N((0,3),I_2).
$$

At fixed total fitting minority count, deliberately assign alpha versus 1-alpha
of fitting minority observations to the two components, for predeclared alpha.
This models unequal representation without confounding it with changing test
population mass; it is a designed fitting-to-evaluation subgroup shift. Also
consider a separately declared arm with equal fitting/evaluation mixture weights
if population imbalance itself is the target. Do not mix the arms' interpretations.
Expose neither latent component IDs nor test labels to region selection; they
are diagnostic truth for per-component recall, generated allocation and coverage.
Evaluate both scarce and dense components; unequal counts alone do not prove
the sparse component needs interpolation. Compare training-defined regions with
simulation-truth subgroups without matching them using outer labels.

### C. Label noise and minority size

Use a fixed A or B geometry and a factorial grid of minority fitting count n_1
and training-label flip probability nu. Hold other factors constant; record
latent and observed fitting counts because flipping labels changes them.
The primary arm corrupts fitting labels only and evaluates the clean population:
this tests training-contamination robustness, not irreducible population noise.
Any alternative noisy evaluation target must be declared separately with its
own risk/meaning. Hidden clean fitting labels are diagnostic-only. Never clean
anchors using latent truth available only to the simulator.

### D. Hidden-label semi-supervised setting

Within the outer-training allocation, hide a predeclared fraction of **fitting**
labels to form U. Proposed labelled proportions are 10%, 25%, 50% and 100%.
Use the same masks and genuinely visible labels across methods; saved hidden
truth never enters a sampler, region selector, graph or pseudo-label calibration.
For simulated mask construction, distinguish uniform missing-at-random from a
class-stratified controlled-budget arm; the latter uses latent labels to construct
the experimental allocation, not as algorithm input, and has different assumptions.

Retain independent labelled H/A to estimate benefit, and count their labels in
the total budget. State both fitting fraction and total visible-label count;
10% fitting labels plus H/A is not “10% total labels”. If a required class is
missing or audit counts are inadequate, report infeasible/inconclusive, not a
silent mask retry chosen for favorable results. At 100% labels with no external
U, the hidden-label U pool is empty; do not add a secret extra pool for that arm.
An external-U experiment is separate with declared provenance and budget.

No outer-test feature or label enters U. Equal U access means equal eligible
data provision for U-capable comparators, not forcing labelled-only baselines to
use it. Compare no U, same-distribution U, volume-matched U, shifted U and
feature-destroyed U; explain transformations and avoid confounding size/compute.
Test ordinary graph SSL/self-training and faithfully implementable resampling
baselines; SSHR does use a separate U pool [H pp4–6], but its exact experimental label-mask and printed metric conventions remain unresolved [H pp6–7].

## Learners, baselines and tuning opportunities

Initial learners: Logistic Regression, Random Forest and RBF-SVM if feasible.
Preserve the original UGO classifier API: it needs predict_proba; probability
calibration/SVM probability fits require fitting-only or declared inner data,
counted in the budget. Do not use outer labels for calibration [S1, S4].

Core controls: no resampling (default and tuned threshold), class weighting
with tuned threshold, random oversampling, SMOTE, Borderline-SMOTE, source-audited
UGO with its explicit cap/pool conventions, and a validation-tuned synthetic-count baseline. Include KMeansSMOTE/ADASYN,
matched point weights and group-valuation/influence controls where feasible.
SSHR/SLEO/GDEO full methods are accessible, with specific reproduction ambiguities
[H pp6–7; L pp5–7; G pp4–6]. An approximation cannot be labelled faithful. Future DRA-UGO appears only after
Stage 4B selects a mechanism and later implementation is separately approved.
[R, D, I, S1, S3, S5, S6 in the matrix] identify why these are strong alternatives.

Give every eligible method comparable threshold, hyperparameter, count-search,
label and compute budgets. Thresholds/counts tune on H or a prespecified nested
inner procedure; fixed-candidate comparisons use independent A. Resampling never
occurs before splitting. Full-inner-label supervised controls must have their
own independent tuning/evaluation; if trained on A they cannot be scored on A
using the fixed-candidate independence bound.

## Metrics and question-specific evaluation

| Metric/diagnostic | Role and reporting |
| --- | --- |
| AUPRC | Declare average precision (AP) as primary implementation, not trapezoidal PR area; report prevalence. Same scores/ranks before threshold tuning [S1, S2] |
| MCC, balanced accuracy, F1 | Decision quality under declared validation-chosen thresholds; specify positive label and zero-denominator handling [S2] |
| Minority recall / majority FPR | Report together, with counts and declared practical harm margins |
| AUROC | Supplementary ranking metric, not a substitute for minority PR behavior [S2] |
| Boundary minority recall | A, on known simulation band and separately declared fitting-defined empirical proxy; proxy is not true boundary |
| Expected mismatch / ambiguous-region synthetic mass | A, simulation diagnostics using known densities, never candidate selection |
| Minority-subgroup recall and regional coverage | B, per-component truth in simulations and frozen fitting-defined regions on benchmarks; report each count and uncertainty |
| Increment from U | C, same label budget/masks and matched no-U/SSL/structural controls across fractions |
| Synthetic count and computational cost | Report by generated class, accepted/rejected candidates, fits, wall time, memory and hardware |

Define coverage before evaluation, e.g. test minority fraction in each frozen
region and recall at a fixed predeclared operating constraint; do not equate
more generated points with better coverage. Undefined subgroup metrics remain
undefined. UGO can select the majority class in the preserved implementation;
report that behavior rather than forcing it to be minority-only.

## Selection, final evaluation and statistical comparisons

Use F for fitting/generation, H for tuning, A for independent candidate audit and
O for final outer evaluation. Preprocessing/regions/graphs fit on F and declared
U only. Freeze the complete procedure, refit policy and hypotheses before O.
Known simulation truth measures harm after decisions; it is not a selection input.
Preserve authoritative outer folds when verified and suitable; do not treat
fold files as independent datasets. Inner label hiding happens anew within each
outer-training allocation, never across the outer boundary [S4].

Ablate uncertainty proposals, regional versus random/count allocation, finite
benefit selection versus count tuning, harm gate, graph support, U and label
fractions. Each is a controlled comparison, not proof its ingredients are novel.
Use the finite-family bounds only for candidates independent of A; adaptive
search requires fresh blocks/confirmation or a justified nested analysis.
The elementary bound does not give an AP guarantee or count all AUC pairs as
independent; see [CANDIDATE_HYPOTHESES.md](CANDIDATE_HYPOTHESES.md).

Statistical comparisons address the frozen selected procedure on independent
outer observations. Respect shared folds, repeated seeds, patient groups and
related dataset variants; repetitions are not independent new labels. Report
paired effects, suitable uncertainty, practical margins and prespecified
multiple-comparison handling. Fix a primary metric and comparisons before
testing; do not cherry-pick favorable datasets/thresholds. Wide intervals mean
inconclusive, not no benefit or safe generation. No tests/results are fabricated.

Implementation remains gated on source/provenance checks, statistical power,
budget review and user approval. This document authorizes no experiment.
