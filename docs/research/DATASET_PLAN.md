# Dataset, simulation and evaluation plan

Updated 2026-10-11. **Proposed studies only: no benchmark data downloaded, no
simulation executed, no model-performance results produced.** The primary
proposed provider is [KEEL's imbalanced-learning repository](https://sci2s.ugr.es/keel/imbalanced.php).
Its index was blocked; the accessible [official KEEL project README](https://raw.githubusercontent.com/SCI2SUGR/KEEL/master/README.md)
does not verify individual dataset metadata. [REFERENCE_STATUS.md](REFERENCE_STATUS.md)
records source limits; [DATASETS.md](../../DATASETS.md) preserves manual handling rules.

## KEEL shortlist and provenance gaps

Names are proposed by the user; even canonical spelling/variant mapping remains
unverified. Do not infer target definitions from suffixes or substitute a source
multiclass dataset for a KEEL binary variant.

| Proposed benchmark | Priority | Exact canonical name/target | Counts and class ratio | Feature types/missingness | Authoritative artifact and predefined folds |
| --- | --- | --- | --- | --- | --- |
| Ecoli2 | Initial | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Glass6 | Initial | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Yeast3 | Initial | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Page-blocks0 | Initial | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Ecoli4 | Initial | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Yeast4 | Initial | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Segment0 | Additional | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Glass2 | Additional | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Abalone19 | Additional | UNVERIFIED | UNVERIFIED | UNVERIFIED | KEEL proposed; archive/manifest and folds UNVERIFIED |
| Pima or Ecoli1 | Lower-imbalance stress test proposed by user | Variant choice and target UNVERIFIED | UNVERIFIED, including proposed lower-imbalance characterization | UNVERIFIED | KEEL proposed; exact problem and folds UNVERIFIED |

Before Stage 5, obtain an authoritative dataset entry and manifest for each
selected variant: provider URL, release/version/date, checksum, license, archive
members, target meanings/class grouping, units/types, missing values, full and
per-fold counts, split IDs and repeated-fold conventions. Distinguish verified
provider statistics from counts computed by the existing loader. Preserve raw
files and source folds; do not silently merge repeated copies of a dataset.
Check duplicate identities across folds and related variants before inference.

The repository loader handles a documented KEEL subset, not general ARFF. Mixed
or missing features require future fitting-only encoding/imputation/scaling;
UGO currently accepts finite numerical matrices. Positive outcome is defined
by documented domain semantics or fitting data, never outer-test frequencies.
No automatic acquisition is authorized in Stage 4A.

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
baselines; SSHR's protocol is unverified [S3 and the matrix].

## Learners, baselines and tuning opportunities

Initial learners: Logistic Regression, Random Forest and RBF-SVM if feasible.
Preserve the original UGO classifier API: it needs predict_proba; probability
calibration/SVM probability fits require fitting-only or declared inner data,
counted in the budget. Do not use outer labels for calibration [S1, S4].

Core controls: no resampling (default and tuned threshold), class weighting
with tuned threshold, random oversampling, SMOTE, Borderline-SMOTE, preserved
UGO, and a validation-tuned synthetic-count baseline. Include KMeansSMOTE/ADASYN,
matched point weights and group-valuation/influence controls where feasible.
Advanced/SSL methods including SSHR/SLEO/GDEO require verified source protocols;
an approximation cannot be labelled faithful. Future DRA-UGO appears only after
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
