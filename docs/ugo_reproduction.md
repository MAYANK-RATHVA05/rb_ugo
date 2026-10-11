# Original UGO reproduction: provisional Stage 3 implementation

## Evidence and status

**This is a prompt-based provisional implementation, not a verified faithful
reproduction.** No paper PDF was available in the workspace. Requests to the
supplied DOI and Crossref metadata endpoint returned proxy `403 Forbidden`.
Consequently, the paper's equations, pseudocode, references, and experimental
settings have not been independently inspected. Runtime use requires
`allow_provisional=True`, emits a warning, and records this status in diagnostics.

User-supplied bibliographic information (not independently verified): Fatih
Sağlam and Mehmet Ali Cengiz, “Uncertainty guided oversampling: A classifier-specific
approach for effective handling of class imbalance in machine learning,”
Knowledge-Based Systems, 2026, DOI `10.1016/j.knosys.2026.115886`.

The section/table references below identify the intended source locations given
by the user. They do not imply those pages were read. No author implementation or
paper benchmark was obtained. This stage does not implement Risk-Budgeted UGO,
held-out batch acceptance, FPR constraints, or any claimed improvement.

## What the implementation does

An internal probability classifier measures disagreement with known fitting
labels. UGO chooses the class with the greatest mean disagreement and generates
additional examples near its most informative clean original samples. It fits a
new cloned classifier on the augmented fitting data at every iteration. A Welch
test can stop the process; a finite budget and iteration guard prevent runaway
generation. No held-out data are accepted or accessed.

`UGO.fit_resample(X, y)` returns numerical NumPy arrays. Original samples are
retained at the start of the output, in the original order; labels are preserved.
The caller's inputs and estimator are not fitted or modified in place. Features
must already be a finite, dense, real numerical matrix. Do not pass category
codes as meaningful distances. Typed pandas categorical columns are rejected;
already-converted integer codes cannot be recognized automatically. A future
evaluation stage must fit any preprocessing on fitting partitions only.

## MMA: supplied Section 3.2 equation

For a known label y and classifier probabilities:

\[
U_{MMA}(x,y)=\max_{k\ne y}P(k\mid x)-P(y\mid x).
\]

With two classes whose probabilities sum to one:

\[
U_{MMA}(x,y)=1-2P(y\mid x)\in[-1,1].
\]

For example, a true-class probability of 0.9 gives -0.8; 0.2 gives +0.6.
Positive values indicate prediction disagreement with the known label. Ordinary
margin uncertainty compares the largest predicted probabilities without needing
the known label; MMA is label-aware and signed. It is not entropy or ordinary
least confidence.

`modified_margin_uncertainty` uses `classifier.classes_` order, including reversed
columns and string labels. It checks two columns, row alignment, distinct classes,
label membership, finite probabilities, [0,1] bounds, and row sums (absolute
tolerance 1e-8). It computes other-minus-true, without silently renormalizing.

## Noise detection: supplied Tables 2 and 3

Pre-detection uses Euclidean k-nearest neighbors on original fitting data. It
excludes a sample by its **row index**, not every zero-distance neighbor. Thus
distinct coincident records remain neighbors. A strict opposing neighbor majority
marks a sample noisy. The boolean mask leaves the dataset intact.

Generated points are checked against **all original fitting rows**. A generated
point whose neighbor majority opposes its assigned label is replaced by the
nearest original, pre-clean, same-class row. Relocation copies that row's actual
coordinates; it does not discard the point or invent a different movement rule.

Provisional conventions pending inspection of the tables:

- A tied binary vote retains the assigned/true label.
- Equal-distance neighbors use ascending reference row index.
- Effective pre-k is min(k_pre, N_original-1); final-k is min(k_after, N_original).
- The pre-mask is computed once. Generated points never become relocation
  references. The standalone relocation function recomputes the pre-mask unless
  an aligned boolean reference mask is supplied.
- Noisy originals remain in the classifier's fitting set, but cannot be seeds,
  uncertainty-pool members, or relocation destinations.
- No eligible relocation destination is an explicit error; no silent dropping.

Reference eligibility and voting against noisy originals have not been verified
against Table 3. Pairwise distances use a full distance matrix with quadratic
memory use; this stage targets small software checks, not large benchmarks.

## Uncertainty and seed selection: Sections 3.1 and 3.4 / Table 4

The scoring and seed pool is provisionally **original pre-clean fitting rows**.
The model fits on all current augmented fitting rows, but class means are measured
on the fixed original clean pool. Synthetic points do not enter that pool.

For class j:

\[
\overline U_j=\frac{1}{|E_j|}\sum_{i\in E_j}U_{MMA}(x_i,y_i),
\qquad c=\arg\max_j\overline U_j,
\]

where E_j is the eligible pool in class j. The selected class may be the original
majority. The implementation does not force minority-only oversampling, nor stop
merely because the class counts are balanced. Class ordering uses `np.unique`;
an exact mean tie would select its first class, although a defined Welch test
normally stops first in this case.

Within c, select the largest MMA scores, with original row index resolving ties:

\[
n_{select}=\max(1,\lceil\lambda_{select}|E_c|\rceil).
\]

The provisional batch size is:

\[
n_{batch}=\min(B_{remaining},
\max(1,\lceil\lambda_{step}N_{c,current}\rceil)).
\]

The paper's selection pool, rounding rules, and batch-size denominator were not
available. These formulas are declared assumptions, not verified paper equations.

## ROS and SMOTE generation

UGO-ROS draws selected seeds uniformly with replacement and copies their feature
vectors. Labels are the selected class.

UGO-SMOTE also draws only selected seeds. For each seed, choose uniformly among
its nearest original pre-clean same-class neighbors, excluding the seed itself.
The neighbor need not be a selected seed. Draw r uniformly on [0,1) and use:

\[
x_{new}=(1-r)x_{seed}+r x_{neighbor}.
\]

This is interpolation anchored at an informative seed, not ordinary SMOTE run on
the entire dataset. Neighborhood size defaults provisionally to k_smote=5 and is
clamped to eligible same-class count minus one. Distinct identical points are
valid neighbors; interpolation can legitimately produce duplicate coordinates.
Fewer than two eligible same-class rows terminates with an explicit reason and
does not switch algorithms. In practice, a too-small Welch group can stop earlier.

ROS draws, interpolation neighborhood eligibility, and step distribution need
comparison against Section 3.4 and Section 4.2's resampling settings.

## Welch stopping: supplied Section 3.3 rule

For the two eligible class uncertainty groups, let n_j, mean_j, and s_j² denote
sample counts, means, and sample variances. The unequal-variance statistic is:

\[
t=\frac{mean_0-mean_1}{\sqrt{s_0^2/n_0+s_1^2/n_1}},
\quad
\nu=\frac{(s_0^2/n_0+s_1^2/n_1)^2}
{(s_0^2/n_0)^2/(n_0-1)+(s_1^2/n_1)^2/(n_1-1)}.
\]

For two groups with positive variances, Welch ANOVA has F=t² with degrees of
freedom (1, nu), and its upper-tail p-value equals the two-sided Welch t-test
p-value. A unit test verifies this identity using SciPy's t and F distributions.
Implementation uses `ttest_ind(equal_var=False, alternative="two-sided")`.

If a **defined** p-value is at least alpha=0.05, stop: the equal-means null was
not rejected. This is not proof that uncertainties are equal. One constant group
can still have a defined Welch t-test when the other varies; two constant groups
(equal or different means), groups smaller than two, nonfinite results, invalid
p-values, or nonpositive degrees of freedom have an explicit undefined status.

The loop stops on undefined tests as an **engineering safeguard**, with a reason
different from `welch_fail_to_reject`. Undefined p-values remain `None`, not a
fabricated 1.0. Handling of degenerate groups is not claimed as a paper rule.

## Generation cap: unresolved Section 3.3 versus Table 4

The prompt reports differing presentations, but supplies neither full expression.
The pages could not be inspected, so the discrepancy cannot be resolved here.
The current implementation uses this **provisional total-size interpretation**:

\[
B=\max(0,\lfloor\phi C N_{max,original}-N_{original}\rfloor).
\]

Here C=2, phi defaults to 2.5, and all size quantities are fixed from original
fitting data. The assumed total augmented budget is phi*C*N_max; subtracting the
original count gives the generated-only budget. This choice uses the quantities
mentioned in the prompt, but is not established as the published expression.

Every batch is clipped to the remaining B. `provisional_generation_cap` records
this reason. `max_iter=100` independently bounds scoring/fit iterations and is
an engineering safeguard. Reaching it records `max_iterations_safeguard`, not
paper convergence. The implementation has no unbounded while loop.

Before claiming faithful reproduction, provide Section 3.3's full cap equation
and Table 4's loop condition, determine whether the cap counts generated or
augmented samples, resolve rounding/comparison/overshoot rules, and replace or
confirm this provisional interpretation with tests against the source.

## Parameters and diagnostics

| Parameter | Default | Meaning/status |
| --- | --- | --- |
| lambda_select | 0.5 | Supplied default; provisional ceil selection fraction |
| lambda_step | 0.1 | Supplied default; provisional ceil current-class batch fraction |
| phi | 2.5 | Supplied default; cap expression unverified |
| k_pre | 5 | Initial neighbors, effective size recorded |
| k_after | 1 | Generated-point voting neighbors |
| alpha | 0.05 | Defined Welch failure-to-reject threshold |
| random_state | 42 | Generator seed; fills unset estimator random_state parameters |
| variant | ros | ros or smote; no automatic variant fallback |
| k_smote | 5 | Provisional same-class interpolation neighborhood |
| max_iter | 100 | Engineering bound |
| allow_provisional | False | Must explicitly opt into unresolved reproduction |

The estimator must be cloneable and implement fit/predict_proba. Explicit random
states on it or nested estimators are preserved; unset ones are filled with 42
(or the requested seed). Random-state bindings are reported. Reproducibility
still depends on estimator behavior and the pinned execution environment; an
arbitrary estimator that ignores its seed cannot be made deterministic here.

`diagnostics_` includes original/final label-count records, synthetic and scoring
iteration counts, pre-noise and relocated-generated counts, the provisional cap,
effective parameters, warnings, and stopping reason. Each history record includes
class means, Welch t/p/df/status, selected class (None if stopping preceded
selection), seeds, drawn seeds, interpolation partners/steps when applicable,
relocation destinations, and generated counts. It contains no performance metrics.

`estimator_` is the last internal **scoring** classifier: when its last iteration
added a batch, it has not fitted that final batch. It is not a final evaluation
model. A no-op can have no estimator_ if it stops before fitting. No validation or
test-set API is exposed; callers must supply only fitting data. The software
cannot infer whether a caller already leaked held-out rows into X.

## Tests and demonstration

Run from the repository root with Python 3.11:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python main_stage3.py
```

Tests cover exact MMA values, reversed probability columns, invalid probabilities,
self exclusion, coincident neighbors, actual clean-reference relocation
coordinates, vote ties, Welch decisions/undefined results, cap arithmetic and
batch truncation, informative seed ranking, majority-class selection, ROS copies,
SMOTE interpolation reconstruction, labels/shapes, input preservation, repeated
seed results, cloned growing fitting sets, original-only reference data,
categorical/nonfinite rejection, missing probability support, no-ops, balanced
data, iteration termination, and both real-estimator variants. Previous stage
tests remain in the same suite. Cap tests verify this declared formula, not the
inaccessible paper. Controlled test estimators are software fixtures, not models
used to claim performance.

The demo uses make_classification with 120 rows, four numerical features,
weights [0.8,0.2], class_sep=0.5, flip_y=0, and seed 42. It uses internal
LogisticRegression(max_iter=1000, random_state=42), with no final model or scoring.
The overlap makes this a useful generation-path sanity check; it is not the
paper's experimental configuration. Both variants print counts, assumptions,
effective parameters, first-iteration details, and stopping reasons.

## Still required for research reproduction

Obtain the complete paper and, if available, its authoritative code; reconcile
Tables 2–4 and Sections 3.1–3.4 and 4.2 against every provisional choice above.
The internal classifier, its settings, reference pools, preprocessing, dataset
variants, splits, repeated seeds, and evaluation protocol still require source
verification. No benchmark evaluation, baseline comparison, empirical superiority,
or novelty claim has been established. Future evaluation must keep outer-test
data outside all fitting, synthesis, uncertainty, stopping, and selection steps.

Stop after Stage 3; further stages require user approval.
