# Original UGO reproduction: paper-checked core with unresolved assumptions

## Evidence and status

**The supplied full UGO PDF has now been inspected. Exact reproduction remains
unresolved because its stopping-limit expressions conflict and several execution
conventions are unspecified.** Runtime use therefore still requires
`allow_provisional=True`, emits a warning, and records
`paper_checked_assumptions_unresolved` in diagnostics.

Source: Fatih Sağlam and Mehmet Ali Cengiz, “Uncertainty guided oversampling:
A classifier-specific approach for effective handling of class imbalance in
machine learning,” *Knowledge-Based Systems* 342 (2026), 115886,
[DOI: 10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886).
The supplied full text verifies Section 3.2's MMA equation, Section 3.3's stopping
criteria, Section 3.4's fixed original-total generation batch, and Tables 2–4's
noise-detection and iteration descriptions. Author code and the supplementary
benchmark collection have not been obtained. No exact published-results
reproduction or methodological extension is claimed.

The Stage 3 prototype used a current-selected-class batch denominator. That has
been corrected to the original input total. This is a correction against the
paper, not a proposed improvement. The original reference/noise pool conventions
are retained below and remain explicit assumptions.

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
already-converted integer codes cannot be recognized automatically. Evaluation
must fit any preprocessing on fitting partitions only.

## MMA: verified Section 3.2 equation

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

## Noise detection: Tables 2 and 3, with explicit conventions

Pre-detection uses Euclidean k-nearest neighbors on original fitting data. It
excludes a sample by its **row index**, not every zero-distance neighbor. Thus
distinct coincident records remain neighbors. A strict opposing neighbor majority
marks a sample noisy. The boolean mask leaves the dataset intact.

Generated points are checked against **all original fitting rows**. A generated
point whose neighbor majority opposes its assigned label is replaced by the
nearest original, pre-clean, same-class row. Relocation copies that row's actual
coordinates; it does not discard the point or invent a different movement rule.

The tables establish neighbor-label checks and nearest clean same-class
relocation. The following implementation conventions are not uniquely settled
by their pseudocode:

- A tied binary vote retains the assigned/true label.
- Equal-distance neighbors use ascending reference row index.
- Effective pre-k is min(k_pre, N_original-1); final-k is min(k_after, N_original).
- The pre-mask is computed once. Generated points never become relocation
  references. The standalone relocation function recomputes the pre-mask unless
  an aligned boolean reference mask is supplied.
- Noisy originals remain in the classifier's fitting set, but cannot be seeds,
  uncertainty-pool members, or relocation destinations.
- No eligible relocation destination is an explicit error; no silent dropping.

Table 3 and the Section 3.4 prose do not settle all reference-pool updates across
iterations. Voting against all original rows and using a fixed clean-original
relocation pool remain declared choices. Pairwise distances use a full distance
matrix with quadratic memory use; this core targets small initial experiments,
not an efficient large benchmark implementation.

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

Section 3.4 defines `n_step = n * lambda_step`, and Table 4 initializes it before
the generation loop. Here `n` is the original input total, not the selected-class
size and not the growing augmented size. The implementation fixes:

\[
n_{step}=\lceil\lambda_{step}N_{original}\rceil,
\qquad n_{batch}=\min(B_{remaining},n_{step}).
\]

The original-total denominator and initialization before the loop are verified
paper rules. Ceil rounding, last-batch clipping, and applying the selection
fraction to the eligible original class count remain conventions. The paper's
`n_select = n^(c) * lambda_select` does not specify integer rounding or completely
resolve whether `n^(c)` is measured before noise exclusion.

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
the entire dataset. Neighborhood size defaults to k_smote=5 (Section 4.2) and is
clamped to eligible same-class count minus one. Distinct identical points are
valid neighbors; interpolation can legitimately produce duplicate coordinates.
Fewer than two eligible same-class rows terminates with an explicit reason and
does not switch algorithms. In practice, a too-small Welch group can stop earlier.

Section 3.4 defines a pluggable generator R rather than complete ROS/SMOTE code.
Uniform seed draws, original-clean interpolation partners, and tie/clamping
behavior are this implementation's explicit generator conventions.

## Welch stopping: Section 3.3 rule

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

The supplied full text contains two different generated-sample limits. Section
3.3 stops when `phi * max_l n^(l) * K - n <= n_syn`; Table 4 instead loops while
`phi * max_l n^(l) > n_syn`. This implementation exposes the conflict through
`cap_policy`, rather than silently treating either expression as authoritative.
The default `section_3_3` policy gives:

\[
B_{section}=\max(0,\lfloor\phi C N_{max,original}-N_{original}\rfloor).
\]

The alternative `table_4` policy gives:

\[
B_{table}=\max(0,\lfloor\phi N_{max,original}\rfloor).
\]

Here C=2, phi defaults to 2.5, and all size quantities are fixed from original
fitting data. For original counts [90,10], the policies give 350 and 225 generated
rows respectively. The expressions are verified from the paper; choosing which
one governed the authors' experiments remains unresolved. Both freezing the
counts and flooring the limits are stated implementation choices.

Every batch is clipped to the remaining B. `provisional_generation_cap` records
this reason. `max_iter=100` independently bounds scoring/fit iterations and is
an engineering safeguard. Reaching it records `max_iterations_safeguard`, not
paper convergence. The implementation has no unbounded while loop.

The literal Table 4 loop can overshoot its limit by a full batch. This code
clips the last batch and never overshoots the selected integer budget, which is
a conservative execution convention, not verified author behavior. Exact
reproduction requires clarification or authoritative source code to establish
which cap, rounding, comparison, frozen counts, and overshoot rules were used.

## Parameters and diagnostics

| Parameter | Default | Meaning/status |
| --- | --- | --- |
| lambda_select | 0.5 | Paper default; ceil eligible-class selection is a convention |
| lambda_step | 0.1 | Paper default; fixed original-total batch, ceil convention |
| phi | 2.5 | Paper default; two conflicting published cap expressions |
| cap_policy | section_3_3 | section_3_3 or table_4; records the selected expression |
| k_pre | 5 | Initial neighbors, effective size recorded |
| k_after | 1 | Generated-point voting neighbors |
| alpha | 0.05 | Defined Welch failure-to-reject threshold |
| random_state | 42 | Generator seed; fills unset estimator random_state parameters |
| variant | ros | ros or smote; no automatic variant fallback |
| k_smote | 5 | Paper SMOTE setting; partner-pool convention unresolved |
| max_iter | 100 | Engineering bound |
| allow_provisional | False | Must explicitly opt into unresolved reproduction |

The estimator must be cloneable and implement fit/predict_proba. Explicit random
states on it or nested estimators are preserved; unset ones are filled with 42
(or the requested seed). Random-state bindings are reported. Reproducibility
still depends on estimator behavior and the pinned execution environment; an
arbitrary estimator that ignores its seed cannot be made deterministic here.

`diagnostics_` includes original/final label-count records, synthetic and scoring
iteration counts, pre-noise and relocated-generated counts, the selected cap
policy/expression, fixed batch size, effective parameters, warnings, and stopping
reason. Each history record includes
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
tests remain in the same suite. Cap tests verify both published expressions and
declared rounding/truncation conventions. Fixed-batch tests check that changing
class counts cannot change the initialized original-total batch. Controlled test
estimators are software fixtures, not models
used to claim performance.

The demo uses make_classification with 120 rows, four numerical features,
weights [0.8,0.2], class_sep=0.5, flip_y=0, and seed 42. It uses internal
LogisticRegression(max_iter=1000, random_state=42), with no final model or scoring.
The overlap makes this a useful generation-path sanity check; it is not the
paper's experimental configuration. Both variants print counts, assumptions,
effective parameters, first-iteration details, and stopping reasons.

## Still required for research reproduction

Obtain supplementary materials and, if available, authoritative code. Clarify
the cap conflict and the remaining pool/rounding conventions. Reproducing all
published results also needs the exact dataset variants, missing UGO generators,
all compared methods and classifiers, splits, repeated seeds, and the evaluation
protocol. This core documentation establishes no empirical superiority or novelty
claim. Evaluation must keep outer-test
data outside all fitting, synthesis, uncertainty, stopping, and selection steps.

The user has authorized a working original-UGO training pipeline beyond Stage 3.
This does not authorize extensions or imply that the full paper benchmark is
complete.
