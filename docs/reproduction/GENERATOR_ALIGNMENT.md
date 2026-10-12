# Stage 4R-3: generator and comparator alignment

The complete original RWO, MCSMOTE and LEAO PDFs are now available. Their
algorithm sections and relevant settings were inspected. This stage implements
numerical versions of all three, UGO-RWO and
UGO-MCSMOTE, and the standalone paper comparator set: NORES, RUS, ROS, SMOTE,
RWO, MCSMOTE and LEAO. This is software alignment to published equations with
declared choices, not author-code equivalence or reproduced experimental results.
The original UGO framework's cap, pools and rounding ambiguities still apply.

## Source identity

| ID | Paper | Relevant source |
| --- | --- | --- |
| RWO-PDF | Zhang and Li, *RWO-Sampling*, Information Fusion 20 (2014), 99–116, DOI 10.1016/j.inffus.2013.12.003 | §3, Eq. 4 and Algorithm 1 |
| MCSMOTE-PDF | Sağlam and Cengiz, *MCSMOTE*, Applied Soft Computing 185 (2025), 113906, DOI 10.1016/j.asoc.2025.113906 | §3, Eqs. 1–4 and Algorithms 1–2 |
| LEAO-PDF | Tao, *Local entropy-adversarial oversampling*, Neurocomputing 661 (2026), 131959, DOI 10.1016/j.neucom.2025.131959 | §3, Eqs. 1–6, Algorithm 1 and §4.2 defaults |

All three files were supplied by the user. Their sizes, page counts and SHA-256
hashes are in [SOURCE_REGISTER.json](SOURCE_REGISTER.json). They are not tracked
in Git. The older SLEO upload is a different algorithm and is not used for LEAO.
The checksum-verified third-party RWO package read earlier remains listed for
history. The original RWO paper is the implementation's current primary source.

## RWO

Original Eq. 4 and Algorithm 1 use a population variance with denominator
`n`, Gaussian noise and `sqrt(n)` scaling:

    x_new[j] = x_original[j] + Normal(0, 1) * population_std(class_feature_j) / sqrt(n)

The source displays a minus sign; symmetric standard normal noise gives the
same distribution for a plus sign. The standalone numerical baseline now visits
every original minority row once per full 100% oversampling pass. A final
partial pass uses a seeded random permutation. This follows Algorithm 1 where
possible while reaching an exact balance count. UGO-RWO centers on informative
seeds, drawn with replacement as UGO batches require. Its spread comes from
all original pre-clean rows of the selected class, frozen across iterations.
That UGO adaptation is a stated pool convention, not specified by RWO.

The paper also defines frequency-weighted roulette sampling for categorical
attributes. The repository currently accepts only dense numerical features;
categorical RWO is unsupported. Integer-coded categories cannot be detected
reliably by dtype. Original-source numerical support is recorded separately
from complete mixed-type RWO support. Zero variance and singleton pools yield
unchanged coordinates; nonfinite values fail explicitly.

## MCSMOTE

We use Euclidean minority distances, a row-specific quantile bandwidth,
`K[i,j] = exp(-0.5 * D[i,j]^2 / sigma[i]^2)`, and alternating Sinkhorn row and
column normalization until both marginals are within `tolerance`. A transition
target is drawn from the selected seed's normalized transition row. The new
point uses Eq. 1: `(1-lambda)*seed + lambda*target`, `lambda ~ U(0,1)`.

Published conflicts and our resolutions:

- Algorithm 2 line 12 prints a **minus** where Eq. 1 and the surrounding text
  prescribe convex interpolation. We follow Eq. 1; diagnostics record it.
- Algorithm 2 line 14 says `until n_syn < n_needed`, the inverse of the desired
  stopping condition. We generate exactly the requested number of rows.
- Algorithm 1 does not say whether row-distance quantiles include the zero
  diagonal. Including it can make `sigma=0`, particularly at small `q`. We
  use strictly positive distances. All-coincident pools use a uniform
  transition. A singleton has transition probability one to itself.
- Sinkhorn's `1e-4` marginal tolerance is looser than NumPy's sampling-row
  sum tolerance. We normalize the chosen row once more for sampling only.
  Nonconvergence raises explicitly after `max_sinkhorn_iter`.
- The paper tunes `q` per dataset by cross-validation without specifying one
  universal value. `q=0.1` here is a declared software default, **not** an
  asserted author setting. `tolerance=1e-4` is one paper-suggested value.

Standalone MCSMOTE uses original minority rows and balances counts. UGO-MCSMOTE
starts only from informative original clean seeds; its target matrix uses all
original clean rows of the selected class. The matrix is recomputed per batch,
which has quadratic memory in the selected-class pool. The source does not
specify this UGO integration or a numerical overflow policy. Both interfaces
record `original_source_with_declared_conventions` and require provisional
opt-in. Generation uses fitting data only.

## LEAO

The comparator estimates minority probabilities with a fitting-only bagging
classifier, computes entropy, allocates synthesis by entropy weight, chooses
nearby majority adversaries, and iterates the Mahalanobis potential gradient
from each minority seed. The source's §4.2 recommended defaults are
`iterations=50`, `alpha=1`, `k=3`, `stop_probability=0.3`, and bagging
probabilities. No test or validation rows enter the probability estimator.

Operational conventions needed to run Algorithm 1:

- Printed Eq. 1 omits the negative sign of standard binary entropy, while
  the text requires high positive entropy for uncertain rows and Eq. 5 uses
  it as nonnegative sampling weight. We use standard positive entropy with
  probability clipping at `1e-12`.
- Eq. 6 gives fractional allocations. Largest-remainder rounding assigns
  exactly the required integer total, with stable original-index ties.
- Covariance can be singular. We use its symmetric pseudoinverse and define
  a zero gradient exactly at either reference point.
- The paper recommends bagging but does not specify ensemble size; we use
  ten default decision trees with seed 42. The initial step-size distribution
  is also unspecified; we draw uniformly from `(0, step)` per instance,
  with `step=1` as a declared software default.
- Majority neighbors use Mahalanobis distance, matching the objective's
  metric. A random majority row is selected from the nearest `k`.
- Algorithm 1 can stop before any update or reach its iteration limit
  without assigning `x_new`. We return the current iterate, which may be
  the seed. We stop after a nonpositive updated potential as printed.

These conventions materially affect results. LEAO is exposed as provisional
and must be evaluated as such. It returns exact class balance on numeric data.

## Comparator interface and usage

`make_comparator(name, random_state=42)` returns a fresh estimator for
`nores`, `rus`, `ros`, `smote`, `rwo`, `mcsmote` or `leao`. The first four use
an identity resampler or the pinned `imbalanced-learn==0.12.4` implementations.
RUS is named in UGO §4.2 but omitted from the main result tables; keep its
results distinguishable when the evaluation stage is built.

```python
from sklearn.linear_model import LogisticRegression
from src.comparators import make_comparator
from src.ugo import UGO

baseline = make_comparator("mcsmote", random_state=42)
guided = UGO(LogisticRegression(max_iter=1000), variant="mcsmote",
             random_state=42, allow_provisional=True)
# Run each independently on one fitting fold after training-only preprocessing:
# X_base, y_base = baseline.fit_resample(X_fit, y_fit)
# X_guided, y_guided = guided.fit_resample(X_fit, y_fit)
```

Source-based checks cover Gaussian/Sinkhorn marginal arithmetic, target pools,
convex interpolation, source conflicts, RWO full-pass schedules, LEAO entropy,
integer allocation, a finite-difference gradient check, deterministic outputs,
balanced counts, invalid inputs and failures. Existing UGO tests now exercise
all four variants on class selection, fixed batches and both cap policies.
No benchmark, paper simulation, held-out metric or statistical result is claimed.
The complete Python 3.11 test suite passes 234 checks; this includes direct
resampling checks for all seven standalone comparators. The Stage 3 artificial
demo continues to exercise ROS/SMOTE without using evaluation data.
