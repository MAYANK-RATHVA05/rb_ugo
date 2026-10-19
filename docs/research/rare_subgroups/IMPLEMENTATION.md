# AW baseline implementation and software checks

Status: baseline software implemented, 19 October 2026. The [pilot runner](RUNNER.md) is now prepared; research experiments have not started. Software validation uses small artificial fixtures only; it produces no research performance table or research result.

## What is implemented

[`AWSMOTE`](../../../src/aw_smote.py) is an independently written, paper-based sampler following the [fixed protocol conventions](PILOT_PROTOCOL.md). Its name in research reports remains **AW-paper-conventions**, not an exact verified reproduction of the paper or author release. The [source audit](AW_SMOTE_SOURCE_AUDIT.md) documents the differences.

The calculation proceeds as follows:

1. Estimate a radius from fifth-other-minority distances and run minority-only DBSCAN. Only clusters with at least two rows are eligible for generation.
2. Count majority neighbors for each eligible minority row, using cleaned fitting data. Original row IDs keep feature rows, labels, and tie breaking aligned.
3. Give each cluster its maximum boundary count as weight. Allocate the original-count generation budget using the paper's half-up rounding and maximum-weight correction.
4. Within each cluster, combine normalized boundary counts and inverse mean pairwise distances using the paper's sigmoid formula. Draw seeds from those probabilities.
5. Interpolate between a seed and another member of its cluster, with lambda drawn from [0,1). Return retained originals followed by the requested synthetic rows.

All-zero cluster weights use uniform cluster weights; all-zero boundary counts within a cluster use uniform boundary weights. These are disclosed conventions inherited in concept from the author release. Self exclusion is by row index, so different rows with identical coordinates remain neighbors.

There is no fitted scaler, classifier, threshold selector, validation-data input, subgroup-based weighting, or hidden fallback in this module. The caller supplies numerical **fitting-only** data on a comparable scale. The API cannot determine whether a caller has accidentally embedded subgroup metadata in X; the future runner must enforce that isolation.

## API and count conventions

```python
from src.aw_smote import AWSMOTE, AWUnsupportedError

sampler = AWSMOTE(random_state=42)  # noise_policy='remove'
X_resampled, y_resampled = sampler.fit_resample(X_fit_scaled, y_fit)
```

Input labels must be numerical 0/1, with label 1 no more frequent than label 0. Defaults are k=5, MinPts=2, epsilon percentile=65, theta=.2 and gamma=5. The radius k is independent of MinPts. At least six original minority rows are needed for the default fifth-other radius; individual rare subgroups can be smaller.

`n_samples=None` requests the majority-minus-minority count **before cleaning**. The pilot's later runner will explicitly request 900. If d minority originals are excluded, removal finishes with `n_minority−d+N` positives. This does not promise final 1:1 balance. No generation cap is silently imposed. A zero requested count returns copies of the originals without cleaning; tied class counts have a zero default request.

The constructor supports `noise_policy='retain_originals'` for **AW_SEED_FILTER_RETAIN**. With the same fitting arrays, parameters and seed, this policy produces exactly the same synthetic rows and provenance as removal, then restores all original rows in their original order. Excluded rows still cannot be seeds or partners. This is not the paper's KeepNoise or SoftNoise algorithm.

Every call starts a fresh `default_rng` from the integer seed. Repeating a call or using a scikit-learn clone reproduces its output in the pinned environment. This deliberately differs from the author's instance-level advancing `RandomState` convention. No global RNG seed is modified.

## Diagnostics and row alignment

All row IDs below refer to positions in the original X/y passed to that call, not pandas index labels. No evaluator subgroup IDs are stored or accepted.

| Attribute | Meaning |
| --- | --- |
| `epsilon_` | Fitting-only radius, after successful estimation. |
| `minority_indices_`, `majority_indices_` | Original positive and negative row IDs. |
| `cluster_labels_` | Length len(X); -2 means majority, -1 means DBSCAN noise, nonnegative values are DBSCAN cluster labels. |
| `clusters_` | Tuple of eligible original-row-ID arrays in ascending DBSCAN-label order. |
| `noise_indices_`, `excluded_indices_` | Noise rows and all minority rows ineligible for generation. |
| `cleaned_indices_`, `eligible_indices_` | Cleaned original fitting rows and eligible minority rows, in original order. |
| `retained_indices_` | Original rows actually returned before synthetic rows, on success or a zero request. |
| `boundary_neighbors_` | Original row IDs, aligned row-by-row with `eligible_indices_`, with exactly k other neighbors. |
| `boundary_counts_` | Length len(X); majority-neighbor counts for eligible rows, -1 elsewhere. |
| `cluster_weights_`, `cluster_quotas_` | Cluster maxima and exact integer generation allocations. |
| `sample_probabilities_` | Length len(X); conditional seed probabilities sum to one **within each eligible cluster**, zero outside eligible clusters. These are not global probabilities. |
| `generated_samples_` | Completed synthetic feature matrix. |
| `generated_seed_indices_`, `generated_partner_indices_` | Original fitting row IDs for each synthetic row. |
| `generated_cluster_labels_`, `interpolation_fractions_` | Per-synthetic cluster label and lambda; arrays align with `generated_samples_`. |
| `diagnostics_` | Status, input/output counts, budget reference, fitted radius, cluster sizes/weights/quotas, exclusions, seed and RNG policy. |

Synthetic provenance describes how a point was constructed. It is not an oracle label for whether the point lies in a true minority region. Attributes become available as stages complete; for example a radius failure has no cluster labels, and a zero request has no clustering results.

## Failure behavior

Invalid parameters, nonfinite/nonnumerical features, nonbinary/misaligned labels, and label 1 being the larger class raise `ValueError`. Valid inputs that cannot meet the declared sampler rules raise `AWUnsupportedError`:

| `exception.reason` | Meaning |
| --- | --- |
| `too_few_neighbors` | The requested number of other fitting references is unavailable; k is not silently reduced. |
| `nonpositive_epsilon` | The fitted percentile radius is zero, including some duplicate-only configurations. |
| `no_valid_cluster` | DBSCAN leaves no eligible cluster; no all-minority fallback is created. |
| `invalid_quota` | The printed allocation/correction rule cannot give nonnegative exact quotas. |
| `nonfinite_distances` | Finite inputs overflow in Euclidean/within-cluster distance calculations. |
| `requested_generation_count_not_met` | Completed generation did not meet N. |
| `nonfinite_generation` | Interpolation produced a nonfinite row. |

The exception carries a diagnostic snapshot, and the sampler records `status='unsupported'`. There is no returned dataset on these failures, so a runner must record the failed status rather than score unchanged data as if AW had succeeded. Partial synthetic rows are never published. Every refit clears learned state before validation, so a failed later fit cannot expose previous synthetic rows or geometry.

The quota edge case is real: six equally weighted clusters with N=4 round to six allocations of one. Correcting by subtracting two from the first cluster gives -1. This implementation reports that problem rather than substitute a different allocation method.

## Verification and next work

Run from the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_aw_smote.py
.\.venv\Scripts\python.exe -m pytest -q
```

The focused checks use independently hand-derived neighbor counts, density/fusion probabilities, quota totals and interpolation constraints. They also cover mixed feature/label order, duplicate/tied coordinates, two-dimensional Euclidean geometry, zero weights/quotas, diagnosed unsupported cases, paired retention, repeated calls/cloning, invalid input, and state clearing. A pipeline test uses a recording stub with no learned classifier to verify training-only scaling and that prediction data are neither fitted nor resampled.

Validation of the baseline stage (`adf8f45`) in the cloud checkout, Python 3.11.17 with the repository environment: **53 focused tests passed; 801 tests passed in the complete suite**, with no reported skips or failures. Document links, JSON status, and the staged diff were also checked. These historical counts describe software verification, not measured research performance; the runner's current checks are documented separately.

The runner now implements the protocol's partitions, hidden-group isolation, ordinary baselines and tuned AW selection, independent threshold calibration and report exports. Execution is authorized for the frozen pilot. KeepNoise/SoftNoise reconstruction and new sampling rules remain deferred.
