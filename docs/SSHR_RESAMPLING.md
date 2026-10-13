# SSHR resampling components — Stage 2

Stage 2 adds the resampling core in [src/sshr_resampling.py](../src/sshr_resampling.py).
It consumes an existing SSHC partition and separate labelled and unlabelled
training pools. The later Stage 3 [estimator and controlled runner](SSHR_INTEGRATION.md)
now supply the public interface and training-only label-mask protocol. This
Stage 2 core itself trains no downstream classifier; real-data experiments remain
for a separately authorized Stage 4.

**Selected pseudo-labelled rows are EXISTING unlabelled observations. Only the
SMOTE fallback generates NEW synthetic feature vectors.** Cluster membership
and confidence do not verify an observation's true minority membership.

The source is Jiang et al. (2023), *A semi-supervised resampling method for
class-imbalanced learning*, [DOI:10.1016/j.eswa.2023.119733](https://doi.org/10.1016/j.eswa.2023.119733),
Algorithm 2 and Eqs.5–8 (pp.5–6), within-cluster SMOTE in Section 3.4 (p.6),
the neighbour default in Section 4.2 (p.7), and Eq.12 (p.10).
[SSHR_METHOD_DECISIONS.md](SSHR_METHOD_DECISIONS.md) retains the full-paper
audit and source conflicts. **Author-code behavior remains UNVERIFIED.** This
implementation follows verified paper statements with declared operational
interpretations; it does not establish exact author-code equivalence.

## Core interface

```python
from src.sshr_clustering import SSHC
from src.sshr_resampling import resample_from_clusters

partition = SSHC(minority_label=1).fit(X_labeled, y_labeled, X_unlabeled)
result = resample_from_clusters(
    X_labeled, y_labeled, X_unlabeled, partition,
    delta=0.1,
    removal_fraction=0.3,  # lambda in the paper
    k_neighbors=5,
    random_state=42,
    smote_pool_policy="observed_and_selected",
)
X_out, y_out = result.X_resampled, result.y_resampled
```

The function is a Stage 2 building block, not the Stage 3 `SSHR.fit_resample`
estimator. It never reclusters or mutates the supplied fitted partition. The
caller must pass the same ordered labelled and unlabelled pools used to fit it.
The numerical inputs follow SSHC's real, finite, dense, binary-label restrictions.
No hidden unlabelled labels are accepted. An empty feature pool is allowed for
an ablation. Original categorical semantics must be checked from source metadata
in the later runner; numerical one-hot coordinates alone cannot reveal them.

Engineering validation restricts `delta` and `removal_fraction` to finite values
in `[0,1]`, neighbours to a positive integer, and the seed to an integer in
`[0,2**32)`. The paper defines `lambda` as a fraction; `delta`'s upper bound,
seed range and simple JSON-compatible label restriction are implementation
limits, not verified paper requirements. Partition dimensions, class identities,
IDs, nearest assignments with persistent-ID ties and observed-label dominance
are cross-checked; these checks still do not prove external row provenance.

`SSHRResamplingResult` separately exposes retained labelled positions, selected
unlabelled positions, synthetic feature rows, output row lineage and diagnostics.
The output order is retained labelled rows in original order, selected unlabelled
rows in confidence order, then generated rows. Original labels are preserved;
all additions receive the fixed designated minority label.

The standalone `majority_removal_weight` and `minority_confidence` helpers expose
the audited equations for numerical testing. `minority_confidence` returns
`None` for a zero denominator; it does not silently return zero. Malformed inputs
raise `ValueError`. An algorithm failure caused by class erasure raises
`SSHRResamplingError`, whose `.diagnostics` retain the failed cleaning counts
and causes. No successful result or addition quota is returned for that failure.

## Verified rules and declared interpretations

The paper specifies small-majority-cluster deletion, removal of observed
majority rows from minority-labelled clusters, weighted majority removal,
confidence-based unlabelled selection and within-cluster SMOTE for a deficit.
Its verified defaults are `delta=0.1`, `lambda=0.3` and five SMOTE neighbours.
The following operational choices are explicit interpretations where the
paper omits details; they must not be reported as verified author-code behavior.

1. **Preliminary cleaning.** Compute majority-cluster sizes from their assigned
   labelled plus unlabelled members, then set the threshold to `delta` times
   their unweighted average size (Eq.12). Delete centers whose sizes are
   **strictly below** this threshold simultaneously. Algorithm 2 step 2.1
   removes every observed labelled member, including minority rows in an
   impure majority-labelled cluster. Those minority removals are counted
   separately. Remove observed majority rows from remaining minority clusters
   in step 2.2. Deleted-cluster unlabelled members cannot be selected later.
   Centroids, cluster labels and surviving assignments stay frozen; counts
   are updated without a second SSHC fit. Erasure of either observed class
   raises a diagnosed failure rather than repairing it using hidden labels.

2. **Integer quotas.** After preliminary cleaning, fix
   `G=max(0,N_majority-N_minority)`, `n_U=floor(removal_fraction*G)` and
   `n_O=G-n_U`. This quota timing and complementary rounding are declared
   interpretations of Eqs.6 and 8. If the fixed designated majority is no
   longer larger, `G=0`: no additional balancing is requested, and identities
   are not swapped. Preliminary cleaning still applies to tied initial counts.

3. **Weighted removal.** Use the stable equivalent of Eq.5:
   `weight=expit(distance_to_own_center-distance_to_nearest_minority_center)`.
   Include all surviving minority-labelled centers as reference centers,
   including unanchored centers retaining a minority birth label. The paper
   does not specify an anchor restriction for this step. Rank remaining
   observed majority rows by descending weight; original labelled positions
   settle ties. Deterministic ranking instead of weighted random sampling is
   a declared choice. No minority center gives an unmet removal quota.
   That shortfall does **not** silently increase `n_O`.

4. **Confidence and pseudo-label selection.** Compute Eq.7 once, after removal:
   `confidence=(n_plus/n)*(1-distance_to_own_center/sum_distance_to_all_centers)`.
   `n_plus` counts surviving **observed minority labels only**; `n` counts
   surviving labelled rows plus originally assigned unlabelled rows in the
   cluster. A surviving minority-labelled cluster needs an observed minority
   anchor to be eligible. The distance denominator uses all surviving centers.
   A zero denominator is undefined and the candidate is skipped; a finite
   zero score remains eligible. Select globally by descending confidence,
   without replacement, with original unlabelled positions settling ties.
   Selection does not change other candidates' confidence counts. Coordinates
   are copied from the selected input rows, not generated or relocated.

5. **SMOTE pools and allocation.** Fill only the deficit `n_O-selected_count`.
   The default `observed_and_selected` pool contains surviving observed
   minority rows and newly selected pseudo-labelled rows within each surviving
   minority cluster. `observed_only` is an explicit alternate policy. Unselected
   unlabelled rows never become seeds or neighbours. Allocate the deficit
   proportional to eligible pool sizes using largest remainders and cluster-ID
   ties; pools need at least two row identities. This allocation is not
   specified by the paper. No cross-cluster interpolation or ROS substitution
   repairs a missing pool.

6. **SMOTE geometry and randomness.** Use
   `effective_k=min(k_neighbors,pool_size-1)` and record clamps. Exclude the
   seed by row identity, even with duplicate coordinates; distance ties use
   stable pool order. Select a seed uniformly, a neighbour uniformly among
   its nearest `effective_k` eligible rows, and a uniform interpolation
   coefficient in `[0,1)`. Record the seed. Duplicate coordinates are possible
   under ordinary interpolation and are diagnosed, not silently replaced by
   another generator. Each generated row records its cluster, anchor and
   neighbour origin/position, and interpolation coefficient.

## Diagnostics and conservation

Diagnostics distinguish preliminary majority and minority removals, weighted
majority removals, selected existing pseudo-labelled rows, SMOTE-generated rows,
unmet removal quotas and unmet addition quotas. They also record policies,
counts before/after cleaning and removal, candidate weights/confidences, cluster
sizes and anchors, allocation, neighbour clamps, excluded/undefined candidates
and coordinate duplicates. No true pseudo-label accuracy is calculated: that
requires hidden labels retained separately by the later evaluator.

The identities are:

```text
output_rows = input_labelled_rows - removed_majority_rows - removed_minority_rows
              + selected_pseudo_rows + smote_generated_rows
requested_additions = selected_pseudo_rows + smote_generated_rows + unmet_additions
requested_weighted_removals = weighted_removals + unmet_weighted_removals
```

For a positive post-clean gap, the remaining class-count gap is
`unmet_weighted_removals+unmet_additions`. A shortfall is not successful balancing.
Duplicate feature coordinates do not erase row identity: two observations can
have identical coordinates, and a generated vector can match an existing row.
The lineage records which operation produced each output row.

`coordinate_duplicates` explicitly distinguishes selected rows matching retained
labelled coordinates, selected rows matching earlier selected coordinates, and
generated rows matching retained/selected/earlier generated coordinates. A
pseudo-labelled row is not counted as a duplicate merely because it matches
its own source observation in `X_unlabeled`.

## Computational cost

Let `n=n_L+n_U`, `K` be the supplied cluster count, `p` the feature width,
`B` the generated-row budget, `m_c` a cluster's SMOTE pool size and `a_c` the
number of different anchors actually drawn in that cluster. Partition validation
and centroid distances cost `O(n K p)`, with per-cluster membership scans adding
`O(n K)`. Global ranking costs up to `O(n_L log n_L+n_U log n_U)`.
Cached per-anchor neighbour search costs
`O(sum_c a_c (m_c p+m_c log m_c))`, where `a_c<=min(m_c,B_c)`.
Generating and recording rows adds `O(Bp)`; coordinate-duplicate checks use
expected linear-time hash operations on feature tuples.

The main dense distance matrix uses `O(nK)` temporary memory; copied inputs and
outputs use `O((n+B)p)`. Neighbour caches use `O(sum_c a_c effective_k_c)` and
lineage/candidate diagnostics use `O(n+B)` records. The existing SSHC diagnostic
history is also deep-copied when present. This functional implementation is
inspectable research software, not the author's verified implementation or an
assertion of the paper's asymptotic simplification.

## Validation

Tests in [tests/test_sshr_resampling.py](../tests/test_sshr_resampling.py) use
artificial partitions and independently calculated equations. They cover
preliminary cleaning, exact threshold equality, simultaneous cluster deletion,
minority removal accounting, quotas, Eq.5 direction and stable ties, global
Eq.7 ranking/counts, zero/undefined confidence in the standalone helper,
unanchored references, alternate SMOTE pools, allocation and neighbour clamps,
self exclusion with duplicate coordinates, interpolation lineage, failures,
shortfalls, nonmutation, reproducibility and partition/input validation.

An independently checked artificial SSHC-to-core example uses labelled features
`[-1,1,10,11,12,13,14,15]`, labels `[1,1,0,0,0,0,0,0]` and unlabelled features
`[0,0.5,13]`. Final SSHC centers are `1/8` and `88/7`; the post-clean gap is four.
The default core removes labelled row 2 (feature 10), selects unlabelled rows
0 and 1 with confidences `352/711` and `338/697`, and generates one SMOTE row.
The ten output rows comprise seven retained labelled rows, two existing
pseudo-labelled rows and one generated row, with class counts five/five.
These are software-fixture checks, not real-data results or evidence of correct
pseudo-label membership.

| Validation in the cloud checkout | Result |
| --- | --- |
| Stage 2 focused tests: `.venv/bin/python -m pytest -q tests/test_sshr_resampling.py` | 84 passed |
| Combined stages: `.venv/bin/python -m pytest -q tests/test_sshr_clustering.py tests/test_sshr_resampling.py` | 153 passed in 0.67 s |
| Full repository suite: `.venv/bin/python -m pytest -q` | 545 passed in 14.18 s; no failures or skips |
| Independent source/method review | Paper equations and declared interpretations checked; artificial conservation/interpolation cases passed |
| Documentation/source validation | Supplied PDF hash, historical access status, relative links and stage boundary verified |

No new dependency or optional-backend change was required. The tested environment
uses the existing pinned Python requirements, BART package and checksum-verified
C5.0 bridge prepared in Stage 1. The changes add one core module and one test
module plus documentation; tracked pre-existing algorithm and test files are
unchanged.

## Limits and stage boundary

There is no population-level overlap guarantee, calibrated-confidence claim,
novelty claim, reproduced paper result or predictive-superiority claim here.
The printed SSHC objective conflict remains inherited from Stage 1. This stage
does not calculate the paper's printed AUC or probability ROC AUC. The paper's
unlabelled-pool construction remains unverified.

Stage 2 validation used artificial software fixtures only and was committed as
`5548cc53dbed5cd23b46f08ec4772a1680f53a73`. The user then authorized Stage 3's
separate estimator/runner, with training-only pools and identical reduced
labelled baselines. Stage 4 requires a separately authorized small real-data
study; its results are not included in either software-validation record.
