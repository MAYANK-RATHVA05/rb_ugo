# SSHC implementation — SSHR Stage 1

Stage 1 implements **clustering only**, in [src/sshr_clustering.py](../src/sshr_clustering.py).
It does not remove training rows, assign training pseudo-labels, generate
synthetic rows, fit a final classifier or run a real-data experiment.
The complete SSHR estimator and runner are not yet implemented.

The source is Jiang et al. (2023), *A semi-supervised resampling method for
class-imbalanced learning*, [DOI:10.1016/j.eswa.2023.119733](https://doi.org/10.1016/j.eswa.2023.119733),
Algorithm 1 (p.4), Sections 3.1–3.3 and Eqs.1–4 (pp.4–5).
The supplied full paper was audited in Stage 0. The linked author code was
unavailable through this environment's proxy; **author-code behavior remains
UNVERIFIED**. [SSHR_METHOD_DECISIONS.md](SSHR_METHOD_DECISIONS.md) separates
verified paper rules from the interpretations below. This is an implementation
of those declared decisions, not a claim of exact author-code equivalence.

SSHR mainly adds **existing unlabelled observations with minority pseudo-labels**.
Only its later SMOTE fallback creates **new synthetic feature vectors**
(Algorithm 2 and Section 3.3.2, pp.5–6). SSHC's inferred cluster labels alone do
not turn the unlabelled pool into labelled training data.

## Interface and input boundaries

```python
from src.sshr_clustering import SSHC

clustering = SSHC(
    minority_label=1,       # fixed source-domain identity
    mu=0.5,                # paper default
    objective_mode="paper_printed",
    prune_scope="majority_only",
    tol=1e-8,
    max_split_rounds=100,
    max_refine_iter=100,
).fit(X_labeled, y_labeled, X_unlabeled)
```

`fit` returns the estimator. The matrices must be finite, dense, real numerical
arrays with equal positive feature width. The labelled pool is nonempty and
must contain exactly two observed classes. Labels may be finite real numeric,
boolean or string scalars; unsupported complex, bytes and datetime labels are
rejected rather than stringified for diagnostics. Feature boolean, categorical,
object, complex and missing-value inputs are rejected. Numeric values that
overflow centroid or Euclidean-distance calculations raise an explicit error
requesting rescaling. The module does not impute, encode or scale features.

`X_unlabeled=None` or an empty `(0, n_features)` array allows a clustering-only
ablation. There is no `y_unlabeled` input. Inputs are copied; failures clear
partial/stale fitted state. The estimator supports sklearn cloning and parameter
inspection. It is deterministic, with no random sampling and no seed parameter.

With `minority_label=None`, the smaller observed class is inferred; tied counts
require an explicit label. An explicit label may fix source-domain identity even
when a later reduced-label pool inverts class counts. Both identities remain
fixed throughout fitting.

**The caller must provide training-only pools.** This module cannot infer whether
a row originated in an outer test fold. Fold construction, source-metadata
categorical rejection, training-only preprocessing and hidden-label isolation
will be enforced in Stage 3. One-hot arrays cannot reveal their categorical
source semantics to this numerical interface.

## Implemented decisions

Algorithm 1 initializes one centroid from the observed members of each class,
then assigns all labelled and unlabelled feature rows by nearest Euclidean
centroid. Labels are not hard assignment constraints. Persistent integer cluster
IDs settle exact distance ties; observed vote ties prefer the cluster's birth
label, then the fixed class order. Dominant labels are recomputed after assignment.

An impure cluster proposes a child from the observed members of each class
differing from its current label. Its parent's centroid is retained during
splitting. All rows are reassigned before the proposed objective is evaluated.
An undefined or strictly lower objective rejects the complete proposal: centers,
IDs, birth labels, current labels and assignments are restored. No tolerance
permits a decrease. The no-progress guard compares member groups **and assigned
row labels**, ignoring changed IDs. A same-membership proposal that changes
labels and improves the objective is accepted. This bounded-stop interpretation
is not an author-verified rule; even a rejected no-progress proposal can have
different centroid coordinates.

The literal Eq.4 default is:

```text
F1_printed = 2 TP TN / [TP(TP+FP) + TN(TP+FN)]
J = mu * observed_accuracy + (1-mu) * F1_printed
```

It can exceed one. Scores are not clipped, and an anomaly flag records this
case. `objective_mode="standard_minority_f1"` explicitly substitutes conventional
minority F1; it is an interpretation, not verified author behavior. An undefined
initial objective raises an error; an undefined proposal is rejected. The
standalone `clustering_objective` function exposes counts, accuracy, the chosen
F1 expression, objective, definition status and anomalies. None of these are
held-out classifier metrics.

Pruning resolves a paper conflict: the prose targets majority clusters, while
the pseudocode targets any below-average cluster. `majority_only` is the default;
`all` exposes the alternate reading. Accuracy uses observed members only; its
average is unweighted across clusters with defined accuracy. Below-average
eligible centers are removed simultaneously and all rows are reassigned.
Pruning removes **centers, not input rows**. Algorithm 1 has no objective-decline
gate for this separate step: pruning can lower `J`. Undefined/all-erased proposals
are rejected. No monotonic-objective claim across the entire algorithm is made.

Empty clusters are discarded. Nonempty unlabelled-only clusters retain their
birth label, have `accuracy=None` and `unanchored=True`, and do not contribute
to average accuracy. These labels are inferred; a later SSHR stage must not use
such clusters as pseudo-label candidates without an observed minority anchor.
Impure final clusters are allowed and reported, as in Section 3.3 (p.5).

Refinement computes each centroid from all its assigned labelled and unlabelled
features, reassigns rows, recomputes dominant labels and evaluates the new model.
A declining/undefined proposal restores the last accepted complete model and
stops. Convergence requires unchanged memberships, IDs and labels plus maximum
absolute centroid movement at most `tol`. Iteration limits are exposed engineering
guards. `guard_reached=True` does not claim centroid convergence or purity.

## Outputs and cost

`centroids_`, `cluster_ids_`, `cluster_labels_` and `birth_labels_` have matching
cluster order. IDs are persistent and may have gaps after pruning; an ID is not
an array position. `clusters_` contains combined-row indices, where the labelled
rows come first and the unlabelled rows follow. Every input row belongs to exactly
one returned cluster. `labeled_cluster_ids_` and `unlabeled_cluster_ids_` retain
the two pool orders. `labeled_predictions_` and `objective_` describe the final
model evaluated on observed labels only.

`diagnostics_` is JSON-serializable without NaN: it records policies, initial
means/assignments, complete accepted/rejected model snapshots, objective counts,
pruning decisions, final cluster sizes, observed class counts, impurity,
unanchored status and stop reasons. Returned memberships are nearest assignments
under the returned centroids even when a guard stops fitting. On guarded or
rejected refinement, those centroids need not be means of their current members.

Let `n=n_L+n_U`, `p` be feature width, `K` the maximum number of live clusters and
`T` the number of assignment passes. Dense distances cost `O(T n K p)` and use
`O(n K)` temporary memory. Per-cluster row scans add `O(T n K)`; labelled uniqueness
operations can add `O(T n_L log n_L)`. Retained full histories use approximately
`O(T (n+Kp))` storage in addition to copied input matrices. This is an inspectable
research implementation, not a memory-optimized large-data clustering engine.

## Validation and stage boundary

Tests in [tests/test_sshr_clustering.py](../tests/test_sshr_clustering.py) use
artificial geometry and independently calculated fractions. They cover literal
and conventional objectives; accepted splits; declining and undefined proposals;
complete rollback; same-membership relabelling; unweighted pruning and both
scopes; pure, impure, singleton, duplicate and unlabelled-only clusters; nearest
ties; guards; convergence; label identities; invalid inputs; nonmutation;
JSON diagnostics; cloning; deterministic refits and failed-refit cleanup.

The baseline raw-file-absence test in `tests/test_dataset_registry.py` now uses
an empty temporary directory. Its assertions are preserved; this removes an
assumption about the developer's local data without deleting any supplied files.
Pinned optional `stochtree==0.2.1` and the existing checksum-verified C5.0 bridge
setup provide the prerequisites for the full suite. These backends do not
participate in SSHC fitting.

Validation in the cloud checkout:

| Check | Result |
| --- | --- |
| SSHC tests | 69 passed |
| SSHC plus isolated dataset-registry tests: `.venv/bin/python -m pytest -q tests/test_sshr_clustering.py tests/test_dataset_registry.py` | 75 passed in 0.66 s |
| Full suite: `.venv/bin/python -m pytest -q` | 461 passed in 14.17 s; no failures or skips |
| Existing classifier-backend/registry tests | 23 passed after backend setup |
| Dependency compatibility and initialization | 26 compatible packages; `main_stage1.py` passed |

Independent review also checked final nearest assignments and objective coherence
on artificial inputs. The no-progress regression accepts a same-membership
relabel proposal improving `J` from `2/3` to `23/33`; an identical-point case
still stops without falsely claiming purity. A distant unlabelled point in the
refinement fixture would lower `J` from `1.1` to `0.3`; the complete model is
restored, even with a large centroid tolerance. These values are hand-calculated
software fixtures, not real-data experimental results.

These checks are not evidence of predictive benefit or correct pseudo-labels.
Original UGO, BFCH-SMOTE, EC-EBO and evaluation defaults are unchanged.
Stop after the Stage 1 commit/push. **Wait for the user's “next” before Stage 2:**
majority undersampling, confidence ranking, selection of existing unlabelled
rows and the SMOTE fallback.
