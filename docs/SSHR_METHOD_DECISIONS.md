# SSHR method decisions — source audit and staged reproduction

**Stage 2 adds SSHR resampling components to Stage 1's SSHC clustering. The full
estimator, runner integration and a real-data SSHR study are not implemented.**
This document preserves the Stage 0 source evidence and distinguishes implemented
decisions from proposed choices for later stages. See the
[SSHC implementation record](SSHC_IMPLEMENTATION.md) and
[SSHR resampling record](SSHR_RESAMPLING.md) for the APIs, checks and limits.
Existing UGO, BFCH-SMOTE, EC-EBO, evaluation code and default method lists are
unchanged.

**SSHR mainly selects EXISTING unlabelled observations and assigns minority
pseudo-labels to them. These are added real observations, not synthetic feature
vectors. NEW synthetic feature vectors are produced only by its SMOTE fallback.**
This distinction follows Algorithm 2, steps 3.1–3.3 and Section 3.3.2, pp.5–6
[P]. All future outputs must preserve the distinction in their row lineage and
diagnostics.

## 1. Sources, access, and repository state

[P] Zhen Jiang, Lingyun Zhao, Yu Lu, Yongzhao Zhan, and Qirong Mao (2023),
“A semi-supervised resampling method for class-imbalanced learning,”
*Expert Systems With Applications* 221, 119733,
[DOI:10.1016/j.eswa.2023.119733](https://doi.org/10.1016/j.eswa.2023.119733).
The supplied full **13-page PDF** was read, including the related work,
Algorithms 1–2, Eqs.1–12, experiment tables, parameter study, ablation,
conclusion, and references. Critical equations and pseudocode on pp.4–7 were
also checked visually. Page references below are the PDF's printed page numbers.

- Supplied file: `1-s2.0-S0957417423002348-main.pdf`.
- PDF size: 1,648,882 bytes.
- SHA256: `89076411fe1dafad46efb1c5dc4fd3cfaa8ee720ed3a91c29210970370318910`.
- [A] Author source/data link, explicitly printed on pp.1–2 and in Data
  availability, p.12: <https://codeocean.com/capsule/5051082/tree/v1>.
- Access attempts on 2026-10-12: a normal public request and an escalated retry
  both failed at the environment's proxy with `CONNECT tunnel failed,
  response 403`; curl exit 56, origin HTTP status `000`.

**Verified paper rules** below are directly supported by [P]. **Verified
author-code behavior: NONE.** No Code Ocean page or source archive was obtained;
language, functions, defaults, execution details, licence, archive checksum, and
source revision are **UNVERIFIED**. The proxy failure does not establish whether
the public capsule itself requires authentication. `v1` is the published URL
suffix, not a source revision inspected here. **Declared interpretations** below
are our proposed choices, not recovered author behavior or novel contributions.
The machine-readable [source audit](SSHR_SOURCE_AUDIT.json) is the historical
Stage 0 record; its `not_implemented` status describes that audit's commit.

The cloud checkout is `/workspace/rb_ugo`; the user's separate Windows checkout
is `C:\ugo\ugo_base`. Origin was fetched before branching. No existing local or
remote `sshr-implementation` branch was present. This branch was created from
the latest `ec-ebo-implementation` head,
`45852025df115eb66491a2ea36c4d5a8745fd45d`, and pushed without force. Tracked
files were clean; pre-existing untracked `data/incoming/` and `papers/` were
preserved. `main` and the friend's implementation branches are not changed by
this stage. Git history includes the earlier independent UGO pilot and its
three reverts; that pilot is not being recovered here.

## 2. Problem, inputs, and limitations

Section 3.2, p.5 defines `D_L={(x_i,y_i)}` and `D_U={x_j}`, whose labels `Y_U`
are unobserved. SSHC clusters their feature union. Observed labels guide the
partition; they are not hard constraints forcing every labelled row to remain
in its original class cluster. Each final cluster has a class label from its
dominant observed labels (Sections 3.2–3.3, p.5).

The working assumption is that observations in a compact cluster tend to share
a class label (Section 3.1, p.4). This is not a guarantee under class overlap,
noise, or an unrepresentative unlabelled pool. The paper explicitly permits
remaining impure clusters (Section 3.3, p.5). Eq.7 confidence is a geometric
heuristic, not verified calibrated class-membership probability. Table 7, p.10
reports pseudo-label accuracy as low as 0.013 on Abalone19; this is an author's
reported value, not a result reproduced here.

The Stage 3 public estimator input will be `fit_resample(X_labeled, y_labeled, X_unlabeled)` with
separate aligned finite numerical matrices and two observed classes. Hidden
labels are not an argument or fitted attribute. Empty `X_unlabeled` is allowed
for an explicitly identified ablation; missing observed classes, mismatched
widths, nonfinite values, and categorical/boolean source columns are rejected.
An explicit source-domain minority label fixes class identities throughout a
run; cleaning does not silently swap majority/minority meanings.

Numeric storage alone does not establish continuous feature semantics. The
runner must check original source feature metadata and reject categorical
sources before one-hot encoding could disguise them as continuous input. This
is a declared restriction for this reproduction, not an assertion that the
paper proves support for every categorical dataset listed in Table 2.

## 3. Verified SSHC — Algorithm 1, p.4

1. Start with **one cluster per observed class**. Assign that class's labelled
   observations and compute its centroid. There is no arbitrary initial `k`
   parameter specified by the algorithm.
2. Assign every observation in `X_L union X_U` to its nearest centroid. For
   each impure cluster `p_h`, create a child for each observed class `j` that
   differs from the parent's label. Initialize the child using the labelled
   rows of class `j` within `p_h`. Reassign all observations and compute Eq.3.
   Restore the former clustering model when the objective decreases. Repeat
   until no new clusters appear.
3. Evaluate each cluster's accuracy on labelled data and delete clusters with
   accuracy below the average cluster accuracy. The scope conflicts with the
   explanatory prose, as recorded below.
4. Reassign all observations to nearest centroids and update centroids using
   the assigned `L union U` observations. Repeat until centroids remain
   unchanged or Eq.3 decreases.

Eq.1, p.5:

```text
m_k = sum(x_i for x_i in p_k) / |p_k|
```

Eq.3 is **maximized**, using cluster-predicted labels of the observed labelled
rows, not downstream classifier scores or validation data:

```text
J = mu * Acc(Y_L, Y_L_pred) + (1 - mu) * F1(Y_L, Y_L_pred)
```

There is **no SSE term in Eq.3**. Eq.2 prints a sum of Euclidean norms,
`sum_j sum_(x_i in p_j) ||x_i-m_j||`, although the adjacent prose calls it
sum of squared error. SSHC replaces that loss with Eq.3; the inconsistency is
not a reason to introduce an SSE penalty into the reproduction.

### Objective conflict: printed Eq.4 is not conventional F1

The printed Eq.4, p.5 is:

```text
F1_printed = 2*TP*TN / (TP*(TP+FP) + TN*(TP+FN))
```

TP refers to correct minority predictions, TN to correct majority predictions,
FP to majority predicted minority, and FN to minority predicted majority. This
expression differs from standard minority `2*TP/(2*TP+FP+FN)` and contradicts
the printed assertion `F1 in [0,1]`. With perfect predictions
`TP=10,TN=100,FP=FN=0`, it gives `2000/1100 = 1.8181818`; Eq.3 with `mu=0.5`
would then give `1.4090909`.

**Declared objective policy for Stage 1:** expose `paper_printed` and
`standard_minority_f1` explicitly. Use **`paper_printed` as the literal-source
default**, with an anomaly flag when the expression exceeds one. Do not clip
it or silently substitute sklearn F1. `standard_minority_f1` is a separately
named interpretation of the prose, not verified author behavior, and must not
be chosen from outer-test scores. An undefined initial objective is an explicit
error; a candidate with an undefined objective is rejected and diagnosed.
Author code or clarification is required to identify the executed objective.

### SSHC operational decisions — declared, not author-verified

| Issue | Paper evidence / omission | Proposed explicit policy |
| --- | --- | --- |
| Distance and ties | Algorithm 1 says closest centroid; tie rules absent. | Euclidean distances; persistent cluster IDs break exact distance ties. |
| Labels | Sections 3.2–3.3 say dominant observed label; pseudocode assigns birth labels without relabelling timing. | Recompute dominant observed labels after assignment. Vote ties prefer the birth label if it is tied, otherwise the fixed input class order. Record `label_policy`. |
| Split parent centroid | Children are initialized from labelled rows; no parent recentering is printed during splitting. | Retain the parent's prior centroid while initializing children; all-member mean updates occur in final refinement. |
| Split rollback | Step 2 restores a lower-objective proposal, but may propose it again indefinitely. | Snapshot centroids, labels, birth labels, IDs, row assignments and objective. Restore and stop on rejection. Also stop if both row groups and assigned row labels are unchanged, ignoring ID renaming. Identical groups with changed labels can improve Eq.3 and must not be rejected by this guard. Record the reason. |
| Pruning scope | Algorithm 1 step 3 says **any** cluster; Section 3.2, p.5 says **majority clusters**. | Default `majority_only`, following the prose; expose literal `all` separately. This is a deliberate resolution of a conflict. |
| Cluster accuracy | `Acc(p)` / `Avg_acc(P)` weighting not defined. | Accuracy on observed labelled members; unweighted average of defined cluster accuracies, computed once before simultaneous pruning. |
| Empty / unlabelled-only cluster | No rule provided. | Remove empty clusters. Keep a nonempty unlabelled-only cluster's birth label as an inferred label, exclude it from labelled-accuracy averaging and from pseudo-label candidates, and report it. Do not invent observed labels. |
| Refinement decline | Step 4 stops when objective decreases; rollback not stated. | Restore the last accepted complete model and stop; this is a conservative interpretation. |
| Floating-point stopping | Paper says unchanged centroids; no tolerance given. | Implemented `tol=1e-8`, `max_split_rounds=100`, `max_refine_iter=100` as exposed engineering safeguards. Tolerance applies only to centroid movement with unchanged memberships and labels, never to objective decline. Hitting a guard is not paper convergence. |
| Impurity / all models pruned | Impure final clusters explicitly allowed; no all-pruned handling. | Report impurity. Reject a pruning proposal that erases the entire model, restore its predecessor, and diagnose the safeguard. |

No proof of global optimality, safe class separation, or correct pseudo-labels
is established by these decisions. Eq.3 evaluates observed fitting labels,
and improving it does not establish an improvement on an independent test set.

## 4. Verified SSHR — Algorithm 2, pp.5–6

### Majority undersampling

Algorithm 2 step 2.1 deletes a majority-labelled cluster smaller than
`min_size` and removes **all labelled rows in it** from `D_L`. If the cluster
is impure, this can include observed minority rows. Step 2.2 removes labelled
majority rows from minority-labelled clusters. These operations are distinct
from additional weighted majority removal in steps 2.3–2.4.

Section 3.3.1, p.5 states 10% of average majority-cluster size. Eq.12,
Section 4.3.3, p.10 clarifies:

```text
min_size = delta * average_majority_cluster_size; delta = 0.1
```

Thus Section 4.2's shorthand `min_size=0.1` is not an absolute 0.1-row cutoff.
For a majority row `x_i`, Eqs.5–6, p.5 give:

```text
w_i = 1 / (1 + exp(d_i_minus - d_i_plus))
n_U = lambda * (N_minus - N_plus)
```

Despite the signs, **`d_i_plus` is distance to its own cluster centroid** and
**`d_i_minus` is distance to the nearest minority-cluster centroid**. A row far
from its own centroid and near a minority centroid receives a larger removal
weight. The counts `N_minus,N_plus` are majority/minority labelled counts;
when those counts are measured relative to preliminary cleaning is unspecified.

### Existing unlabelled-point selection

For an unlabelled row `x` assigned to minority-labelled cluster `p_j`, Eq.7,
p.6 prints:

```text
C(minority | x) = (n_plus / n) *
    (1 - ||x-m_j|| / sum_(k=1..|P|) ||x-m_k||)
n_O = (1 - lambda) * (N_minus - N_plus)                 [Eq.8]
```

The paper calls `n` the cluster sample count and `n_plus` its minority sample
count; labelled-only versus total membership is not operationally clarified.
**Unobserved true labels must never supply `n_plus`.** Algorithm 2 steps 3.1–3.2
select rows according to this confidence and add their existing coordinates to
`D_L` with a minority pseudo-label. They are not generated feature vectors.

### SMOTE fallback

Algorithm 2 step 3.3 and Section 3.3.2, p.6 use SMOTE for the deficit when
eligible unlabelled rows cannot satisfy `n_O`. Section 3.4, p.6 explicitly says
SMOTE is performed **within clusters**. Section 4.2, p.7 sets its nearest
minority-neighbour count to five. The main text does not define integer
allocation among clusters, whether newly selected pseudo-labelled rows are
SMOTE seeds/neighbours, or singleton-cluster handling.

### SSHR operational decisions — declared, not author-verified

| Issue | Proposed explicit policy |
| --- | --- |
| Small-cluster size | Use assigned `L union U` membership for cluster sizes and their majority-cluster average, reflecting Eq.1; record `cluster_size_basis`. Compute threshold once and delete eligible clusters simultaneously. |
| Small-cluster labelled deletion | Literal default removes **all observed labelled rows in each deleted small majority cluster**, as Algorithm 2 step 2.1 states. Count removed minority rows separately as well as majority rows. Do not silently call this majority-only row deletion. |
| Class erasure | If preliminary cleaning erases either observed class, raise a diagnosed failure; never swap label identities or inspect hidden labels to repair it. |
| Reassignment after cleaning | No reclustering/reassignment is printed in Algorithm 2. Freeze the SSHC model; remove deleted cluster centroids from `P`, exclude their unlabelled members from selection, and update surviving membership counts after labelled-row removal. Record this interpretation. |
| Quota timing | After preliminary cleaning compute `G=max(0,N_minus_remaining-N_plus_remaining)`. Set `n_U=floor(lambda*G)`, `n_O=G-n_U`. This closes the remaining integer gap if quotas are met. Using pre-clean counts could overbalance after extra cleaning; timing and rounding are interpretations. |
| Weighted removal | Rank remaining observed majority rows by descending Eq.5 weight, without replacement; stable original labelled-row positions resolve ties. Ranking versus weighted random selection is not established by the paper. Use a numerically stable equivalent logistic calculation. Nearest-minority references include all surviving minority-labelled centroids, including unanchored birth-minority centroids; the observed-anchor restriction applies to pseudo-selection and SMOTE, not this distance reference. No surviving minority centroid gives a weighted-removal shortfall, which does not increase the addition quota. |
| Confidence counts | `n_plus` counts surviving **observed labelled minority rows**; `n` counts surviving assigned labelled rows plus assigned unlabelled rows. No iterative pseudo-label updates to confidence. An observed minority anchor is required for an eligible cluster. |
| Confidence degeneracy | A zero total centroid-distance denominator gives an undefined score; skip that candidate and report it. Finite zero scores remain rankable; no undocumented confidence threshold is introduced. No eligible minority centroid produces a diagnosed shortfall. |
| Pseudo-label ranking | Select globally among eligible minority clusters by descending Eq.7 score, without replacement. Stable original unlabelled-row positions resolve ties. Keep original coordinates and record source row IDs and scores. |
| SMOTE pool | Default `smote_pool_policy="observed_and_selected"` uses remaining observed minority rows plus newly selected pseudo-labelled rows within surviving minority-labelled clusters. Including selected rows follows the stage order but is not verified author code. Expose `observed_only` separately. Never use unselected `U` or hidden labels as SMOTE anchors. |
| SMOTE allocation | Allocate the remaining deficit among clusters with at least two eligible minority rows, proportional to those pool sizes; largest-remainder rounding and cluster-ID ties preserve an exact requested budget. This allocation is not specified by [P]. |
| Neighbours / shortage | Use `min(5,cluster_pool_size-1)` neighbours, recording clamps. Singleton/no-eligible clusters cannot interpolate; report unmet quota. Do not replace missing synthesis with ROS, arbitrary duplicates or cross-cluster interpolation. |
| Randomness / conservation | One recorded seed (project default 42) controls SMOTE choices. Preserve original retained labelled-row order, then append selected `U` rows, then SMOTE rows. Keep input arrays unchanged. |

If preliminary cleaning makes the designated majority no longer larger, request
no additional balancing (`G=0`) and report the changed counts; do not rename
classes. Preliminary cleaning still applies with tied or inverted initial
counts; `G=0` means no additional balancing, not necessarily unchanged rows.
This clarifies the Stage 0 tie-handling proposal. No fallback
is claimed to establish that generated or pseudo-labelled rows belong to the
minority class. Exposure of policy choices is required before implementation.

## 5. Printed defaults and experimental protocol

| Item | Verified paper statement | Reproduction qualification |
| --- | --- | --- |
| `mu` | 0.5; Section 4.2, p.7 and Section 4.3.3, p.10 | Weight in Eq.3; its F1 conflict remains explicit. |
| `delta` | 0.1; Eq.12, p.10 | Multiplier for average majority-cluster size. |
| `lambda` | 0.3; Sections 4.2/4.3.3 | Under/over split of labelled class gap. |
| SMOTE neighbours | 5; Section 4.2, p.7 | Cluster-size clamps/shortfalls are our safeguards. |
| Data | 44 KEEL binary problems, names/counts in Table 2, p.7 | Not the UGO paper's 74 datasets. |
| Classifier | RBF SVM, sigma=0.01, tradeoff C=100; Section 4.2, p.7 | Eq.11 prints `exp(-||x_i-x||^2/sigma)`, giving literal sklearn `gamma=1/sigma=100`; do not infer `1/(2*sigma^2)`. |
| Validation | Five-fold stratified CV, five repeats; Section 4.3, p.7 | Seeds, exact folds and unlabelled-pool construction are not stated. |
| Statistics | Friedman and Holm, alpha=0.05; Section 4.3.2, p.8 | No inference of superiority from our future two-dataset pilot. |

Sections 4.2–4.3 refer to the same setup as CDBH, but the supplied SSHR text
does not define the origin of `D_U`, label availability, a masking ratio,
masking seeds, or whether test features were permitted in `D_U`. These details
are **UNVERIFIED**. Do not infer that the authors leaked test data; equally,
do not claim that our strict training-only protocol recovers their executed
experiment. The unreachable source might resolve these omissions later.

The paper's Section 3.4 quotes approximate SSHC/resampling complexity bounds;
they are not runtime guarantees for this Python implementation. Actual
cost will depend on split/refinement rounds, number of clusters, sorting,
SMOTE neighbour search, and memory policy. No new convergence or generalization
guarantee is claimed.

## 6. Metric audit: paper AUC is not established as probability ROC AUC

Section 4.1, p.6 correctly defines `Recall=TP/(TP+FN)` and
`Precision=TP/(TP+FP)`, but prints:

```text
Specificity_printed = FP/(TN+FP)        # actually false-positive rate
Sensitivity_printed = FN/(TP+FN)        # actually false-negative rate
F_measure_printed = (1+beta)*Recall*Precision /
                    (beta^2*Recall+Precision)          [Eq.9]
AUC_printed = (1+Sensitivity_printed-Specificity_printed)/2  [Eq.10]
```

Eq.9's numerator is `1+beta`, rather than the usual `1+beta^2`; at the stated
usual `beta=1` both coincide with standard minority F1. Eq.4's clustering
objective is a **different printed expression**, so Eq.9 does not resolve it.

Literal substitution in Eq.10 yields `(1+FNR-FPR)/2`. Perfect classification
would receive 0.5; predicting every observation as majority would receive 1.
This cannot be treated as standard ROC-AUC. Even if the sensitivity definition
were corrected to true-positive rate, `(1+TPR-FPR)/2` would be a hard-label
balanced-accuracy expression, not a verified area calculated from continuous
classifier scores. The authors' actual metric computation is **UNVERIFIED**;
Tables 3–4 alone do not establish which implementation was executed.

The current repository's `src/metrics.py` computes ROC-AUC from held-out
positive-class probabilities. Keep that behavior unchanged. The future study
will label its measure **probability ROC-AUC**, use standard minority F1,
`G-mean=sqrt(TPR*TNR)`, MCC, and `FPR=FP/(FP+TN)`, and retain confusion counts.
These are the user's requested evaluation measures, not an assertion of exact
comparability to the paper's AUC column. A literal printed-metric diagnostic,
if reported, must have a separate name and must not replace probability AUC.

## 7. Stage 3 integration and controlled label-hiding protocol — not implemented

1. Freeze and save outer fold definitions before preprocessing or masking.
   The SSHR paper uses five folds/five repeats; the current runner uses ten
   folds. Extend the opt-in experiment path later without changing existing
   runner defaults, and record the chosen folds/repeat count explicitly.
2. Within **each outer training fold only**, form disjoint `L` and `U` pools
   using a deterministic, documented label mask. Proposed first pilot:
   retain 50% of each training class's labels, seed 42 plus recorded fold/repeat
   derivation, with at least two observed rows per class. Stratified masking
   is an experiment-design choice; the method never receives withheld labels.
3. Preserve hidden `y_U` only in the evaluator for retrospective pseudo-label
   accuracy. No SSHC objective, cluster label, confidence count, parameter
   choice, quota, or fallback may read it. Empty-selection accuracy is undefined,
   not 100%. In a genuinely unlabelled pool this retrospective accuracy cannot
   be measured without subsequently obtaining labels.
4. Give **NORES, SMOTE, UGO-SMOTE, BFCH-SMOTE and EC-EBO the identical reduced
   labelled set `L`**. SSHR receives that same `L` plus `X_U`. Do not train
   baselines on all original training labels while SSHR gets fewer.
5. Fit preprocessing on `L` only, transform `U` and outer-test features using
   that fitted transformation, and fit a fresh final classifier on each method's
   returned labelled data. Scaling learned from `L union U` could be a separate
   experiment, but is not silently introduced into this controlled comparison.
6. Outer-test **features and labels must never enter `U`**, preprocessing fit,
   clustering, pseudo-labelling, or resampling. Label masking is not a way to
   make an outer-test row a training observation. Preserve row identities and
   test disjointness even when predictions are cached or runs are resumed.
7. Restrict this path to original numerical features, explicitly rejecting
   categorical/boolean sources. Existing methods and default registries remain
   unchanged; SSHR is opt-in. Cache identities must include mask indices,
   policies, seeds and source/code hashes, not just method and dataset names.

## 8. Stage 2 diagnostics and conservation contract — implemented core

Report separately:

- Original and retained observed class counts; labelled rows removed by small
  clusters, minority-cluster cleaning and weighted removal, separated by class.
- **Removed majority rows**, plus any minority-row removal caused by the literal
  small-majority-cluster policy; do not hide the latter in a majority count.
- **Selected existing pseudo-labelled rows**, their source `U` IDs, cluster IDs,
  confidence values and assigned minority label.
- **SMOTE-generated rows**, source anchor/neighbour IDs, interpolation values,
  cluster allocation, effective neighbour count and duplicate coordinates.
- Requested weighted-removal and addition counts; actual counts;
  **unmet quotas** by cause; skipped undefined confidence and singleton cases.
- Cluster labels, observed-label counts, unlabelled counts, sizes, impurity,
  centroid updates, objective values/mode, rejected proposals, and stop reason.
- Every declared interpretation and engineering safeguard used.

Required accounting identities are:

```text
output_rows = input_labelled_rows - removed_majority_rows - removed_minority_rows
              + selected_pseudo_rows + smote_generated_rows
requested_additions = selected_pseudo_rows + smote_generated_rows + unmet_additions
requested_weighted_removals = weighted_removals + unmet_weighted_removals
```

No selected `U` row is counted as SMOTE-generated. Distinct original rows may
have identical coordinates; lineage distinguishes observation provenance from
coordinate uniqueness. Inputs remain intact, and every original/output row must
be traceable. Shortfalls do not imply successful class balancing. For a positive
post-clean gap, the final count gap equals unmet weighted removals plus unmet
additions. [SSHR_RESAMPLING.md](SSHR_RESAMPLING.md) documents the result object,
diagnosed class-erasure failure and actual Stage 2 validation.

## 9. Stage status and small-study gate

Stages 1 and 2 were separately authorized by the user's **“next”**. Stages 3 and
4 require a further **“next”**; they are not part of the resampling core:

| Stage | Planned deliverable and validation |
| --- | --- |
| 1 — implemented | Separate SSHC module and hand-calculated initialization, assignments, splits, literal/standard objectives, pruning conflict modes, rollback, convergence guards, ties, pure/impure/small/unlabelled-only clusters and invalid-input tests. Validation is recorded in [SSHC_IMPLEMENTATION.md](SSHC_IMPLEMENTATION.md). |
| 2 — implemented core | SSHR cleaning, ranked confidence selection and within-cluster fallback; row conservation/lineage, nonmutation, labels, empty `U`, zero-distance helper, singleton cases, erased classes, reproducibility and unmet quotas. Validation is recorded in [SSHR_RESAMPLING.md](SSHR_RESAMPLING.md). |
| 3 | Full estimator and opt-in runner; train-only feature provenance, hidden-label isolation, identical reduced baselines, categorical rejection, fixed outer folds, persistence/cache identities and final-fit tests. Focused plus full pytest, commit and push. |
| 4 | At most two local numerical datasets, hash and metadata checked first. Same folds/classifiers for the six requested methods; save held-out F1/G-mean/MCC/probability AUC/FPR, pseudo-label accuracy, lineage counts, cluster sizes, shortfalls and training-only overlap proxy. Commit a descriptive report. No 44/74-dataset benchmark. |

Ecoli4 and Yeast4 are candidate numerical inputs listed in the SSHR paper's
Table 2 and supplied locally; their exact files must be hash-verified before the
later study. Duplicate/group handling must be resolved before forming outer
folds. A training-only overlap proxy can count appended rows whose nearest
observed-labelled neighbours favour majority, using the same original reduced
`L` reference for every method, **before** method-specific deletion. Record its
neighbour count and ties; it is not irreducible overlap or hidden-label truth.
Use the same downstream classifier configurations across methods. Paper-SVM
settings, if used, are a separate explicitly recorded configuration; do not
equate repository defaults with the paper's RBF parameterization.

Report all methods, not only the best one. Do not tune from outer-test scores or
retrospective pseudo-label accuracy. A two-dataset comparison cannot establish
general superiority, paper-result fidelity, novelty, or a class-overlap guarantee.

## 10. Historical Stage 0 validation

The pre-existing full test suite was executed at the branch's unchanged base:
**387 passed, 5 failed** in 15.21 seconds. Four failures are missing optional
BART (`stochtree==0.2.1`) and native C5.0 backends; the fifth is
`test_eight_historical_candidates_are_unverified_and_absent_here`, which assumes
raw files are absent even though the user's supplied datasets are present.
These are baseline findings, not SSHR test results. Stage 0 does not delete data,
disable tests, change source/tests, or claim a green full suite. Later stages
must account for these baseline prerequisites and the environment-dependent test.

Stage 0 documentation validation checked source metadata/hash, relative links,
equation counterexample arithmetic, stage status, no-author-code claims, and a
documentation-only diff. No SSHR implementation or real-data training study ran
in that stage. Its audit was committed as
`2b05f0b8d975ccc3e94755f28b960468eb4de0bc`; the user then authorized Stage 1.
Stage 1 resolved the test prerequisites without deleting supplied data and was
committed as `393d319c867c6d5a70b8fbca59b95b1b2ec376de`; the user then authorized
Stage 2. The implementation validation records are separate. **Stop after
Stage 2 and wait for the user's “next” before implementing Stage 3.**
Source-code access/clarification remains
required for any claim of exact author-implementation fidelity.
