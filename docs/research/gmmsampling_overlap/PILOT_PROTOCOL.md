# Proposed bounded protocol — execution disabled

This is a **future design**, not an implemented runner or an executed/funded
experiment. It is proposed before viewing any results for this candidate.
Resolve the source/novelty and baseline issues in [AUDIT.md](AUDIT.md), then
obtain separate implementation authorization. Do not expand this design to a
real-data benchmark automatically. No result is claimed here.

## Prerequisites

1. Read the final SOMM/NDESO texts and JSM MCD abstract; compare their actual
   class competition, component allocation and refill/fallback rules. If an
   exact equivalent is established, do not implement it under a novel name.
2. Freeze a multiclass-only research interface and labels/classes ordering;
   keep existing binary loaders/metrics untouched. No reconstruction of a
   multiclass target from unrelated stored binary variants.
3. Freeze the code-conventions baseline, distinct from a verified paper
   reproduction: fitted GMM responsibilities plus `1e-8` allocation, last
   accepted development-likelihood component count, and explicit all-zero
   unsafety policy. For exact budgets use largest-remainder quota rounding
   and exact per-component draw counts, identically in every GMM arm.
   Document these shared engineering corrections separately from the candidate;
   static/raw-author checks must not be scored as a competing new method.

## Data: isolate minority–minority overlap

Use only one fully specified 2D synthetic family in this first future pilot:
three class labels, two latent components within each minority class.

| Population | Fitting | Validation | Untouched test |
| --- | ---: | ---: | ---: |
| Majority label 0 | 1000 | 1000 | 5000 |
| Minority label 1 | 100 | 100 | 1000 |
| Minority label 2 | 100 | 100 | 1000 |

Majority is Gaussian, mean (-8,0), coordinate SDs (.5,.5). For minority
label 1/2, y-coordinate means are -delta/2 and +delta/2 respectively. Common
component: x mean 0, SDs (.7,.8). Rare component: x mean 3, SDs (.3,.35).
Independent coordinates. Component proportions within **both** minority
classes are either 50/50 or 90/10, in all three partitions.

Delta=4,1.5,0 gives weak, partial and identical minority-class distributions.
The majority distribution, class counts, component locations along x and
covariances remain fixed. Majority overlap is intentionally negligible,
not asserted to be mathematically zero for Gaussian tails. Delta=0 is a
negative control: equal priors and identical minority distributions imply
Bayes pairwise error 1/2 conditional on being one of the minorities. Do not
expect oversampling to recover missing class information.

Seeds 42–46, with independent keyed streams for fitting/validation/test
(stream IDs 100/200/300). Share standard-normal variates and original majority
rows across paired delta settings, using mean shifts only. For component
proportions use shared within-component prefixes; fix row shuffles by stream.
Six conditions x five seeds = **30 cases**, not 30 independent datasets.
No new label-noise or high-dimensional factors in this stage.

True component IDs, analytical densities and Bayes posteriors belong only
to the evaluator. They cannot choose components, validate/tune a GMM, select
tau, choose a classifier, redraw a case, or allocate synthetic samples.

## Shared GMM and candidate rules

Fit StandardScaler on original fitting X only. Within each original minority
class, allocate 25% to GMM component-selection development; choose among
K=1..4 with the declared last-improving rule. Fit the final class GMM on
all original fitting class rows. Full covariance, `reg_covar=1e-6`, n_init=10,
max_iter=100, tol=1e-3, explicit seeded local RNG. Do not use the independent
classifier-validation partition for fitting GMMs. Record nonconvergence,
singularity and insufficient-support failures rather than silently substituting
an unrelated sampler. GMM component IDs are estimated, not true group IDs.

For each case, cache the same two GMM fits across GMM arms. Compute safe
levels/quotas using original fitting data only. Target size is 550 per class
for the fixed 1000/100/100 sizes. All primary GMM arms keep the identical
original minority rows and identical 550 selected original majority rows.
Original-ID tie breaking and self exclusion must be explicit. Do not update
densities or neighbour evidence after generating any class.

Freeze the common majority selection with the original-data k=7 safe scores.
The baseline's k grid below varies only allocation-neighbour evidence in
this controlled comparison. This shared-undersampling convention is an
explicit experimental adaptation, not exact execution of the author API
where one k controls both operations. It prevents changing majority removal
from explaining a difference attributed to the generation gate.

Candidate gate uses `log f_c(z) - logsumexp_j log f_j(z)` over the two
minority classes, with equal priors. Tau=0 means a bypass, reproducing baseline
draws exactly with the same streams. Otherwise redraw from the original
component until its quota is met or 20 attempts per requested row are used.
Duplicate original hard-MAP members of that **same** estimated component
for any remainder. Empty membership for a positive quota is an explicit
unsupported case. Do not move quota to a safer component or count fallback
duplicates as newly accepted Gaussian vectors. A baseline zero quota stays
zero; do not introduce a new subgroup quota formula in this candidate.

## Tuned arms and bounded budget

Use one final multinomial logistic-regression family (L2, max_iter=2000),
C in {.1,1,10}. It supplies multiclass probability vectors without requiring
tree/bootstrap or new-model complications. This is a feasibility probe, not
evidence of classifier-independent superiority.

| Arm | Purpose / sampler tuning |
| --- | --- |
| No resampling | Original-data control; tune classifier C only |
| Balanced class-weighted classifier | Strong simple control; tune C only |
| ROS | Minority duplication to the common target, shared majority subset; tune C |
| Multiclass SMOTE | Same target/subset; k in {3,5,7}, tune C |
| Code-conventions GMMSampling | Main baseline; safe-level k in {5,7,11}, tune C |
| Same cross-minority gate, **global refill** | Mechanism control: fixed k=7, same Q/GMM/target and candidate scores, but rejected slots may be refilled by other components using Q; tau in {0,.55,.70}, tune C |
| Proposed component-quota gate | Fixed k=7; tau in {0,.55,.70}, tune C; within-component redraw/fallback only |
| SOMM | After final-text/source verification; K in {5,7,11}, tune C |
| NDESO | After final-text/source verification; k in {3,5,7}, tune C |

Select on **untouched validation macro-AP over all three classes**, break
ties with validation macro-F1 then grid order. No subgroup-aware tuning or
test-selected thresholds. Freeze argmax class prediction; any later operating
curve analysis must be separately labelled exploratory. Do not refit on the
validation partition after selection; cache selected fitting-only models.

Main candidate/baseline/global-gate pipelines have nine sampler/classifier
candidates each. Controls with no sampler hyperparameter have three unique
classifier candidates, not fictitious duplicated trials. At most nine arms
x nine candidates x thirty cases = **2430 classifier fits**, with lower
actual count from controls/caching. GMM fits/EM initializations are counted
separately; do not describe those as free. No retries or added seeds after
seeing performance. The fixed-k=7 tau=0 models supply an additional cached
paired mechanism check without new fitting.

Primary GMM comparisons have exactly matched output sizes/majority selection.
ROS/SMOTE also use that target. SOMM/NDESO native policies may differ: first
document them from final primary sources and record output counts/original
relocation. Any target-size adaptation must be named as an adaptation and
fixed before execution; native unequal-size results are contextual, not an
isolated test of the new gate. If an arm cannot be verified, pause/refine the
protocol instead of silently replacing it with an invented reconstruction.

## Saved evidence and metrics

- Primary task measure: conditional minority-pair confusion,
  `(n(true=1,pred=2)+n(true=2,pred=1))/(n(true=1)+n(true=2))`.
  Keep minority-to-majority errors separate so sending every ambiguous
  minority row to majority cannot masquerade as success.
- Report full and minority-only macro-AP, macro-F1, balanced accuracy,
  complete row-normalized confusion matrices, each minority's precision/
  recall and majority-to-minority FPR.
- Evaluator-only coverage: recall for each of four true minority subgroups,
  and their minimum. Log fitted-component allocation and accepted/duplicate/
  unfilled counts; these do not certify genuine subgroup coverage.
- Evaluator-only generation diagnostics: class-conditional true
  rival-dominant mass, ambiguous mass (minority posterior in [.4,.6]), and
  soft true-component coverage using analytical conditional responsibilities.
  These are density/lineage diagnostics, not proofs of synthetic mislabelling.
- Save fitting/validation/test IDs, hashes, full validation/test probability
  vectors, selected parameters, proposal/acceptance masks, original and
  synthetic row provenance, fallback reasons, density fits, runtime, peak
  memory, rejection rates and every failure. Validation vectors are required
  after the previous HS analysis exposed their absence as a limitation.
- Pair effects by seed within conditions. Report all five values, mean, SD
  and descriptive paired intervals. Shared conditions/seeds are correlated;
  no independent-dataset significance claim.

## Predefined advancement criteria

Require all of the following; these are practical feasibility cutoffs, not
proofs of novelty, significance or generalization:

1. Full case/candidate accounting with no suppressed failures, label leakage
   or changed output budgets. Tau=0 reproduces the ungated baseline exactly.
2. At partial overlap delta=1.5, **in both component-proportion conditions**,
   reduce mean pair confusion by >=.02 versus tuned GMMSampling, with a
   reduction in at least four of five seeds.
3. In those cells, no true subgroup's mean recall loses >.02 versus the
   baseline; minority-to-majority error cannot increase >.01. Mean full
   macro-AP gains >=.005 across the two cells and macro-F1 loses <=.01.
4. Versus the **same gate with global refill**, improve mean worst-subgroup
   recall by >=.03 in the 90/10 partial-overlap cell, without increasing pair
   confusion >.01. Otherwise quotas have not shown a distinct benefit over
   generic rival-density filtering.
5. In weak-overlap controls, AP/F1 lose <=.01 versus baseline. Delta=0 is
   reported as a negative control; do not require or advertise separation
   beyond the information limit.
6. In partial-overlap cells, at least 75% of the added rows must be newly
   accepted Gaussian draws, not fallback duplicates. Otherwise the candidate
   is largely a constrained ROS policy and must be described/refined as such.
7. Stop if NORES or class weighting covers both partial-overlap conditions
   within .01 pair confusion, .01 worst-subgroup recall and .005 macro-AP of
   the candidate simultaneously. Inspect verified SOMM/NDESO controls too;
   beating an untuned baseline alone is insufficient.

If unmet, retain the negative result and stop this exact candidate. If met,
recommend a separately authorized independent validation/real-data stage;
do not launch it automatically. This audit neither runs nor authorizes that
first implementation stage.
