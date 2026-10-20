# Bounded synthetic pilot: finalized before execution

This protocol supersedes only the future execution settings in ../PILOT_PROTOCOL.md;
the original audit and its source-access limitations remain preserved. The user has
explicitly authorized a usefulness pilot without a repeated literature search.
Novelty remains unverified. No real-data work is authorized here.

Configuration: configs/research/gmmsampling_overlap_pilot.json. The runner records
its SHA256 and the source commit. This document/configuration must be committed
before any pilot performance is observed. Do not revise the criteria afterward.

## Shared conventions and exact method

Use the previous protocol's synthetic distributions, all 30 conditions/seeds,
partition counts, GMM settings, component development selection, logistic family,
validation selection and evaluator-only identities. Gaussian coordinates are
independent; original rows are preserved in all arms, including majority rows.
This disables author majority undersampling (latest user instruction), and replaces
the earlier proposed majority subset. Each minority still requests 450 additions
to target 550. No classifier sees the evaluator's true subgroup identities.
Class order is [0,1,2]. StandardScaler fits once on original fitting rows only.

Author-source conventions: source bytes and MIT notice are archived alongside
this document, pinned to develop 7a89dd674a7a10d73ac03548a2ee9b48cd04c654.
Responsibilities include learned mixture weights (code, unlike paper's uniform-prior
component-density presentation); weights are sum((1-safe)*responsibility)+1e-8.
If all unsafety is zero the epsilon yields uniform component allocation. Safe scores
use original class-size ratios. Explicit self-ID exclusion and distance/original-row
index tie breaking replace the author's first-neighbor-position deletion.

K selection reproduces last accepted development likelihood: split each original
class 75/25 with the case seed; evaluate K=1..4 in order, stop at first decrease,
retain last accepted K; refit that K on all original class rows. Do not evaluate
an otherwise unused K=5 trial. Gaussian covariances are full with reg_covar=1e-6,
n_init=10, max_iter=100, tol=1e-3. Nonconvergence/singularity is an explicit case
failure. No re-fitting on classifier validation rows. Author-default regularization
is sklearn's same 1e-6; no additional cross-class covariance adjustment.

Largest remainder rounds component budgets, fractional ties by component index.
Exact requested draws replace author's independent floors/minimum d+1 batch
overshoot, shared across A-D. Local numpy Generator streams replace author global
np.random.seed; exact helper/selection agreement is tested separately from random
byte identity. All GMM fits are shared across arms. Original rows stay first.

For generated z of class c, compute log f_c with fitted class GMMs. Equal-prior
minority-only log posterior is log f_c - logsumexp(log f_1,log f_2).
With tau>0 accept iff this >=log(tau). Thus thresholds .55/.70 reject rival-dominated
points plus some ambiguous points. Tau=0 is an exact bypass. Own finite/rival -inf
passes; own -inf/rival finite fails; any NaN/+inf or all -inf fails and counts as
numerical rejection. Every proposal consumes budget, including numerical failures.
No clipping densities, fake uniform posterior or silent rescue. This is a fitted
competition score, not a ground-truth synthetic label correctness indicator.

Hard-MAP original component membership uses argmax responsibility (ties lowest
component index). Store original row indices as parents. Positive quota with no
hard members is unsupported and fails explicitly; no quota reassignment.
Fallback samples those original members uniformly with replacement; it may duplicate
an ambiguous original, which remains in the dataset. Separate keyed fallback streams
make D's first m duplicates identical to C's m fallback duplicates per component.

A: ungated draws, safe k in [5,7,11].
B: k=7, one proposal per requested slot; gate/discard with no refill.
C: k=7, per component at most 20*q proposals, stop immediately on q acceptances;
then duplicate original members for remaining slots. No cross-component refill.
D: k=7, fill every component quota by original-member duplication only.
NORES and balanced-class-weighted logistic regression are required simple controls.
A/B/C have 9 configurations (3 sampler choices x 3 C); D/NORES/WEIGHTED have 3
unique configurations. Total 36 fits/case, 1080 classifier fits; GMM development
and final EM fits/initializations counted separately. No expanded baseline search.
B and C share the first q proposals within a component; all tau values use the same
proposal stream. Tau=0 A(k7)/B/C exact generated-data identity is a required check.

Tune by untouched validation macro AP, then macro F1, then grid order; retain fitted
model without validation refit. Argmax test prediction. Save every candidate's
validation probabilities and selected test probabilities, row IDs, labels, evaluator
subgroups, scaler/GMM parameters, sampling diagnostics/proposals/provenance and
runtime. Validation/test identities never enter a sampler or classifier. Shared
standard-normal prefixes and row-shuffle streams pair conditions as previously
specified. No subgroup-aware selection. No additional test-selected thresholds.

## Predefined advancement decision

Primary predictive measure: mean full three-class macro AP. Retain the earlier
practical thresholds, adapting its replaced global-refill arm to requested B and
requiring D to avoid interpreting C>B's size difference as a quota benefit.
All gates below must pass; no significance or novelty claim from five seeds.

1. All 30 cases and 1080 candidate fits complete; no missing/invalid predictions,
   leakage, budget violations or tau=0 mismatch. B's unfilled quota is intentional.
2. At delta1.5 in BOTH mixtures, C reduces pair confusion vs tuned A by >=.02 and
   wins at least 4/5 seeds.
3. In BOTH partial-overlap cells, each group's mean recall loss vs A <=.02,
   minority-to-majority error increase <=.01, macro F1 loss <=.01. Mean macro AP
   gain across these two cells >=.005 (primary); require the same >=.005 pooled
   partial-cell AP gain against D as a duplication-control safeguard.
4. At partial overlap 90/10, C improves mean worst-group recall vs B >=.03 without
   pair confusion increasing >.01. C>B alone is insufficient because B is smaller;
   criterion3 also requires AP benefit vs A and D.
5. In BOTH weak-overlap cells (delta4), AP/F1 loss vs A <=.01.
6. In EACH partial-overlap cell, pooled fresh accepted Gaussian additions / all
   additions for selected C >=.75 (fallback <=.25).
7. Fail advancement if either simple control in BOTH partial cells has pair
   confusion <= C+.01, worst-group recall >= C-.01 and macro AP >= C-.005.

Delta0 is a negative control with identical minority distributions (pairwise Bayes
error .5); no advancement requirement there. Report all paired seed values, mean,
SD and descriptive mean +/- t(4)*SE intervals. Related conditions share streams;
intervals are descriptive, not independent-dataset inference. Compare C-A/B/D on
AP, F1, confusion and subgroup recall; report real counts, fallbacks and runtime.
If any gate fails, preserve failed advancement and stop this exact bounded pilot.
No post-result parameter/method modifications, real-data runs or larger benchmark.
