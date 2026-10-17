# Question B: subgroup representation and performance

Assessment date: 2026-10-17. Inspected source revision:
`3e5b713c773aa3dbaec8f7224d9cb85b9947b960`.

**Status:** repository and source assessment completed; the intended extension
has not been specified or selected. No new sampler, comparative subgroup pilot,
or methodological novelty is claimed. This document concerns Question B only.
It preserves the original implementations and historical experiment artifacts.

## 1. What already exists

| Component | Verified behavior and reuse |
| --- | --- |
| [UGO](../../src/ugo.py), [uncertainty](../../src/uncertainty.py), [stopping](../../src/stopping.py) | Clone/retrain the selected probability classifier on augmented fitting data; score original eligible rows; select a class using mean signed uncertainty; select its highest-scoring eligible seeds; generate a fixed batch; apply neighbor-based correction. ROS, SMOTE, RWO and MCSMOTE variants exist. |
| [Generators](../../src/generators.py), [noise handling](../../src/noise_detection.py) | Reusable original-row generation and deterministic neighbor utilities. UGO records eligible seed indices, drawn seed indices and, for SMOTE, neighbor indices/interpolation coefficients. These allow evaluator-only lineage accounting. Relocation destinations must also be tracked. |
| [EC-EBO clustering](../../src/ec_ebo_clustering.py), [allocation](../../src/ec_ebo_entropy.py), [sampler](../../src/ec_ebo.py) | Existing separate Question B comparator: externally constrained minority clustering, mean logistic-probability entropy, exact integer quotas and coordinate generation with explicit shortfalls. Reuse utilities only if the intended proposal adopts those definitions. Do not silently equate it with SLEO. |
| [Other comparators](../../src/comparators.py) | NORES, RUS, ROS, SMOTE, RWO, MCSMOTE, LEAO, BFCH-SMOTE and EC-EBO constructors. SSHR has a separate [controlled interface](../../src/sshr_evaluation.py). |
| [Classifiers](../../src/classifier_registry.py) | BART, BNNET, C5.0, CART, GNB, GKNB, KNN, LDA, LogReg, NNET, QDA, RF, SVMLinear and SVMRadial. Interfaces work, but the registry explicitly labels author-package/settings equivalence as approximate. |
| [Preprocessing](../../src/preprocessing.py), [evaluation](../../src/evaluation.py), [results](../../src/results.py) | Fit preprocessing inside each training partition, fit a fresh final classifier after resampling, record folds/configuration/software identity, retain failed/unavailable jobs and resume matching runs. |
| [Simulations](../../src/simulations.py), [pilot](../../src/pilot.py) | Existing linear/nonlinear UGO reconstructions retain pre-noise labels. They do not expose independent minority subgroup identities or implement the requested 70/25/5 held-out study. |
| Existing studies | [UGO simulation pilot](../reproduction/SIMULATION_PILOT.md), [BFCH study](../BFCH_SMALL_STUDY.md), [EC-EBO study](../EC_EBO_SMALL_STUDY.md), [SSHR study](../SSHR_SMALL_STUDY.md). These are historical evidence, not new Question B results. |

The [dataset collection](../reproduction/DATASET_COLLECTION.md) records 74
historical approximate inputs, including 18 accepted approximate variants. The
small real-data studies concern Winequality/Vowel and Ecoli4/Yeast4. Current
`data/raw/` contains only `.gitkeep`: source manifests are available, raw inputs
and individual historical job caches are not. No real-data pilot was rerun.
SSHR code is present in this checkout after the merge, despite older branch
guidance in the README and scope document.

The [metric module](../../src/metrics.py) provides IAM, MCC, F1, G-mean and
probability ROC-AUC. The SSHR evaluator additionally records FPR. Neither runner
currently provides true minority-subgroup recall, macro/worst subgroup recall,
average precision, or a threshold-tuning/class-weighting comparison arm.
Balanced accuracy and FPR are derivable from stored confusion counts; AP needs
held-out scores and labels. KMeansSMOTE is installed through imbalanced-learn,
but is not registered in these runners.

## 2. The missing proposal and historical candidates

The current [scope](../PROJECT_SCOPE.md) marks RB-UGO/CSG inactive. There is no
active proposal document or partially implemented subgroup extension in `src/`.
The user's new request authorizes extension work once its mechanism is known;
the earlier inactive status is not an approval barrier.

History contains [two DRA-UGO candidates at 67e296c](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/67e296c63c24606039affa051d05f45bcd5b0f1d/docs/research/CANDIDATE_HYPOTHESES.md):

1. **Control-adjusted local augmentation contrasts:** propose a finite family
   of regional fitting-only interventions; choose thresholds on a separate H
   set; audit balanced-risk/ranking/harm contrasts on independent A; allow
   inconclusive outcomes. An ordinary within-region interpolation arm is
   described, but regions, practical margins, budgets and a final selection
   rule still require decisions. It is a diagnostic/selection framework, not a
   specified iterative replacement for UGO.
2. **Anchor-compatible graph sensitivity intervals:** bound a graph consistency
   potential over fields with fixed observed anchors and an energy constraint.
   It is a proposed structural veto, optionally using unlabelled features, not
   a subgroup need/allocation/stopping algorithm. It does not certify labels.

The historical [novelty decision](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/67e296c63c24606039affa051d05f45bcd5b0f1d/docs/research/NOVELTY_DECISION.md)
establishes neither candidate as a novel method. These records have not been
reactivated, combined, or silently chosen on the user's behalf.

| Required extension decision | Current status |
| --- | --- |
| Minority subgroup definition; fixed or recomputed groups | Not selected. Learned clusters and evaluation-only true groups must remain distinct. |
| Need score and its estimation data | Not selected. Count, entropy, MMA and independent intervention benefit are different quantities. |
| Allocation/normalization and integer rounding | Not selected. Existing EC-EBO's mean-entropy/largest-remainder policy is available, not assumed. |
| Generator and allowed anchors/partners | Not selected. Current UGO-SMOTE allows partners across minority subgroups. |
| Noise, tiny clusters, no eligible seeds, zero total need | Not selected. Filtering every tiny group would prejudge the research question. |
| Budget, class selection and stopping | Not selected. Retaining or replacing UGO's Welch/class-selection rules changes the method materially. |

The necessary clarification is the exact proposal text/path or intended
historical candidate/version. A new method cannot be faithfully implemented
by filling this table with convenient defaults. A fundamental problem in a
particular proposal cannot be diagnosed until its rules are identified.

## 3. Existing UGO mathematics and its subgroup limitation

For numerical labelled fitting data F with n original rows and binary labels,
let E be the original rows retained by the k-neighbor pre-filter. At iteration
t, fit a fresh classifier on the current augmented data and calculate

\[
u_i^{(t)}=P_t(Y\ne y_i\mid x_i)-P_t(Y=y_i\mid x_i)
         =1-2P_t(y_i\mid x_i),\qquad i\in E.
\]

This is signed label disagreement in [-1,1], not a calibrated decomposition of
epistemic/aleatoric uncertainty. Both original noisy and clean rows remain in
classifier fitting data; pre-filtered rows cannot be seeds. With eligible class
sets E_c, compute \(\bar u_c=|E_c|^{-1}\sum_{i\in E_c}u_i\). A defined
two-sided Welch test stops on \(p\ge\alpha\); otherwise choose
\(c^*=\arg\max_c\bar u_c\), which can be the majority class. Undefined tests
terminate with a separate engineering reason, not evidence of equality.

Choose the top \(\max(1,\lceil\lambda_{select}|E_{c^*}|\rceil)\) scores,
with stable original-row tie order. The nominal batch is
\(b=\max(1,\lceil\lambda_{step}n\rceil)\). The default total addition cap is

\[
B=\max(0,\lceil2\phi n_{max}-n\rceil),
\]

with an explicit alternate Table-4 interpretation
\(B=\max(0,\lceil\phi n_{max}\rceil)\). These conflicting paper expressions
remain declared interpretations; the batch is clipped to remaining budget.
UGO-SMOTE draws a selected seed i, an original pre-clean same-class neighbor j,
and \(\lambda\sim U[0,1)\), then forms
\(z=(1-\lambda)x_i+\lambda x_j\). Insufficient same-class neighbors terminate
generation. A generated point failing the post-vote check is replaced by a
nearest original eligible same-class row. The output is copied originals plus
added rows, diagnostics and the last scoring model; evaluation must fit a
fresh downstream model. See [the fidelity decisions](../ugo_reproduction.md).

For an eligible minority partition \(G_1,\ldots,G_K\),

\[
\bar u_{minority}=\sum_g {|G_g|\over|E_{minority}|}\bar u_g.
\]

Thus a 70/25/5 partition with means -0.9, -0.4 and +0.8 has overall mean -0.69.
If the majority mean is -0.6 and generation proceeds, class selection favors
the majority despite the rare minority group's high disagreement. This is an
algebraic construction, not an observed classifier trajectory or test-performance
result. Welch non-rejection also establishes no subgroup recall guarantee.

For UGO-SMOTE, the present dense distance implementation costs
\(O(n^2d+n^2\log n)\) for pre-filter distances and full row sorting. Across
iterations add classifier fitting/scoring, same-class neighbor matrices
\(O(n_c^2d+n_c^2\log n_c)\), and batch-to-reference correction distances/sorting
\(O(b_tnd+b_tn\log n)\). Distance storage is quadratic in the relevant
reference sizes. Classifier and other-generator costs differ. The extension's
cost, inputs/outputs, equations and small-cluster safeguards remain undefined
until its mechanism is selected.

## 4. Direct evidence of a possible seed-exclusion failure

The following fitting-only artificial diagnostic was executed with seed 42.
Minority counts are exactly 42/15/3 (70/25/5); no labels are flipped. Three rare
points lie inside a six-majority-point ring. After the existing fold preprocessor
and UGO's default five-neighbor vote, eligible counts are **42/15/0**. Even if
the rare points received the highest MMA scores, none could become a UGO seed.
This diagnoses a mechanism to investigate, not its frequency or downstream harm.
Subgroup identities are used only by the diagnostic evaluator.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python - <<'PY'
import numpy as np
from src.noise_detection import noise_pre_detect
from src.preprocessing import FoldPreprocessor

rng = np.random.default_rng(42)
minority = np.vstack([
    rng.normal((-2, 2), .08, (42, 2)),
    rng.normal((2, 2), .08, (15, 2)),
    [[0., -.02], [0., 0.], [0., .02]],
])
angles = np.arange(6) * np.pi / 3
majority = np.vstack([
    rng.normal((0, -3), .12, (174, 2)),
    .08 * np.c_[np.cos(angles), np.sin(angles)],
])
X = np.vstack([majority, minority])
y = np.r_[np.zeros(180, dtype=int), np.ones(60, dtype=int)]
groups = np.r_[np.full(180, -1), np.repeat([0, 1, 2], [42, 15, 3])]
fitting = FoldPreprocessor().fit_transform(X, y)
excluded = noise_pre_detect(fitting, y, k_pre=5)
counts = [int(np.sum((groups == g) & ~excluded)) for g in range(3)]
print(counts)  # [42, 15, 0]
assert counts == [42, 15, 0]
PY
```

## 5. Primary-source assessment and novelty limits

The six attached PDFs were parsed; methodology passages relevant below were
inspected. Page numbers are PDF pages. Source authors' experimental outcomes
are not reproduced by this audit, and author code was not verified.

| Source | Inspected evidence and consequence for Question B |
| --- | --- |
| [UGO](https://doi.org/10.1016/j.knosys.2026.115886), 20 pages | pp.5–8: original labelled pools, MMA, class-level selection/stopping and neighbor noise handling. It has local filtering and instance selection, but no explicit subgroup allocation guarantee. |
| [SLEO](https://doi.org/10.1016/j.knosys.2025.115009), 18 pages | pp.4–7: constrained minority clustering, cluster entropy, majority relocation and cluster-weighted generation. Eq.7 already allocates in proportion to cluster entropy. Algorithm 1 does not take a separate unlabelled feature pool despite the name. |
| [SSHR](https://doi.org/10.1016/j.eswa.2023.119733), 13 pages | pp.4–6: labelled/unlabelled clustering, majority removal, minority pseudo-label selection and fallback SMOTE. These are separate mechanisms; adding them would move beyond a purely supervised Question B extension. |
| [GDEO](https://doi.org/10.1016/j.neucom.2026.134121), 12 pages | pp.4–6: Gini probability weights, weighted seeds and gradient-based generation. Uncertainty-weighted seeds alone do not establish a distinct contribution. A verbal no-crossing claim is not a verified posterior constraint. |
| [Hamilton, Don't oversample the boundary](https://doi.org/10.1016/j.neucom.2026.134831), 17 pages | pp.4–6 and 8–9: remedy decomposition, matched category-targeted interventions and threshold parity. These directly motivate threshold/weight/count controls and limit claims that measuring intervention response is itself new. |
| [Ma & Zhang, arXiv:2601.16120v3](https://arxiv.org/abs/2601.16120v3), 90 pages | pp.9–12 and 21–22: class-proportion/generator mismatch and validation-tuned synthetic size. Size search is existing methodology; the theory has stated statistical assumptions and is not a universal guarantee for UGO, SMOTE or RF. Supplementary proofs were not exhaustively checked. |

SLEO needs special caution before any implementation adaptation. Eq.3 prints
\(p\log p+(1-p)\log(1-p)\) without the usual entropy minus sign; this was
also checked visually in the supplied PDF. Its nonpositive value conflicts
with Algorithm 1's positive-entropy cleaning branch. Even after a sign change,
a positive entropy times exponential decay does not reach zero at finite
radius without a tolerance. Eq.4 sums member values, so equal mean difficulty
does not remove size effects. Resolving these issues would be an explicit
interpretation, not exact author-code reproduction or a newly proposed method.

For KMeansSMOTE, the [pinned 0.12.4 implementation](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/0.12.4/imblearn/over_sampling/_smote/cluster.py)
was inspected: eligible-cluster sparsity is mean pairwise distance raised to an
exponent, divided by minority count; normalized sparsities determine allocations.
Per-cluster ceiling can exceed the requested total. This verifies library
behavior, not every equation in the [original paper](https://arxiv.org/abs/1711.00837),
whose HTML/PDF requests both returned HTTP 403 in this environment.

A-SUWO's primary article
([DOI](https://doi.org/10.1016/j.eswa.2015.10.031)) is cited in the supplied
papers, but its full methodology was not obtained here. AROSS/PPEO are not
identified by an exact primary citation in the supplied materials inspected.
Their user-reported mechanisms remain prior-art leads, not newly verified
source findings. They are absent from the runnable comparator registry, as
are SLEO and GDEO. These gaps prevent a comprehensive novelty/superiority claim.
No mechanism is classified as a distinct new contribution while the proposal
itself is unspecified; combining the existing ingredients is insufficient.

### Supplied-file provenance (SHA-256)

```text
UGO    87cbd8f4237a19fd68e34cd8d57e664d6bc47adc0cb8e8ce9eedc1da5c10bc8b
SLEO   2a98334c90e86472d3a53e992bc5537870bfe32306f9a5f55a6cf6f7849305e4
SSHR   89076411fe1dafad46efb1c5dc4fd3cfaa8ee720ed3a91c29210970370318910
GDEO   c7dda7d02022b29aba7a4289a3da13a0df35a3d6955d2c4b44a20263728c1be5
Hamilton c9e8df9d65783041fa325bff3c08d28449fda3830d346483db52b56621d98f5c
Ma-Zhang 41ddb9ceaa56f51840a733f9571eb282d089f2ae2b34e0d6af2c8b4bf1ee18af
```

Uploads and long text extracts stay outside Git. EC-EBO/BFCH assessments above
rely on current code and existing repository method audits; their primary PDFs
are not among these six attachments.

## 6. Comparison safeguards and the pending experiment

The inspected standard evaluation path fits preprocessing on training rows,
passes only those rows to resampling, and refits a fresh final classifier. These
are reusable safeguards. No new leakage bug in that path was established by
this inspection. The following changes are prerequisites for the requested
comparison, not evidence that the existing historical studies are invalid:

- Add inner training/selection boundaries for thresholds, hyperparameters and
  any learned intervention policy. Refit preprocessing and resampling inside
  each inner fit; fitting them on the full outer-training set before inner
  validation leaks selection data. Outer test data remain untouched.
- Keep NORES at its original decision rule and add threshold-tuned NORES,
  class weighting, ROS, SMOTE, KMeansSMOTE, original UGO and existing close
  comparators. Give every method the same threshold-selection opportunity and
  comparable search/compute budgets; record the chosen objective and failures.
- Freeze true simulated subgroup identities independently of fitted clusters.
  Hide those identities from the sampler unless the proposal requires them.
  Generate independent test observations; vary subgroup size and overlap/noise
  separately, so scarcity is not automatically equated with difficulty.
- Use group/time-aware supplied folds where needed. The generic ten-fold
  splitter does not itself keep duplicate/entity groups intact; the dedicated
  SSHR small-study runner already contains group-preserving design utilities.
- Current UGO can generate majority rows and can exceed a class-balance gap;
  other methods can delete rows or miss quotas. Record requested and realized
  counts by class. Count-matched controls must specify what they match, rather
  than treating all resampled training sets as equal interventions.
- Record rare-mode eligibility before/after noise handling, selected and drawn
  seeds, interpolation partners, relocation destinations, actual additions,
  deletions, shortfalls and time. Seed lineage is not the true semantic subgroup
  of an interpolated or relocated point. Separate known injected-label
  corruption from a nearest-majority intrusion proxy.

For true test minority group g, use \(R_g=TP_g/(TP_g+FN_g)\), macro recall
\(K^{-1}\sum_g R_g\), and worst recall \(\min_g R_g\). Also report global AP
from scores (not ROC-AUC or trapezoidal PR-AUC), F1, balanced accuracy
\((TPR+TNR)/2\), and FPR. A group absent from a test partition is undefined;
report its denominator rather than imputing a favorable score.

Run grouping/allocation/stopping ablations only for components the actual
proposal changes. Include realized-count-matched ordinary generation and
duplication/weight controls. For repeated independent simulation seeds, pair
method differences by seed and form intervals at that independent replicate
level. Correlated CV folds and repeated fits on the same test observations
cannot be counted as independent evidence. A factorial interaction needs a
contrast such as \(M_{11}-M_{10}-M_{01}+M_{00}\), with uncertainty; beating the
two single components does not demonstrate interaction.

This is an experiment specification, not an implemented runner or a completed
subgroup pilot. Its method-specific arms/parameters cannot be frozen before
the proposal is known. No superiority, equal-recall, or noise-safety result is
inferred from the eligibility diagnostic or existing global study metrics.

## 7. Checks completed and commands

Executed in the prepared Python 3.11.17 environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/workspace/.cache/matplotlib \
  .venv/bin/python -m pytest -q

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/workspace/.cache/matplotlib \
  .venv/bin/python -m scripts.run_sshr_study --software-check \
  --output results/question-b-baseline-software-check
```

The full suite completed with **738 passed**. The existing six-method artificial
software check completed **12 successful jobs**, two per method, with zero
failed/unavailable jobs. It verifies existing execution paths, not the requested
Question B contribution. The 70/25/5 eligibility diagnostic above also passed.
No new algorithm, runner, dependency, test or frozen study artifact was changed.
This assessment document is the only intended repository change.

No executable extension/pilot command is supplied because those components do
not exist. Once the exact proposal is provided, reuse the interfaces above,
resolve its mathematical gaps explicitly, implement it separately with failure
tests, and run the paired held-out subgroup study without altering UGO.
