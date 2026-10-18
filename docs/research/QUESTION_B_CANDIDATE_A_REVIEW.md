# Candidate A review: insufficient for the new-method objective

2026-10-18; reviewed `a42780f` and the **complete**
[candidate specification](QUESTION_B_EXTENSION_DESIGN.md), including its grouping,
bounds, calibration, fallback, expanded catalog and pooled-training controls.

**Classification for this project's goal: insufficiently distinct or feasible.**
A is a coherent adaptation of established robust selection. No substantive new
statistical bound, discovery mechanism or optimization principle is established.
Its count requirement is also incompatible with the executed pilot's labelled
development allocation when the regions separately represent the rare subtype.
Withdraw the recommendation to implement A as the project's new method. It may
remain an empirical comparison, subject to an explicit adequate data budget.
This review supersedes the earlier prototype recommendation; all historical
designs, results and the closed filter candidate remain unchanged.

## 1. Actual differences from the closest verified methods

The comparison below concerns verified implementation behavior, not an assertion
that all GEORGE configurations are identical or that A reproduces GEORGE's full
DRO training procedure.

| Component | Candidate A | Closest verified precedent and assessment |
| --- | --- | --- |
| Grouping | Standardized original fitting positives; KMeans K=2…min(8,floor(n_positive/3)), 20 initializations, minimum size3, silhouette≥.25; 20 refits on80% subsamples with mean matched Jaccard≥.8. | GEORGE clusters within coarse classes using KMeans/GMM and supports silhouette-based K search. Its AutoK score is the **unweighted mean of per-cluster silhouettes**. A does not specify row-average versus cluster-average silhouette; that ambiguity is not novelty. Raw features, initialization count and thresholds are engineering choices. |
| Access and group reliability | Partition fitted on F only; later rows assigned by nearest centroid. | GEORGE's basic clustering fits training activations and predicts later assignments. Its optional overclustering also uses validation activations for size checks and training losses to retain difficult subclusters. A's stricter F-only grouping supports its independent-selection-sample assumptions; sample splitting is established statistical practice. |
| Selection objective | Maximize minimum positive-region recall lower bound. | GEORGE explicitly supports worst inferred-subclass validation accuracy and AUROC checkpoint selection. Group DRO saves by minimum validation-group accuracy. Within a positive-only region, classification accuracy **is recall**. Restricting the utility to positive regions at an FPR constraint is a task adaptation. |
| Confidence | Bonferroni-adjusted one-sided Clopper–Pearson bounds; maximize their minimum. | This exact criterion was not found in the inspected GEORGE/Group-DRO code. It is nevertheless standard pessimistic selection using classical bounds. Group DRO also implements a size-dependent loss adjustment, which is **not** the same CP rule. A provides no new bound or improvement/selection-regret guarantee. |
| Calibration | Independent negative order statistic with a binomial tail and family correction. | Standard tolerance-bound construction. It differs from the inspected GEORGE checkpoint metric, but is shared operating-point control rather than a new subgroup-learning mechanism. Its guarantee concerns FPR, not semantic-group recall or achieved FPR exactly5%. |
| Fallback | Any unstable partition or S group smaller than20 returns global-BA selection. | GEORGE's overclustering already guards unreliable groups: default absolute train/validation size threshold25, silhouette/loss tests, and merging rejected subclusters. A's whole-selector fallback differs operationally, but abstention does not establish learning or methodological novelty. |

**The precise remaining decision difference is marginal-bound pessimism under
unequal group sample sizes.** With equal group sizes n and common beta, the CP
function b_n(a) is increasing, so

\[
\arg\max_m\min_j b_n(a_{mj})
=\arg\max_m b_n(\min_j a_{mj})
=\arg\max_m\min_j\widehat R_{mj}.
\]

Thus confidence and empirical worst-group selection have identical rankings
and, with common ties, identical decisions. They also coincide if one fixed
group determines both minima for every candidate. Unequal sizes can change
which group binds; the rule uses marginal success counts, not paired
between-model error covariance. Simultaneous coverage is not a theorem that
its chosen model has better worst-group prediction.

For example, with group sizes (400,100,20), success counts (160,40,10) versus
(120,50,14) give empirical minima .40 versus .30. For M=8, J=3,
delta_R=.025, CP minima are .17836 versus .23204, reversing the choice. This is
arithmetic illustrating the decision rule, **not evidence of improved true risk**.
The previous A/B example in the design made both robust selectors choose the
same pipeline; it did not isolate a confidence-bound benefit.

## 2. Label requirements and fallback probability

There is no approved real-application label budget in the repository. The
executed allocation was **F600 + V2000 =2,600 development labels**. A's proposed
F1200+C6000+S6000 uses 13,200, **5.08 times** that allocation; it was a proposed
scenario, not demonstrated data availability. Test labels are not selection data.

For an informative count calculation, assume perfect alignment between three
inferred regions and true subtypes, and successful clustering. At prevalence
.10 and conditional weights (.70,.25,.05), a rare-subtype positive is .005 of
all labelled rows. Under A's fixed-class-count, IID-within-class design, let
n=N_S/10. Then (N_0,N_1,N_2) is multinomial(n;.70,.25,.05), and the exact
probability of passing **all three** size gates is

\[
p_n=\sum_{r=20}^{n-40}\mathrm{BinPMF}(r;n,.05)
\left[F_{\mathrm{Bin}(n-r,\, .25/.95)}(n-r-20)
-F_{\mathrm{Bin}(n-r,\, .25/.95)}(19)\right].
\]

It is zero for n<60. These are deterministic probability calculations, checked
also by direct multinomial summation; no data were sampled or models fitted.

| Labelled selection rows N_S | Expected rare positives | All three counts≥20 | Count-gate fallback |
| ---: | ---: | ---: | ---: |
| 1,000 | 5 | 0.00000674% | 99.99999326% |
| 2,000 | 10 | 0.26646% | 99.73354% |
| 3,000 | 15 | 11.90469% | 88.09531% |
| 4,000 | 20 | 53.20301% | 46.79699% |
| 5,000 | 25 | 87.27655% | 12.72345% |
| 6,000 | 30 | 98.05467% | 1.94533% |

Even the optimistic use of all 2,000 validation rows as S gives 99.73% fallback,
before finding independent calibration data. Keeping the development budget
and splitting V into C1000/S1000 gives 99.99999326% count fallback. In the
**archived pilot itself**, V contained exactly ten rare positives, so any split
has fewer than 20: an aligned rare region fails deterministically. Independently
of semantic alignment, 100 S positives cannot satisfy the gate if the fitted
partition has six or more groups.

The first selection sizes, in multiples of ten, achieving 80/90/95/99% count-pass
probability are **4,710/5,150/5,540/6,310** labelled S rows, respectively.
These exclude fitting/calibration labels. Expected count20 is not high-probability
availability: N_S=4,000 passes only 53.2% of the time.

Training support further constrains aligned grouping. Under the proposed IID
mixture sampling, the probability that every fitting subtype has at least three
positives is 58.26% for F600 and 94.25% for F1200. Combining independent F/S count
conditions at the proposed F1200/S6000 gives 92.42% availability, before silhouette
or stability checks. The archived F600 instead had fixed counts42/15/3; that
58.26% calculation is not a claim about its already-realized fitting sample.

**Actual inferred-region fallback remains unmeasured.** Conditional region
masses depend on F; fragmentation can make counts smaller, while merging the
rare subtype can make the gate pass without separately monitoring it. Neither
passing the gate nor merging a group proves successful or failed rare learning.
The table quantifies the intended distinct-subtype setting, not a universal
lower bound on fallback for arbitrary learned partitions. Natural cohorts with
random class counts require the corresponding unconditional calculation.
For actual inferred groups, replace the three weights by their unknown
conditional masses q_j(F) and include the F-only stability event; the stated
class/subtype prevalences alone cannot determine that full fallback rate.

Twenty examples also do not ensure precise selection. With observed recall
3/30=.10, the CP lower bound is .00301 for M80 and .00665 for M8 (J3,
delta_R=.025). Even20/20 gives only .63225/.70939, respectively. Count eligibility
does not establish power for five-point differences. No larger label budget,
oracle enrichment, duplicated independent observations or changed minimum is
assumed to make A active.

## 3. A smaller catalog is sufficient

**The 80/89 counts are not necessary for the central hypothesis.** They came
from48 sampler/capacity combinations,32 matched duplication/weight controls,
one original reference and eight pooled-training references. That is a broad
method study, not a prerequisite for comparing selectors. It adds fitting and
adapter burden and makes the family-adjusted bounds more conservative.

A minimal controlled initial catalog can use **M=H=8**, all with the same
RF100, leaf1, n_jobs1 and fitting-only preprocessing:

- NORES; balanced class weights; ROS; audited EC-EBO.
- Four unchanged UGO-SMOTE configurations with k_pre=[5,3,1,7], phi1 and the
  existing remaining diagnostic parameters.

For F600, controlled UGO caps and ROS/EC requested additions are480; achieved
counts/shortfalls stay recorded. All eight fitted predictions are shared by
every selector. This retains ordinary controls, existing UGO tuning choices and
an implemented subgroup-aware comparator. No classifier grid, extra adapters,
count-matched copies, pooled-data fits or separate calibration-only predictors
are needed to isolate confidence handling. This initial catalog cannot support
a claim of general method superiority, sample efficiency or a new generator.

For20 fresh paired seeds142–161 and the existing two regimes, the optional
component study would require320 final fits and160 UGO trajectories, rather
than3,560 final fits and680 trajectories. **It is not recommended for execution
at the current label allocation**, because the activity calculation already fails.

## 4. Decisive comparison, conditional on adequate existing data

Keep F600/C1000/S1000/T10000 as the budget reference; do not silently replace it
with the larger scenario. A usable active comparison would require a separately
declared adequate label budget. Without that, there is no well-supported
efficacy experiment for unchanged A at the intended distinct rare-group scale.

The correct three-way comparison, if feasible, is:

\[
G=\arg\max_m\widehat{BA}_{m,S},\qquad
W=\arg\max_m\min_j\widehat R_{mj,S},\qquad
L=\arg\max_m\min_jL^{CP}_{mj,S}.
\]

Use **one identical F-only partition, eight fitted models, C-derived thresholds,
S rows, tie order and tuning budget**. W and L must have the same grouping,
stability and minimum-count gate; both return G on failure. Clarify A's silhouette
aggregation before any execution (standard row-average is an explicit engineering
choice). True subtype identities enter only the evaluator. Do not refit models
or change gates, K, confidence levels or catalog after inspecting S/T outcomes.

Use alpha=.05, delta_F=delta_R=.025, H8 for the common calibration and beta=.025/(8J)
for L. With900 calibration negatives, the order-statistic rule permits27
exceedances; it is conservative, not exact achieved5% FPR. All three selectors
receive that same calibration, so calibration cannot explain L−W.

**Primary component contrast: L−W**, mean paired test worst-true-subgroup recall
in overlap/RF100. W−G measures the objective change; L−G measures the combined
policy effect and **cannot establish the confidence component's contribution**.
Report selection disagreement and which group binds each criterion. Equal-size
groups provide an algebraic identity check, not a trained experimental arm.

Retain all seeds/fallbacks. For a practical-benefit claim require a lower95%
paired-seed interval above **+.05** for L−W, at least16/20 active replicates,
and no worsening relative to G in worst-group recall (lower paired limit≥0).
Against W and G, require upper FPR-increase limit≤.005, lower AP-difference
limit≥−.005 and upper population-error-increase limit≤.002; also achieved test
FPR≤.06 per primary replicate and upper mean-FPR limit≤.05. Freeze the same
10,000 paired-seed bootstrap resamples/RNG42 for these descriptive simulation
intervals. No secondary superiority claim follows without multiplicity control.

If the upper primary interval is below+.05, reject the claimed practical benefit;
if the interval straddles the margin, evidence is inconclusive. Frequent fallback
rejects applicability at that budget, not efficacy conditional on activity.
L≈W with W better than G supports objective alignment only; a positive L−W
supports an empirical confidence-selection benefit, **not novelty by itself**.
Shared fitted predictions isolate that policy component from different search,
training, grouping or calibration. Generated counts can change with the selected
model, so there is still no new-sampling-mechanism claim.

## 5. Decision and verification limits

Do not implement A as the project's proposed new methodological extension.
Its defensible status is an established-method adaptation, currently unsupported
at the demonstrated data allocation. A possible empirical study is a different
claim and must not be used to satisfy novelty by adding more components.

Freshly inspected author code, pinned at the revisions cited in the original
design:

- GEORGE [cluster search/overclustering](https://github.com/HazyResearch/hidden-stratification/blob/fc9c9d05d8f96fbfc1badb546a812d6d893c1dc3/stratification/cluster/models/cluster.py#L78),
  especially lines104–124 and179–210; [group fit/assignment](https://github.com/HazyResearch/hidden-stratification/blob/fc9c9d05d8f96fbfc1badb546a812d6d893c1dc3/stratification/cluster/george_cluster.py#L87),
  and [worst-group metric/checkpoint](https://github.com/HazyResearch/hidden-stratification/blob/fc9c9d05d8f96fbfc1badb546a812d6d893c1dc3/stratification/classification/george_classification.py#L580).
- Group DRO [worst-group validation selection](https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/train.py#L205)
  and [size-adjusted robust loss](https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/loss.py#L62).

Author README/paper identities were checked; arXiv full-PDF requests again
returned proxy HTTP403. This is an implementation comparison with explicit
paper-access limits, not an exhaustive novelty proof. Raw sources remain outside
Git. Probability values were checked using binomial conditioning and independent
direct multinomial sums; no sampler, benchmark, model replay or Monte Carlo
experiment was implemented or run. Historical source, protocols and evidence
are preserved.
