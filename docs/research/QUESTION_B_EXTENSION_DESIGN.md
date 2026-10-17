# Supervised Question B: subgroup-sensitive selection as a bounded extension

2026-10-18. Reviewed baseline: `c857d43`. **Design only; no sampler implemented
or benchmark launched.** This stage reopens extension design under the user's
new request; it preserves the [filter-candidate closure](QUESTION_B_DECISION_MEMO.md),
[complete pilot](QUESTION_B_DIAGNOSTIC_RESULTS.md), and frozen evidence.

**Recommend candidate A as a repository extension and falsifiable research
hypothesis: an adaptation of established robust model selection, not a novel
oversampler.** No candidate currently establishes a globally new mechanism.
If global algorithmic novelty is mandatory, selection remains blocked by the
direct GEORGE precedent and missing evidence below; renaming this adaptation
would not resolve that problem.

Contribution sentence: **Select unchanged supervised resampling pipelines by
simultaneous lower bounds on recall across fitting-inferred minority regions,
under a common global-FPR calibration procedure.** Whether this helps meaningful
rare subtypes beyond strong existing controls remains to be tested.

## 1. Opportunities, evidence and alternatives

| Narrow limitation / candidate | What existing UGO does | Evidence and unresolved consequence | Why existing methods might suffice |
| --- | --- | --- | --- |
| **A. Selection utility can disagree with subgroup sensitivity.** | UGO uses class-mean MMA and Welch internally; the diagnostic runner selects configurations by global validation balanced accuracy. | **Experiment:** overlap RF capacity tuning raised AP .33303→.34180 while worst recall fell .050→.006; paired difference −4.40 pp, exploratory 95% interval [−6.20,−2.60]. **Mathematics:** prevalence-weighted recall can prefer sacrificing a small group. This demonstrates a trade-off, not that inferred-group selection fixes it. | GEORGE/Group DRO already use robust group selection. The same criterion applied to ROS, weighting or EC-EBO may suffice. **Supported for a bounded adaptation.** |
| **B. In-sample MMA may conceal generalization error.** | Each iteration fits augmented F, then scores eligible original fitting rows; MMA is 1−2P(true). | **Mathematics:** an interpolating learner can give all originals MMA=−1 while failing unseen observations. **No pilot attribution** establishes consequential harm from this optimism. | Honest/CV difficulty is established; A-SUWO already uses held-out cluster errors. Full fold-specific resampling could fix the signal without a distinct method. **Screen out pending mechanism evidence.** |
| **C. Same-class interpolation can connect separate modes.** | SMOTE partners are nearest eligible same-class originals, without a region constraint. | **Mathematics:** a three-point mode with k=5 includes remote partners when eligible. But tuned clean UGO had only .2 mean outside-support additions and strong recovery; no residual bridging loss was established. | KMeansSMOTE, A-SUWO, EC-EBO and existing local generators already constrain regions. Outside-support counts alone do not establish prediction harm. **Screen out as unsupported and closely duplicated.** |

Sources: [UGO contract](../ugo_reproduction.md), [iteration code](../../src/ugo.py),
[generator](../../src/generators.py), and [validation selector](../../src/subgroup_diagnostic.py).
B's smallest change would replace in-sample scores with out-of-fit scores;
ordinary RF OOB is insufficient once held-out originals' synthetic descendants
can enter bags. Honest fold-specific augmentation is needed. C's smallest change
would restrict existing partners to a fitted region, retaining the original
eligibility mask and reporting shortfalls. Neither currently merits a developed
sampler specification.

The pilot constraints remain binding: zero generation is not failed learning;
tuned UGO, duplication and EC-EBO already recover the clean deficit; filter
relaxation has no demonstrated incremental need. In overlap, ROS rare recall
was 4–18% across seeds despite common-mode means 27.44%/22.12%. The bound
FPR≥.20×weighted recall is not a worst-group optimum. An **oracle** that recognizes
positive supports and accepts one quarter of every disk attains FPR=.05 and
every group's recall=.25. This mathematical benchmark uses unavailable support
knowledge; it proves neither learnability from a few labels nor A's efficacy.

## 2. Candidate A: exact proposed adaptation

### Information, grouping and abstention

Inputs are labelled numerical fitting data F, independent labelled calibration
data C and selection data S, and a finite predeclared pipeline catalog. No true
subgroup identifiers or unlabelled pool enter any learner. Test T remains sealed.
Meaningful Euclidean geometry is an application assumption, as in existing UGO.

Fit preprocessing on F. On **all original fitting positives**, fit KMeans with
20 initializations for K=2,…,min(8,floor(n_positive/3)). Retain partitions with
every cluster containing at least three originals; choose maximum silhouette,
ties favor smaller K; require silhouette≥.25. These are engineering defaults,
not published rules or semantic guarantees.

Check stability using 20 independently seeded 80% positive subsamples, refitting
the chosen K with the same preprocessing. Each refit assigns **all original F
positives**, including omitted rows. For original cluster C_j and refit clusters
C_l^(b), define

\[
q_j={1\over20}\sum_b\max_l
 {|C_j\cap C_l^{(b)}|\over|C_j\cup C_l^{(b)}|}.
\]

Require min_j q_j≥.8. Assignment h_F(x) is nearest original-fit centroid, with
fixed index ties. Grouping never changes in response to candidate predictions.
If grouping fails, or any inferred positive group has fewer than 20 S members,
return the **global-BA catalog selector** and record why. Do not drop an awkward
group or use evaluator identities to repair it. Every original row remains in
each UGO fitting input, including seed-ineligible observations.

### Calibration and selection

Fit all M selectable catalog predictors s_m on F before inspecting C/S. Let H
also count the separately reported, nonselectable reference predictors; all H
must be fixed before C. In the planned study M=80 and H=89, including eight
pooled-training controls described below. Let n_C be calibration negatives,
alpha=.05 and delta_F=delta_R=.025. Choose

\[
k=\max\{r\in\{0,\ldots,n_C-1\}:
 \Pr[\mathrm{Bin}(n_C,\alpha)\le r]\le\delta_F/H\}.
\]

For each of the H predictors, set tau_m to the (n_C−k)-th ascending negative
score; predict positive only when s_m(x)>tau_m. If no k exists, use +infinity. Under independent
class-conditional calibration draws, this standard order-statistic construction
and a union bound control all H population FPRs at alpha with confidence
at least 1−delta_F. Score ties are conservative. It controls an upper bound,
**not achieved FPR equal to 5%**. No threshold is tuned on T.

On selection positives, define

\[
n_j=\sum_{i\in S}1\{y_i=1,h_F(x_i)=j\},\quad
a_{mj}=\sum_{i\in S}1\{y_i=1,h_F(x_i)=j,s_m(x_i)>\tau_m\}.
\]

With J inferred groups and beta=delta_R/(MJ), use the one-sided Clopper–Pearson
lower bound

\[
L_{mj}=\begin{cases}
0,&a_{mj}=0,\\
\operatorname{BetaQuantile}(\beta;a_{mj},n_j-a_{mj}+1),&a_{mj}>0.
\end{cases}\qquad
m^*=\arg\max_m\min_j L_{mj}.
\]

Break ties by the frozen catalog order, starting with NORES/leaf=1. Return the
already-fitted model and threshold; **do not refit after selection**. Conditional
on F/C and group counts, independent S draws give simultaneous inferred-group
bounds. Combining the two error budgets gives at least 95% joint confidence
per replicate, not across all benchmark seeds. This provides no guarantee about
latent semantic groups, improvement over NORES, AP or population risk.

```text
fit F-only preprocessing and fixed minority partition; check stability
fit every declared pipeline on F; construct F-only matched controls
calibrate all thresholds on C negatives with the common order-statistic rule
score S once; compute global BA for the reference selector
if grouping invalid or any positive group has <20 S rows:
    return global-BA choice, with fallback reason
compute simultaneous subgroup recall lower bounds
return the first pipeline maximizing its minimum lower bound, without refitting
```

This changes **model selection only**. Each supplied UGO keeps its own filter,
MMA, class selection, Welch stop, seed/partner generator and postcorrection.
There is no subgroup quota, point/batch valuation or new stopping rule. Grouping
is needed because true subtypes are unavailable; held-out recall measures the
declared utility; bounds/fallback acknowledge tiny-group uncertainty; common
calibration makes false positives visible. These are established components.

**Decision example, not experimental evidence.** Suppose S has inferred-group
positive counts (420,150,30). At the same FPR=.05, pipelines A and B correctly
classify (113,27,1) and (97,33,6). Global BA prefers A (.5925 versus .5883), while
worst recall prefers B (.20 versus .0333); the stated lower-bound rule also
prefers B. UGO's in-sample ranking and entropy/sparsity quotas need not make this
choice. GEORGE-style robust selection can: the example illustrates adaptation,
not novelty.

**Failure behavior.** Tiny/unstable groups trigger fallback, not successful
learning. Stable clusters may still merge distinct semantic subtypes. Stable
label errors can resemble rare concepts; confidence is not label correctness.
The selector neither repairs corrupted labels nor removes Bayes overlap. An
unlearnable group can dominate its criterion. Linear capacity failure remains
capacity failure. Failure, fallback and semantic misalignment must remain visible.

## 3. Closest work and novelty conflicts

| Work and verified evidence | Objective / eligibility / allocation / generation | Consequence for these candidates |
| --- | --- | --- |
| **GEORGE**, author code [G], paper-linked README | Fits within-class clusters; assigns later rows with the fitted map; code explicitly supports inferred-subclass robust validation accuracy/AUROC checkpoint selection. | **Direct conflict with A's central mechanism.** FPR-constrained positive recall and finite-catalog confidence handling are task adaptations, not an established new objective family. |
| **Group DRO / JTT**, author code [D,J] | DRO updates q_g proportional to q_g exp(eta loss_g), and selects worst-validation-group accuracy. JTT identifies initial-model errors and upweights them on retraining. | Worst-group selection and model-specific error emphasis are established; difficulty alone is not a contribution. |
| **A-SUWO**, pinned third-party code [A] | Constrained minority clusters; CV LDA error e_g (singleton=1), allocation e_g/sum(e), uniform if all errors are zero; inverse-majority-distance seed weights; within-cluster synthesis. Noise removal uses zero same-label neighbors. | Held-out cluster difficulty and local generation already exist. Full primary-paper equivalence remains unverified. |
| **KMeans-SMOTE**, author/library code [K] | Author eligibility: (n_0g+1)/(n_1g+1)<IRT and n_1g>1; sparsity weight proportional to mean_distance^p/n_1g; within-cluster SMOTE. | Direct conflict with C and naive rarity quotas. Installed 0.12.4 exponent, eligibility and rounding differ from author defaults. |
| **EC-EBO**, primary-paper audit and current implementation [E] | Externally constrained clusters; mean entropy and largest-remainder quotas in this interpretation; coordinate generation using positive decreasing influence. Printed sum/mean and sign conflicts remain declared. | Mean difficulty allocation and majority-constrained generation already implemented. Do not claim exact author fidelity. |
| **SLEO**, primary pp.4–7 and author code [S] | Cluster entropy sum E_g=sum_i H_i; quota proportional to E_g; entropy-decay majority translation and cluster-centered radial generation. Code smooths probabilities by .001 without renormalizing. | Clustering plus uncertainty quotas is established; its moved originals and geometry confound allocation-only claims. |
| **AROSS**, self-described project code [R] | Safe/half-safe category budgets weighted by area counts; within-category quotas proportional to minority-neighbor counts; representative-centered Gaussian offsets and half-safe neighbor acceptance. | Region gating, representatives and local allocation are established in this code; paper identity/linkage remains unresolved. |
| **VTSS / Hamilton**, supplied primary papers [V,H] | VTSS selects synthetic size by CV; Remark 2 explicitly allows other task-relevant metrics. Hamilton shows learner-dependent threshold effects and an MLP exception. | A subgroup validation metric does not create novel stopping/count selection; shared thresholds and capacity controls remain necessary. |
| **ADPCHFO, SD-CSMOTE, PPEO** | Exact/refined searches and supplied extracts did not establish primary identities or verified implementations. New API/source requests returned proxy HTTP 403. | Equations and conflicts remain unknown. Failed searches are **not evidence of novelty**. |

Established: inferred-group robust selection, confidence bounds, calibration,
clustering, and each generator. Adaptation: their explicit supervised UGO
portfolio interface and recall-at-FPR protocol. Potential research contribution:
evidence about when this proxy objective helps real/evaluator-defined subtypes,
and when it fails. Global methodological novelty remains unestablished.

## 4. Comparison and selection

| Candidate | Evidence | Distinction | Coherence | Undergraduate feasibility / effort estimate | Main risk / mechanism test |
| --- | --- | --- | --- | --- | --- |
| **A: conservative inferred-group selection** | Moderate: measured exploratory utility trade-off plus mathematics; recovery unknown. | New repository capability; strong GEORGE conflict. | Complete finite-catalog rule with explicit assumptions/fallback. | Good: roughly 1–2 weeks including adapters, tests and reporting. | Proxy groups/selection noise; compare selectors on identical frozen predictions. **Recommend only as an adaptation.** |
| **B: honest MMA** | Mathematical possibility; no attributed residual loss. | A-SUWO/CV-hardness proximity; novelty unresolved. | Requires descendant-safe fold-specific resampling, not naive OOB. | Moderate: roughly 2–3 weeks; multiplies internal fitting. | No useful signal change, noise emphasized; needs scoring-optimism attribution first. **Defer.** |
| **C: local partners** | Mathematical bridging; no consequential pilot residual. | Closely duplicates existing clustered generators. | Simple restriction with explicit shortfalls. | Easy: roughly 1–3 days, but weak scientific case. | Clustering errors/omitted tiny modes; existing-method/count controls may explain everything. **Reject now.** |

These are planning estimates, not measured implementation times. Allocation or
subgroup stopping is not selected merely because filtering was unsuccessful.

## 5. Decisive fresh experiment and rejection rules

**Hypothesis.** On the declared overlap law, A improves mean evaluator-defined
worst-subgroup recall over global-BA selection from the **identical fitted
catalog**, by more than five percentage points without unacceptable FPR, AP or
population-risk loss. This is a selector hypothesis, not a new-generator claim.

**Smallest planned test.** Twenty paired fresh seeds **142–161**, clean and .20
overlap, using the existing disk/annulus geometry, prior .10 and conditional mode
weights 70/25/5. F/C/S/T totals **1,200/6,000/6,000/20,000**, with exact class
counts but **IID within-class mixture draws, not fixed latent-mode strata**.
Expected rare positives are 6/30/30/100. Each split has an independent RNG
namespace; every method receives identical labelled data. This deliberately
more estimable setting does not establish recovery of the original 3/10 pilot
condition. Larger validation label cost must be reported, not hidden.

Before any new outcomes, freeze source/dependency hashes, grouping defaults,
catalog order, folds/seeds, all settings and this protocol in a **new** manifest.
Never modify the historical diagnostic. Synthetic identities and support
knowledge are evaluator-only; score semantic alignment after selection, without
using it to choose K, methods or favorable replicates.
Report adjusted Rand index and per-subtype inferred-cluster coverage alongside
stability and fallback rates; none is a learner input or a case-selection rule.

**Catalog and strong controls.** Four UGO k_pre choices [5,3,1,7], plus NORES,
balanced class weighting, ROS, SMOTE, KMeansSMOTE, EC-EBO, A-SUWO and SLEO:
12 choices crossed with RF100 leaf sizes [1,2,4,8] =48 configurations. Keep the
diagnostic's UGO/SMOTE/KMeansSMOTE and audited EC-EBO settings. Declare A-SUWO
third-party defaults (proportion1, neighbors5, majority-clusters7, c_thres .8,
simplex dimension2); SLEO author defaults (alpha20, bagging20 trees, p_norm2),
with exception-safe seeded RNG scopes. Do not retune them after test access.
UGO phi=1 gives
cap960; other samplers request 960 additions. For each of the 16 UGO
configurations, precompute F-only exact final-class-count ROS and corresponding
point-multiplicity weights, fitted with the same RF settings: at most **80**
catalog models. Keep duplicates rather than reducing M after inspecting scores.
Natural phi=2.5/k=5/leaf=1 UGO is a separately reported original reference.

**Total-label-budget controls:** also fit NORES and ROS on F∪S with each of the
four leaf sizes, before C, with preprocessing fitted on F∪S. Pooled ROS balances
720/6,480 positives/negatives by adding **5,760** rows; it deliberately has more
training data and additions than the catalog control. For each method select
capacity by global BA on C after the same calibration. This C reuse is covered
by uniform FPR calibration; no recall confidence bound is claimed for that choice.
These eight models never enter A's selectable
catalog or its S-based recall bounds. Including them and the original reference
gives H=89 for the shared FPR correction. They test whether spending selection
labels on ordinary training is preferable; no label-efficiency claim follows
from the primary same-F comparison alone.

A-SUWO removes some originals and SLEO moves majority originals: they are
full-method controls, not exact count/geometry ablations. Report removals,
movements, requested/achieved counts and shortfalls. Do not fabricate exact ROS
matches when final counts fall below original counts. Source adapters must be
checked against the pinned implementations before execution; unavailable or
failed methods stay visible and block a comprehensive superiority claim.
Require every declared arm and all 20 paired primary cells for the planned
primary conclusion. Any missing arm/cell or time-cap interruption makes that
study incomplete/inconclusive; do not publish a favorable surviving-catalog or
complete-case primary. Keep M=80/H=89 penalties. Source-stage infeasibility may
revise the manifest only before new held-out outcomes, with the revision recorded.

Every selector shares the same predictions and calibration procedure, tuning
data, candidate fitting budget and random seeds. Compare:

1. Global-BA selection over the full catalog: the primary comparator.
2. Unpenalized minimum inferred-group recall over that catalog: a direct
   GEORGE-style criterion adaptation, testing whether confidence handling adds value.
3. A's criterion over the non-UGO base methods, and separately NORES/weighting/ROS;
   use the same full-catalog confidence penalty. These test whether ordinary
   existing choices explain the result.
4. A with one pooled positive group, and without the stability/count fallback:
   prespecified selection-only ablations, not new generators.

All scoring models are fitted once and reused. A's overhead is F-only clustering
plus O(M|S|) scoring/counting and MJ beta quantiles, with no extra classifier fit.
Compare the selected UGO model to its precomputed count/multiplicity controls;
RF bootstrapping means those controls are not equivalent. Changing which existing
pipeline is selected can change achieved counts; that is a policy effect, never
evidence for a new geometric generator. Report selected methods, capacity,
counts, thresholds and subgroup denominators for every seed.

**Frozen primary:** the all-seed paired difference in test minimum true-subgroup
recall, A minus global selector, for **overlap/RF100**. Require the lower end of
a paired-seed 95% bootstrap interval (10,000 replicates, RNG42) to exceed **.05**.
Keep fallback cases in that primary estimate. Require valid grouping/counts in
at least **16/20** overlap replicates; otherwise reject applicability in this
declared data regime, even if efficacy when active remains uncertain.

Trade-off gates against the global selector use paired confidence limits:
upper 95% FPR-increase limit≤.005; upper population-error-increase limit≤.002;
lower AP-difference limit≥−.005. Require achieved test FPR≤.06 in every primary
replicate and the upper 95% interval for mean test FPR≤.05. Report population
error as .9FPR+.1(1−weighted recall), AP, all recalls and achieved FPR. Clean
performance is a negative-control guard: lower paired worst-recall difference
limit≥−.02. All gates must pass; no isolated seed establishes success.

Use the same whole-paired-seed bootstrap resamples for the primary and all
trade-off gates. These twenty-seed intervals remain limited simulation evidence.
Other comparisons are exploratory; no secondary superiority claim is authorized
without a separately frozen multiplicity-adjusted procedure.
No retrospective per-seed test winner is a deployable policy. Failing to beat
the raw robust selector provides no evidence for A's confidence component;
recovery by the non-UGO/simple family supplies no UGO-specific contribution.
If pooled-training NORES/ROS is better, reject a label-efficient/practical
advantage claim even if the selection-mechanism contrast succeeds.

**Reject further method claims** if the primary practical margin or any guard
fails, the selector is mostly inactive, inferred improvements do not transfer to
true groups, or the same gains are explained by ordinary controls. A positive
primary result supports this task adaptation only. It warrants a separately
frozen independent geometry/real-data confirmation with meaningful held-out
subtype metadata before any broader claim; it does not authorize a new sampler.

**Cost estimate:** 40 seed/regime cells, at most 3,200 catalog final fits,
320 pooled-control fits, 40 original-reference fits and **680 UGO trajectories**.
Internal scoring fits are separate: at most 640×8+40×35=**6,520**, because the
fixed batch120 and controlled/natural caps960/4200 limit rounds before max_iter100.
Also record A-SUWO/SLEO internal fitting. Plan approximately **1–4 CPU hours**
for this small RF100 study; cap execution at four CPU hours and retain incomplete
jobs. This is an estimate: old F600 medians were .114 s for a NORES final fit
and .423 s for one k=5 UGO resampling run, not measurements of this larger study
or the new adapters. No compute was spent on that experiment in this stage.

## 6. Implementation and source-verification plan

Next stage: add a separate opt-in selection utility and runner, reusing
[preprocessing](../../src/preprocessing.py), [result identities](../../src/results.py),
[comparators](../../src/comparators.py), RF registry and metric/reporting patterns.
Keep `src/ugo.py`, historical entry points, manifests and CSV/JSON evidence intact.
Test F/C/S/T and group-access boundaries, IID mixture sampling, label/score ties,
order-statistic indexing, simultaneous-bound arithmetic, stability label
permutations, tiny/missing groups, fallback, original-row behavior and failed
adapters. Validate pinned adapters, then run focused and full tests before any
new benchmark. This document is not an implemented API or execution command.

Verified sources are distinguished from inaccessible papers:

- **[G]** Sohoni et al., *No Subclass Left Behind*, [paper identity](https://arxiv.org/abs/2011.12945)
  from author README; pinned [GEORGE schema](https://github.com/HazyResearch/hidden-stratification/blob/fc9c9d05d8f96fbfc1badb546a812d6d893c1dc3/stratification/utils/schema.py#L94),
  [clustering](https://github.com/HazyResearch/hidden-stratification/blob/fc9c9d05d8f96fbfc1badb546a812d6d893c1dc3/stratification/cluster/george_cluster.py#L91)
  and classification checkpoint code directly inspected. Full paper inaccessible.
- **[D,J]** Sagawa et al., [Group DRO paper identity](https://arxiv.org/abs/1911.08731)
  and [verified training code](https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/train.py#L205);
  [JTT paper identity](https://arxiv.org/abs/2107.09044) and
  [verified error-identification code](https://github.com/anniesch/jtt/blob/1de6cc5b65e56e8d4db9ad6beb27c3070ba3ee1b/process_training.py#L30).
  Author implementations/READMEs verified; full papers inaccessible.
- **[A,K,E,S,R]** Exact paper/code citations and revisions in the existing
  [source ledger](QUESTION_B_RESEARCH_PROPOSAL.md#11-references-provenance-and-unresolved-work)
  and [EC-EBO audit](../EC_EBO_METHOD_DECISIONS.md). Rechecked cached pinned
  A-SUWO allocation lines258–302, KMeans author filtering/weights174–227,
  SLEO weighting88–93 and primary Eqs.4,7. AROSS publication identity and
  A-SUWO/KMeans primary full-paper access remain unresolved.
- **[V,H]** Supplied Ma–Zhang [v3](https://arxiv.org/abs/2601.16120v3),
  Section6/Remark2 pp.20–21 and Algorithm1 p.22; Hamilton
  [primary](https://doi.org/10.1016/j.neucom.2026.134831), Section5.5/Table5.
  Read as qualified evidence, not universal guarantees for this pipeline.

This turn's Crossref/OpenAlex/Semantic Scholar/arXiv/OpenReview/PMLR and search-API
attempts were blocked by HTTP403; reachable author raw-code files and existing
cached sources supplied the implementation checks. Raw papers/source snapshots
and request logs stay outside Git. The excluded provisional PDF, inactive
historical designs, Question C and unlabelled-data generation were not used.
