# Question B proposal: establish rare-mode failure before changing UGO

Date: 2026-10-17. Source baseline inspected:
`3e5b713c773aa3dbaec8f7224d9cb85b9947b960`.

**Recommendation:** select one diagnostic research direction: determine when
UGO's neighborhood noise filter excludes a genuine rare minority mode, whether
that exclusion causes independent prediction loss, and whether existing tuning
or existing methods already remove it. Selecting a new sampler is premature.
This is a fresh supervised study proposal, not an approved algorithm or a claim
of methodological novelty.

Only labelled fitting, validation and test data are in scope. No separate
unlabelled pool or semi-supervised generation is proposed. The user-excluded
`UGO_Question_B_Provisional_Proposal.pdf` and inactive RB-UGO/CSG or other
historical candidate designs were not used as research evidence or mechanisms.

## 1. State of the repository and boundaries of this stage

The preceding task added only `docs/research/QUESTION_B_ASSESSMENT.md` in
`95aaaca`. At the user's request, forward revert `a1795d1` undid that exact
commit and was pushed to `question-b-assessment`. The resulting tracked tree
matched the source baseline above. No other work was reverted or history
rewritten. Existing ignored software-check outputs were preserved. No repository
experiments or tests were running when inspected.

This proposal is on `question-b-research-proposal`. No sampler, baseline,
dependency, experiment runner, test, frozen manifest or historical result has
been changed. No predictive experiment was launched. Read-only source inspection
and small mathematical/eligibility checks were performed.

Existing components to reuse:

- [UGO](../../src/ugo.py), [MMA](../../src/uncertainty.py),
  [noise handling](../../src/noise_detection.py),
  [stopping](../../src/stopping.py), and [generators](../../src/generators.py).
  UGO-ROS/SMOTE/RWO/MCSMOTE already exist. Keep their fidelity decisions in
  [ugo_reproduction.md](../ugo_reproduction.md) explicit.
- [Preprocessing](../../src/preprocessing.py), [paired evaluation](../../src/evaluation.py),
  [result identities/caching](../../src/results.py), and
  [classifier registry](../../src/classifier_registry.py). A fresh downstream
  classifier is already fitted after resampling. The last UGO scoring model
  must not be used as the evaluated final classifier.
- [Comparators](../../src/comparators.py), including ROS, SMOTE, BFCH-SMOTE and
  EC-EBO. Existing [EC-EBO clustering](../../src/ec_ebo_clustering.py) and
  [mean-entropy allocation](../../src/ec_ebo_entropy.py) are useful close
  comparators, not a new contribution to rename.
- [Simulations](../../src/simulations.py) and [pilot reporting](../../src/pilot.py)
  provide patterns for declared data-generation choices and paired records.
  They do not yet implement this study's independent subgroup identities.

The metric module supplies global F1, MCC, G-mean, IAM and probability ROC-AUC.
This study would need AP, balanced accuracy, FPR and subgroup metrics, plus
validation-only threshold selection and classifier-weight controls. KMeansSMOTE
is available in the installed imbalanced-learn version but is not registered
in the existing runner. No raw historical datasets are currently present in
`data/raw/`; this proposal does not rely on recovering them.

## 2. Prediction objective: counts are not the problem definition

Suppose a fixed target population has

\[
P^*(Y=1)=0.10,\qquad
P^*(G=g\mid Y=1)=(0.70,0.25,0.05).
\]

These frequencies can be perfectly legitimate. Equalizing them is not a goal.
For a score s and decision \(d_\tau(x)=1\{s(x)>\tau\}\), define

\[
R_g(\tau)=P^*(d_\tau(X)=1\mid Y=1,G=g),\quad
FPR(\tau)=P^*(d_\tau(X)=1\mid Y=0).
\]

The proposed scientific objective is **worst-group sensitivity at a fixed
false-positive operating point**, \(\min_g R_g\) at a validation-selected
FPR target \(\alpha=0.05\). This is appropriate only when missing a meaningful
rare subtype matters independently of its prevalence. It is a declared study
utility, not a discovered deployment requirement or a claim about arbitrary
clusters. Real applications would need independent semantic group definitions
and a justified cost/FPR target before transferring this recommendation.

Always also report population-weighted recall
\(\sum_g w_gR_g\), population error
\(0.9FPR+0.1\sum_gw_g(1-R_g)\), and balanced error
\(\tfrac12[FPR+\sum_gw_g(1-R_g)]\), using the same fixed weights.
Macro group recall changes the utility weighting and must be labelled explicitly.
For example, changing recalls from (.95,.90,.20) to (.95,.90,.80) changes
population-weighted minority recall from .90 to .93, but macro group recall
from .6833 to .8833. Those are different benefits, not interchangeable evidence.
Any FPR increase or loss in common-group performance remains visible.

## 3. Evidence ledger

Labels: **P** = explicit paper algorithm/statement; **R** = authors' reported
experimental result; **I** = mathematical or methodological inference;
**H** = unverified research hypothesis. Implementation evidence is identified
separately. PDF page numbers refer to the supplied PDFs, not journal pagination.

| Claim | Status and precise evidence | Consequence |
| --- | --- | --- |
| UGO uses local filtering and label-aware MMA, but class summaries drive class choice/stopping. | **P:** UGO pp.6–8, Sections 3.2–3.4, Tables 2–4. **Code:** `UGO.fit_resample`, `noise_pre_detect`, `welch_stopping`. | Trace each stage; do not describe UGO as uniformly sampling the minority class. |
| A small subgroup can have little influence on a class mean. | **I:** \(\bar u_1=\sum_g(n_g/n_1)\bar u_g\). A 70/25/5 mixture of (-.9,-.4,+.8) yields -.69. | Algebra establishes dilution, not actual generation choices, subgroup harm or recoverability. |
| UGO can already emphasize eligible rare difficult points. | **P:** UGO p.7, Section 3.4; Table 4, p.8. **I:** if three rare points rank highest in a 30-seed pool, their seed share is 10%, exceeding a 5% original share. | Allocation failure must be measured, not assumed from class means. Class choice and stopping still intervene. |
| A vote-based noise filter can reject a correctly labelled small mode. | **I + executed construction:** Section 7 below. **Code:** fewer than k/2 same-label neighbors marks a row; original labels are retained for fitting but excluded from generation. | Establishes a possible exclusion mechanism. **H:** it causes independent prediction loss in realistic conditions. |
| SLEO already clusters the minority class and uses summed cluster entropy for allocation. | **P:** SLEO pp.4–7, Sections 3.1–3.3, Eqs.1–8, Algorithm 1. **Author code:** `semi_cluster`, `compute_entropy`, `compute_weight`. | Minority clustering plus entropy quotas is existing machinery. Equal per-point entropy still produces size-proportional summed weights. |
| SLEO performs well on its evaluated benchmarks. | **R:** Sections 4–5, pp.7–16, global AUC/G-mean/F-measure comparisons. | These results do not establish independently known rare-subtype recall under our fixed target or universal superiority. |
| Threshold and learner capacity can explain apparent augmentation advantages. | **R:** Hamilton Section 5.5, Table 5, pp.8–9: threshold parity depends on learner; the single-MLP ranking exception is discussed. **I:** equal threshold and capacity controls are necessary here. | Do not convert the paper's results into a theorem that oversampling never helps. |
| Synthetic-distribution mismatch and quantity matter. | **P:** Ma–Zhang pp.9–12, Eqs.2–4 and stated theorem assumptions; p.22, VTSS Algorithm 1. | Measure sample quantity and mismatch; global count tuning is not itself a new mechanism. The theory is not a guarantee for arbitrary UGO/tree pipelines. |
| High uncertainty identifies useful generation regions. | **H**, not established by MMA, entropy or low density. GDEO pp.4–6, Eqs.1–7 provide another existing uncertainty-weighted generator. | Independent prediction checks and noise/overlap controls are essential. |

The UGO cap/pool/rounding ambiguities remain as documented in the repository.
In binary code MMA is \(1-2P(y_i\mid x_i)\); it is signed disagreement, not a
formal epistemic/aleatoric decomposition. Welch non-rejection is not proof of
subgroup equality or acceptable risk.

## 4. Close prior work: mechanisms and verification limits

### Verified sources and implementations

| Method/source | Grouping and allocation | Noise/generation and evaluation relevance |
| --- | --- | --- |
| **KMeans-SMOTE, author repository** [K1] | K-means on all fitting rows; retain clusters passing a majority/minority ratio threshold and having more than one minority point. `_filter_clusters` uses sparsity \(\bar d_g^{p}/n_{1g}\), normalized over retained clusters; default p is feature dimension in this code. | Within-cluster SMOTE; author code rounds counts and falls back to ordinary SMOTE if all cluster weights vanish. Sparse-region allocation and cluster eligibility are therefore existing ideas. The author package targets older dependencies and was inspected, not installed or executed. |
| **KMeansSMOTE, installed library 0.12.4** [K2] | Similar pipeline, but default exponent/eligibility details differ; normalized cluster sparsities and per-cluster ceiling are explicit. | Realized total can differ from the requested budget. Do not attribute this version's automatic exponent to an unread original-paper equation. No subgroup metric is supplied by the sampler. |
| **A-SUWO, smote_variants implementation** [A1] | Majority agglomeration and constrained minority merging; `adaptive_sub_cluster_sizing` normalizes cluster-specific held-out LDA minority error. Singleton error is assigned 1 in this implementation. | `remove_noise` removes rows having **zero** same-label neighbors, a different rule from UGO's opposing-majority vote. Within-cluster weights use capped inverse nearest-majority distances; generation uses the library's simplex framework. These are verified third-party code behaviors, not a verified reading of the original paper or author-code equivalence. |
| **SLEO, supplied paper and paper-linked code** [S1,S2] | Minority single-link merging constrained by nearest-majority distances; sample entropy is **summed** within clusters and normalized for quotas. | Majority translation and cluster-centered radial generation are part of the method; it changes more than allocation. Its benchmark uses global metrics, not our known-group estimand. The paper/code distinctions below matter for a faithful comparator. |
| **AROSS, self-described project repository** [AR1] | README expands AROSS as *Area-based Representative Points Oversampling with Shifting*. Code clusters, extracts representatives, divides safe/half-safe area budgets by weighted area counts, then allocates within a category in proportion to minority-neighbor counts. | Code generates truncated Gaussian coordinate offsets around representatives and applies same-class neighbor acceptance in half-safe areas; optional representative shifting is documented. Code uses global NumPy randomness. Bibliographic citation is a placeholder in this repository, so matching it to a specific published primary paper and reported evaluation remains unverified. A search hit calling AROSS “Adaptive Representative Oversampling” was not accepted as authoritative. |
| **EC-EBO, existing repository implementation** | Externally constrained minority clusters; mean member entropy, largest-remainder quotas, explicit shortfalls. | A close existing Question B comparator. Its current interpretation differs from the printed sum rule discussed in its [method decisions](../EC_EBO_METHOD_DECISIONS.md); the EC-EBO PDF was not among the current attachments. |
| **GDEO, supplied primary paper** [G1] | Probability Gini weights affect seeds; no explicit subgroup coverage constraint appears in the inspected pp.4–6 method. | Gradient-based generation is another close uncertainty/difficulty ingredient. A verbal equilibrium/no-crossing description is not proof of minority membership or improved rare-mode recall. |

SLEO's paper prints Eq.3 without the usual entropy minus sign. Its actual
paper-linked code at the pinned revision uses
`-sum((probs + .001) * log2(probs + .001))`, without renormalizing after the
addition; sums these quantities within clusters; uses a **0.01** decay cutoff;
and uses `np.rint` for quotas. These are observed author-code rules, not repairs
invented here. Radial directions are normalized cube draws, not necessarily
uniform directions on a sphere; radius is uniform. No code from SLEO was run.

### Sources not verified adequately

| Requested competitor | What was checked | What remains unknown |
| --- | --- | --- |
| ADPCHFO | Exact acronym searches in GitHub repository names/readmes and supplied supervised PDF extracts. | No exact primary publication or official implementation identified. Grouping, allocation, noise rule, objective and evaluation cannot be asserted. |
| SD-CSMOTE | Same search scope, including quoted hyphenated acronym. | Same unresolved bibliography/mechanism/evaluation status. |
| PPEO | Exact acronym plus oversampling in repository/readme search and supplied extracts. | Same unresolved bibliography/mechanism/evaluation status. |
| A-SUWO original paper | DOI identified in supplied references and library citation: `10.1016/j.eswa.2015.10.031`. Implementation examined as above. | Full primary methodology/results not obtained; the library flags paper ambiguities, which were not independently checked against the paper. |
| KMeans-SMOTE original paper | Author README/code and installed library examined. | Original arXiv HTML/PDF returned HTTP 403; original-paper equations and benchmark protocol not newly verified. |

Crossref API and Google search requests failed at the network CONNECT proxy
with HTTP 403; GitHub pages/raw files/Git were accessible. These searches are
limited and do not establish absence of publications. The original AROSS
repository was located through a broader README search after a name-only search
returned a derivative project. No mechanisms or citations were invented for
the unresolved acronyms. Exhaustive novelty or superiority claims remain blocked;
the diagnostic proposal below does not require asserting either.

## 5. At most three candidate directions

| Direction | Support and closest work | Contribution, coherence, effort and falsification | Decision |
| --- | --- | --- | --- |
| **1. Diagnose the noise-filter scale before changing minority seed eligibility.** | UGO vote rule plus the executable exclusion construction; A-SUWO's less aggressive isolation rule and KMeans/AROSS eligibility are close controls. | A controlled account of when reliable modes disappear from the generation pool, whether prediction suffers, and whether existing tuning suffices. One existing parameter can already fix the toy. Low instrumentation effort; no asserted new algorithm. Falsified as a useful redesign target if tuned UGO/strong controls recover the mode or exclusion does not reduce test sensitivity. | **Selected diagnostic direction.** |
| **2. Introduce explicit subgroup allocation.** | Class-mean dilution is algebraically possible, but ranking may already favor eligible rare points. KMeans-SMOTE, A-SUWO, SLEO and EC-EBO already allocate across clusters. | Would need a demonstrated allocation deficit after eligibility, stable training-only group definitions, a utility-matched need estimate, and a distinct mechanism. Moderate effort and substantial prior-art proximity. Reject if apparent gains are explained by count matching or ordinary weighting. | Defer; insufficient performance evidence and no distinct contribution established. |
| **3. Replace class-level stopping with a subgroup-sensitive stop.** | Welch does not imply subgroup adequacy; no actual harmful stop trajectory has been demonstrated. Existing quantity selection, including VTSS, is a necessary control. | Requires reliable groups and an independent selection objective; noisy tiny groups could prolong generation indefinitely. Higher selection/validation complexity. Reject if extra compute/count alone explains gains or test FPR worsens. | Defer; no justification to alter stopping now. |

Exactly one **research direction**, not one new sampler, is selected. It does
not revive historical intervention-valuation or graph mechanisms. If existing
configuration or prior methods solve the measured issue, the output should be
a configuration recommendation or a negative result, not a new method name.

## 6. Selected direction: precise specification

**Research question.** At a fixed target population and equally calibrated FPR
operating point, when does UGO's local opposing-majority filter exclude a clean
rare mode with the same population separability as larger modes, and is any
resulting sensitivity loss removed by existing neighborhood tuning or controls?

**Plain-language problem.** A few legitimate observations can be surrounded by
many observations from another class. A local vote may call them noise. We need
to find out whether keeping them available to the existing generator matters
for new examples, rather than assuming that they need extra synthetic points.

**Verified limitation:** the filter can remove all rare-mode anchors from the
seed pool. **Unverified consequences:** real UGO trajectories allocate too
little to that mode, stopping is harmful, retaining the anchors improves useful
prediction, or a new method is needed.

**Preliminary contribution:** a reproducible distinction among (a) seed
exclusion, (b) class/seed-selection or stopping effects, (c) classifier capacity,
(d) operating-point effects, and (e) unlearnable overlap/noise. This is a
proposed empirical contribution, not established scientific novelty.

Let \(N_k(i;F)\) exclude i itself and define

\[
a_i(k)=1\left\{\sum_{j\in N_k(i;F)}1\{y_j=y_i\}\ge k/2\right\},\quad
C_g(k)={1\over n_{1g}}\sum_{i:y_i=1,G_i=g}a_i(k).
\]

Use effective \(k=\min(k_{pre},|F|-1)\) and the repository's stable tie rule.
\(C_g\) is evaluator-only eligible coverage. A mode of q points whose other
same-class modes lie beyond the first k neighbors has at most q-1 same-label
neighbors per point. If \(q-1<k/2\), every point is ineligible even when its
label is correct. This is a finite-neighbor statement, not a population
impossibility theorem.

Record the following path separately: eligible coverage; actual class chosen;
ranked seed eligibility; drawn anchors and partners; generated/relocated rows;
stopping reason; final independent predictions. UGO leaves filtered originals
in classifier fitting data. Therefore zero seed coverage does **not** imply
zero classifier exposure or prove zero recall.

All methods choose thresholds from validation data only:

\[
\tau_\alpha=\min\left\{t\in\{-\infty,+\infty\}\cup
\{s(x_i):i\in V,Y_i=0\}:\widehat{FPR}_V(1\{s>t\})\le\alpha\right\}.
\]

This fixes the tie convention and gives every method the same monotone
operating-point opportunity without validation subgroup labels. Empirical
validation FPR does not guarantee test FPR; measure the latter with uncertainty.
Constant scores can select an all-negative rule and must remain visible.

**Inputs:** disjoint F,V,T; shared numerical feature semantics; declared
target distribution and noise regime; seeds; existing samplers/classifiers.
Only the evaluator receives latent G and any pre-corruption labels.
**Outputs:** eligibility/trajectory traces, global and subgroup test metrics,
paired differences/intervals, timings, counts, failures, and a decision to stop,
retune existing UGO, or investigate a justified minimal change later.
**Parameters:** initial k values 5 and 3; primary FPR target .05; default UGO
parameters otherwise; seed list and distributions frozen below. No new
allocation, generator or stopping equation is proposed in this stage.

Missing training groups, insufficient interpolation neighbors, constant scores,
unavailable comparators and unmet quotas must be diagnosed explicitly. A group
missing from a test set has undefined recall, not zero or perfect recall.
Clustering instability and local label corruption are reasons to distrust a
retention rule; a coherent small cluster alone does not establish meaningfulness.

### Diagnostic pseudocode (planned, not implemented)

```text
freeze target law, regimes, seeds, methods, objectives and primary contrast
for each independent replicate:
    draw disjoint fitting F, validation V and test T from the declared law
    retain true subgroup/noise identities in evaluator storage only
    for each prespecified learner and arm:
        fit preprocessing on F; transform V and T without fitting
        run the existing arm on F only; log eligible anchors and trajectory
        fit a fresh downstream classifier on that arm's fitting output
        choose its threshold using V, never G or T
        evaluate once on T; join subgroup identities only in the evaluator
    retain all failed/unavailable arms and pair shared successful replicates
report primary contrast, subgroup denominators, counts, FPR and uncertainty
apply the prespecified rejection/continuation rules; do not tune on T
```

The smallest potential intervention, **if later warranted**, is an explicitly
selected existing neighborhood configuration. A more permissive same-class
support rule would first be an A-SUWO-inspired adaptation requiring a noise
control, not an automatically novel algorithm. Introducing grouping, new quota
weights and new stopping simultaneously would prevent this diagnostic from
isolating the question.

## 7. Mathematical sanity checks completed

One fitting-only artificial construction used minority counts 42/15/3, no label
flips, and seed 42. Large modes are near (-2,2) and (2,2); the rare points are
(0,-.02),(0,0),(0,.02), surrounded by six majority points at radius .08.
Other majority points are near (0,-3). After the existing fold preprocessor:

| Existing UGO `k_pre` | Eligible counts by true minority mode |
| --- | --- |
| 1 | 42 / 15 / 3 |
| 3 | 42 / 15 / 3 |
| 5 | 42 / 15 / 0 |
| 7 | 42 / 15 / 0 |

These computations were executed using the current `noise_pre_detect`; all
expected counts matched. They demonstrate both possible exclusion and an
existing parameter control that fixes this construction. No UGO trajectory or
prediction-performance claim follows. Separately, equal per-point entropy in
clusters of size 20 and 2 yields summed allocation proportions 20/22 and 2/22;
changing from sum to mean is an existing alternative, not proof of novelty.

Reproduce the eligibility check from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python - <<'PY'
import numpy as np
from src.noise_detection import noise_pre_detect
from src.preprocessing import FoldPreprocessor
rng = np.random.default_rng(42)
minority = np.vstack([rng.normal((-2, 2), .08, (42, 2)),
                      rng.normal((2, 2), .08, (15, 2)),
                      [[0., -.02], [0., 0.], [0., .02]]])
a = np.arange(6) * np.pi / 3
majority = np.vstack([rng.normal((0, -3), .12, (174, 2)),
                      .08 * np.c_[np.cos(a), np.sin(a)]])
X = np.vstack([majority, minority])
y = np.r_[np.zeros(180, dtype=int), np.ones(60, dtype=int)]
g = np.r_[np.full(180, -1), np.repeat([0, 1, 2], [42, 15, 3])]
X = FoldPreprocessor().fit_transform(X, y)
for k in (1, 3, 5, 7):
    excluded = noise_pre_detect(X, y, k)
    counts = [int(np.sum((g == j) & ~excluded)) for j in range(3)]
    assert counts == ([42, 15, 3] if k in (1, 3) else [42, 15, 0])
    print(k, counts)
PY
```

## 8. Smallest prediction diagnostic before any method development

**Fixed target law.** Use three meaningful *by construction* compact minority
modes centered at (-3,0),(0,0),(3,0), each uniform in a radius-.10 disk.
Their target conditional weights remain 70/25/5. Majority data are an equal
mixture of uniform annuli with radii .25–.45 around the same centers. Sample
disk radius as \(.10\sqrt U\), annulus radius as
\(\sqrt{.25^2+U(.45^2-.25^2)}\), and angle uniformly. Clean supports are
disjoint: population Bayes error is zero, with the same separation geometry
for all minority modes. These are controlled latent modes, not a claim that
arbitrary clusters are meaningful application subtypes.

Each replicate independently samples F=600, V=2,000 and T=10,000 at fixed
class/subgroup counts matching the target law. F therefore has 540 majority
and 42/15/3 minority observations; V has 1,800 and 140/50/10; T has 9,000
and 700/250/50. Exact-count stratification is a declared simulation design,
not a naturally sampled demographic distribution. Features are independently
drawn within each stratum. All methods receive identical F,V,T for that replicate.

**Initial matrix:** seeds 42–51, two existing learners (LogReg and RF), and
four frozen arms: threshold-tuned NORES, threshold-tuned class-weighted NORES,
original UGO-SMOTE with k=5, and original UGO-SMOTE with k=3. This is 80 final
models, plus UGO's internal fits; log those internal costs too. Use repository
classifier settings, one native thread, unchanged UGO cap interpretation,
and the same validation threshold rule. There is no hyperparameter search in
this initial controlled one-parameter diagnostic. Both UGO settings remain
labelled existing UGO configurations. This matrix has **not been run**.

RF is essential because linear models cannot represent minority disks inside
majority annuli; a poor LogReg result alone is not an oversampling gap. No group
identity enters either practical learner, sampler or threshold selection.
Compare the primary RF k=3 versus k=5 paired effect on worst-group recall,
while showing each group's recall and its denominator. Class-weighted and
unweighted RF/LogReg separate simple decision/capacity remedies from synthesis.

Actual UGO counts/stopping can differ across k. The first contrast measures the
**total effect of changing an existing filter parameter**, including its later
trajectory, not a pure allocation effect. If prediction differs, repeat with
training-only ROS/point weights matching the realized final counts of each
class. Match the number of repeated fits/selection opportunities in a
no-synthesis computation control. A seed-mask replay with a frozen trajectory
is an explicitly artificial diagnostic, not another practical method.

If no repeatable loss appears, or a simple control removes it, stop method
development for this supposed gap. If a gap survives, use new seeds/test draws
for the expanded study below. Test results may reject a proposal; they cannot
be reused to tune a revised method and then serve as untouched confirmation.

## 9. Expanded study, ablations and rejection criteria

Only after the initial diagnostic justifies it, preregister a separate matrix:

- Add ROS, SMOTE, KMeansSMOTE, EC-EBO, BFCH-SMOTE and source-validated A-SUWO,
  SLEO and AROSS adapters where feasible. Resolve the other requested primary
  sources before any broad comparative novelty claim. Unavailable/failed methods
  remain in coverage reports. No legacy third-party code is assumed compatible.
- Use the same fixed target distribution while varying fitting sample size
  (600 versus 2,000), training subgroup representation, and geometry separately.
  Include a matched-representation training arm 1/3–1/3–1/3 while retaining the
  70/25/5 test target; it tests sampling shift rather than declaring equal counts
  intrinsically correct. Preserve the training/target distinction in reporting.
- Add a known-overlap negative control by replacing fraction .20 of the
  majority conditional law with the minority conditional law; labels remain
  sampled from the declared mixture. Separately corrupt .02 of training-majority
  labels, retaining clean target V/T. Gold corruption indicators stay evaluator
  only. These distinguish sparse clean structure from overlap and label noise.
- Give methods the same validation-only operating-point selection. For broader
  tuning, cap each method at the same number of declared candidate settings
  and classifier configurations; refit preprocessing/clustering/resampling on
  each inner fitting partition. Search quantity over a common declared grid
  (e.g. 0, half and one training class-gap), and report realized per-class
  counts, removals, shortfalls, selection fits and runtime. Retain a separately
  labelled natural-budget UGO arm; its total cap and possible majority additions
  are not equivalent to ordinary minority balancing.
- Match ROS and normalized sample-weight controls to realized fitting-only
  class counts where the learner supports them. Fix regularization conventions
  and report any differences between duplication and weighting. For majority
  translation/removal methods, class-count matching does not match geometry
  or row retention; expose those additional changes explicitly.
- Evaluate test per-subgroup recall, macro/worst recall, AP, F1, balanced
  accuracy, FPR, and target-weighted risk. Count eligibility, selected/drawn
  seeds, partner identities, generated/relocated rows and injected-noise
  ancestry separately. Anchor lineage is not a true semantic subgroup label
  for an interpolated point. Use known-support/mismatch checks only as labelled
  evaluator/oracle diagnostics, never as practical generation filters.
- Isolate one eligibility change first, keeping generator, allocation and
  stopping rules fixed. If future grouping/allocation/stopping changes are
  justified, use separate arms and realized-count controls. A beneficial
  interaction requires a factorial contrast and uncertainty, not merely a
  combined arm beating its components.

For paired metric differences \(D_r=M_{r,a}-M_{r,b}\), report
\(\bar D=R^{-1}\sum_rD_r\), all replicate values, and 95% intervals by
resampling **whole independent simulation replicates**, not CV folds or test
rows as independent training runs. The 10-seed initial intervals are exploratory
and can be wide. A later confirmation should use, for example, 30 new seeds
142–171 and a frozen primary contrast; secondary multi-comparator analyses
need multiplicity handling. Preserve common-success coverage and sensitivity
to exclusions; never drop failed jobs silently.

Predeclare .05 absolute worst-group recall as the initial practical effect
margin and .01 absolute FPR increase as the largest tolerated difference.
These are proposed study conventions, not domain-derived constants. Promotion
would require independent confirmation of a useful recall contrast, acceptable
absolute test FPR and no material target-risk/ranking regression; a small pilot
alone cannot establish it. Population risk/AP tolerances would need freezing
with the confirmatory protocol rather than choosing them after seeing outcomes.

Reject or substantially revise the direction if:

1. The independent gap disappears with existing k tuning, threshold tuning,
   weighting or an adequately expressive no-resampling learner.
2. Excluded anchors do not reduce actual generation access or held-out recall;
   original-row classifier exposure already suffices.
3. Count-matched ROS/weights or extra computation explains the apparent benefit.
4. Retaining low-support observations amplifies injected noise or raises FPR;
   apparent benefit occurs only with evaluator/oracle labels or group access.
5. Effects fail on new seeds, intervals remain inconclusive, or improvements
   depend on unstable clusters or an undeclared target-distribution change.
6. Source-validated existing subgroup methods already address the same failure
   with comparable cost. Report their adequacy rather than claiming novelty.

## 10. Implementation mapping for a later approved stage

| Future need | Existing location and boundary |
| --- | --- |
| Trace existing eligibility/class choice/seed selection/stopping | `src/ugo.py`, `src/noise_detection.py`, `src/generators.py`; consume existing diagnostics. Keep original UGO unchanged. |
| Compact-mode simulation with evaluator-only identities | Add a separate supervised simulation utility alongside `src/simulations.py`; preserve the historical UGO reconstructions. |
| Independent fitting/validation/test runner | Reuse `FoldPreprocessor`, classifier/comparator constructors and `ResultStore`; add an opt-in supervised runner. The generic ten-fold runner has no validation boundary, so it cannot simply be reused unchanged for tuning. |
| Metrics and uncertainty | Extend evaluator/reporting utilities without changing historical metric definitions or artifacts; AP requires scores and fixed label orientation. |
| Third-party competitor adapters | Pin source revisions and reconcile RNG/API/count semantics in separate wrappers. No new package installation or adapter implementation occurred here. |
| Validation | Later focused tests should enforce no V/T/group-label access in training, exact splits/identities, metric denominators, threshold ties, RNG reproducibility and failure reporting; then run the full suite. |

The present document and its sanity-check command were checked for internal
links, reproducibility and whitespace. Source/tests/manifests and historical
CSV/JSON artifacts are unchanged. No new sampler or experiment command is
advertised as implemented.

## 11. References, provenance and unresolved work

- **[U1]** Sağlam & Cengiz, *Uncertainty guided oversampling: A classifier-specific
  approach for effective handling of class imbalance in machine learning*, KBS
  342 (2026), 115886. [DOI](https://doi.org/10.1016/j.knosys.2026.115886).
- **[S1]** Tao, Wang & Wang, *Semi-supervised local entropy-decayed oversampling
  for imbalanced data*, KBS 333 (2026), 115009.
  [DOI](https://doi.org/10.1016/j.knosys.2025.115009).
- **[S2]** Paper-linked [SLEO code](https://github.com/taoliangliang/SLEO/tree/2b4b78e31726aa59e798faf21c8e453093fffa7e),
  revision `2b4b78e31726aa59e798faf21c8e453093fffa7e`.
  `sleo.py`: `compute_weight` lines 88–93, `entropy_decay` 95–129,
  `compute_entropy` 131–153, `semi_cluster` 155 onward; `prob.py` and distance
  utilities also inspected. Code was read, not executed.
- **[H1]** Hamilton, *Don't oversample the boundary: A remedy decomposition of
  the minority deficit in imbalanced learning*, Neurocomputing 704 (2026),
  134831. [DOI](https://doi.org/10.1016/j.neucom.2026.134831).
- **[M1]** Ma & Zhang, *Synthetic Augmentation in Imbalanced Learning: When It
  Helps, When It Hurts, and How Much to Add*,
  [arXiv:2601.16120v3](https://arxiv.org/abs/2601.16120v3), 7 March 2026.
  Relevant main-text passages inspected; the 90-page supplement/proofs were not
  exhaustively audited.
- **[G1]** Jiang, Zhu & Liu, *GDEO: Gini-guided discriminative equilibrium
  oversampling for imbalanced classification*, Neurocomputing 696 (2026),
  134121. [DOI](https://doi.org/10.1016/j.neucom.2026.134121).
- **[K1]** Last, Douzas & Bacao, *Oversampling for Imbalanced Learning Based on
  K-Means and SMOTE*, [arXiv:1711.00837](https://arxiv.org/abs/1711.00837),
  identified from the [author repository](https://github.com/felix-last/kmeans_smote/tree/e65e17e5456a8a0722cd81334c4a6661c92d400c).
  `kmeans_smote.py`: `_filter_clusters` lines 174–227, `_fit_resample` 230 onward.
  Original full paper unavailable here; author code/README inspected.
- **[K2]** [imbalanced-learn 0.12.4 source](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/0.12.4/imblearn/over_sampling/_smote/cluster.py),
  `_find_cluster_sparsity`, `_fit_resample`.
- **[A1]** [smote_variants A_SUWO source](https://github.com/analyticalmindsltd/smote_variants/blob/c26cdd9d4a2a61886924fc9b081f0480f24b4108/smote_variants/oversampling/_a_suwo.py),
  revision `c26cdd9d4a2a61886924fc9b081f0480f24b4108`: `remove_noise` 132–163,
  `adaptive_sub_cluster_sizing` 258–302, `within_cluster` 304–355.
  Cites Nekooeimehr & Lai-Yuen, A-SUWO, ESWA 46 (2016), 405–416,
  [DOI](https://doi.org/10.1016/j.eswa.2015.10.031). Third-party implementation
  evidence only; no claim of verified primary-paper equivalence.
- **[AR1]** [AROSS project code](https://github.com/wangyz0915/AROSS/tree/b32cf18cff09a8d135dce2453806721887b6c222),
  revision `b32cf18cff09a8d135dce2453806721887b6c222`: README, `aros/aros.py`
  (`generate`, `calc_weight`), `aros/area.py` (`gen_safe_area`),
  `aros/generate.py` (`Gaussian_Generator_`, `filter`). Publication identity,
  reported benchmark design and author-paper linkage remain unverified.

Supplied PDFs inspected for this proposal, SHA-256:

```text
UGO: 87cbd8f4237a19fd68e34cd8d57e664d6bc47adc0cb8e8ce9eedc1da5c10bc8b
SLEO: 2a98334c90e86472d3a53e992bc5537870bfe32306f9a5f55a6cf6f7849305e4
Hamilton: c9e8df9d65783041fa325bff3c08d28449fda3830d346483db52b56621d98f5c
Ma-Zhang: 41ddb9ceaa56f51840a733f9571eb282d089f2ae2b34e0d6af2c8b4bf1ee18af
GDEO: c7dda7d02022b29aba7a4289a3da13a0df35a3d6955d2c4b44a20263728c1be5
```

PDFs, downloaded source checkouts and long extracts remain outside Git. Open
questions are primary-source access for the unresolved competitors, the
operational meaning/cost of real subtypes, faithful runnable comparator
adapters, and whether the proposed independent diagnostic shows any useful
performance gap at all. None requires inventing or requesting an old proposal.
