# Stage 4A: two falsifiable candidate mechanisms

These are mathematical proposals for review, not implemented algorithms or
established novel contributions. Citations [R], [D], [I], [S1]–[S7] refer to the
[verified reference register](PRIOR_ART_MATRIX.md#verified-reference-register-and-audit-limits).
The six requested papers remain UNVERIFIED; either may already cover an apparent
difference. No new method name or combined heuristic product is proposed.

DRA-UGO is the provisional project name, not a new name for either mechanism.
Both candidates must answer A/B/C separately; neither provides an intrinsic
remedy decomposition. That specific competing literature remains unverified.

## Candidate 1: control-adjusted local augmentation contrasts

### Definition, input and output

Input: disjoint labelled fitting F, threshold/selection H and audit A sets;
optional unlabelled fitting U; training-defined regions G_r; a fixed learner;
a finite, predeclared set of synthetic interventions a=(r,g,m,seed); and matched
controls. Neither A nor any outer-test observation is a generation anchor.
Each candidate is built from F (and declared U); its threshold is chosen on H.
All scored candidates must be fixed independently of A before the audit.

For each a, define the control set C_a to include:

1. No augmentation with an H-tuned threshold.
2. Class weighting with the same threshold-selection opportunity.
3. Region ROS or equivalent normalized point-multiplicity weights, when supported.
4. Same-count ordinary/random-region synthetic generation.
5. A generator whose sample count was selected on H rather than by region utility.

All use the same fitting labels, learner/search budget and meaningful regularization
normalization. Published comparison targets are validation reweighting [R],
group data valuation [D], and training influence [I]; threshold/ordinary sampling
mechanisms are verified in [S1, S5, S6]. Full retraining is used here rather than
claiming an influence approximation exactly predicts a finite batch.

For candidate a and each control j, calculate on A:

$$
\widehat\Delta_{R,a,j}=\tfrac12\sum_{y=0}^1
\frac1{n_y}\sum_{i:Y_i=y}
\left[1\{d_j(X_i)\ne Y_i\}-1\{d_a(X_i)\ne Y_i\}\right].
$$

For each region, preselect independent class-conditional pairs
$(X^+_{r,k},X^-_k)$, each audit observation used at most once **within** that
contrast. Pairing uses a fixed random seed, not scores. Define
$\psi(s;x^+,x^-)=1\{s(x^+)>s(x^-)\}+\frac12 1\{s(x^+)=s(x^-)\}$ and:

$$
\widehat\Delta_{Q,a,j}=
\frac1{P_r}\sum_{k=1}^{P_r}
[\psi(s_a;X^+_{r,k},X^-_k)-\psi(s_j;X^+_{r,k},X^-_k)],
\qquad P_r=\min(n_{1r},n_0).
$$

This estimates minority-region versus global-majority ranking benefit. It is not
a claim to estimate local Bayes overlap. The disjoint-pair definition sacrifices
power for a transparent independent-unit analysis; ordinary all-pairs AUC cannot
be treated as P_r*n_0 independent trials. Negative observations may be reused
across different hypotheses; the union bound below does not require independent
hypotheses. Global AP [S2] is a supplementary descriptive ranking measure, not
the target of the elementary confidence formula below.

Against a predeclared tuned no-augmentation reference b, define majority harm:

$$
\widehat h_{a,b}=
\frac1{n_0}\sum_{i:Y_i=0}[d_a(X_i)-d_b(X_i)].
$$

Let J count all predeclared scalar means audited, including class-specific risk
differences, pairwise-ranking differences, harm contrasts and any threshold-only
contrasts. For delta in (0,1), use:

$$
b(n)=\sqrt{2\log(2J/\delta)/n},\quad
b_R=\tfrac12[b(n_1)+b(n_0)],\quad
b_Q=b(P_r),\quad b_h=b(n_0).
$$

These are **audit-derived conservative bounds**, not claimed novel statistics or
equations attributed to any inaccessible paper. Each summand difference lies
in [-1,1]. For n independent such differences, the bounded exponential-moment
inequality gives
$P(|\bar Z-EZ|>e)\le2\exp(-ne^2/2)$. Substituting b(n) bounds failure by
delta/J for each scalar mean; a union bound gives simultaneous coverage at least
1-delta. Weighting the two class-mean bounds by 1/2 gives b_R. Conditional on F,H,
fixed candidates and class/region counts, this applies to independent audit draws.
It does **not** extend to models adaptively built using the same A labels.
The proof uses only bounded variables and independent sampling; S4 provides the
separate-selection/evaluation design rationale, not this particular equation.

Predeclare practical margins epsilon_R>0, epsilon_Q>0 and acceptable FPR increase
epsilon_h>=0. A region/candidate receives **supported incremental benefit** only if:

$$
\min_{j\in C_a}(\widehat\Delta_{R,a,j}-b_R)>\epsilon_R,\quad
\min_{j\in C_a}(\widehat\Delta_{Q,a,j}-b_Q)>\epsilon_Q,\quad
\widehat h_{a,b}+b_h\le\epsilon_h.
$$

These are separate contrasts/constraints, not multiplied heuristic scores.
The policy does not tune only an oversampling amount: it attempts to distinguish
ranking benefit beyond controlled weighting/location/threshold explanations.
However, that extra evaluation discipline alone does not establish a new
oversampling algorithm. The same finite-family analysis can apply to an ordinary
hyperparameter search. Choosing among accepted candidates needs a predeclared
rule (e.g. smallest added count, then fixed region order), not another look at O.

Output: risk/ranking/harm contrasts, uncertainty intervals, minority counts, and
one of **supported benefit / observed harm / threshold-compatible explanation /
inconclusive**. A threshold-compatible explanation requires a separately confirmed
default-to-tuned baseline decision benefit, not merely a non-significant ranking
contrast. “Observed harm” needs its own confirmed adverse contrast. Nothing is
labelled irreducible or intrinsically data-limited from these observations.
Missing classes/region observations make the output inconclusive. Bounds wider
than the possible effect also force inconclusive; they must not be clipped into
apparently informative guarantees.

### Uncertainty, locations, subregions and U: proposed operational specification

To make the intervention family concrete, one **ordinary proposal generator**
can use a clone of a probability learner fitted on F. For a labelled minority
anchor i, define $u_i=1-2s_F(1\mid x_i)$; the local repository implements the
other-minus-true form in [src/uncertainty.py](../../src/uncertainty.py). This is
signed disagreement, not verified author UGO fidelity or epistemic/aleatoric
uncertainty. Its in-fitting optimism is a limitation; a fitting-only cross-fit
score is a separately prespecified variant, never an audit-label estimate.

Let E_{r,k} be the min(k,n_1r) highest-u original minority anchors in a frozen G_r, with
row index breaking ties. Fix k, neighbor count, m and seed independently of A.
For a declared interpolation arm, draw i uniformly from E_{r,k}, j uniformly
from its nearest original minority neighbors within G_r, and lambda uniformly
on [0,1); generate $z=(1-\lambda)x_i+\lambda x_j$ with assigned label 1.
Do not use synthetic rows or U pseudo-labels as observed minority anchors.
Insufficient anchors/neighbors make that arm infeasible; do not silently switch
to copying. A separate copying/point-weight arm is an explicit matched control.
This is an existing interpolation ingredient [S5], not the contribution.
Original UGO runs unchanged as a comparator and may choose the majority class;
this proposed minority-region arm does not redefine its selection rule.

Uncertainty thus proposes a finite family, while independent risk/ranking/harm
contrasts accept, reject or leave an intervention inconclusive. It is **not** an
entropy/density product or an assumption that high-u regions need more data.
A predeclared region budget tests each eligible rare/dense region with reported
fitting/audit counts; allocation can only follow confirmed effects under a fixed
selection rule. It must not guarantee equal recall or force synthesis in a
rare region without evidence. A global gate can miss small useful disjuncts,
so report regional effects even when no policy is accepted.

Optional U enters only a fitting-defined structural proposal/veto, for example
Candidate 2 below. It contributes no labels to u_i, thresholds or the audit.
Compare that arm against identical anchors/generation without U and ordinary
SSL/support filters at matched compute. The U extension is optional and must
not be counted as evidence from an additional independent candidate unless
prespecified in the finite family. No generated point is declared correctly
labelled by either acceptance test. A/B are conditional diagnostic targets;
C requires separate U incremental-value experiments [DATASET_PLAN.md](DATASET_PLAN.md).

### Assumptions and interpretation

- F,H,A have independent relevant sampling units and matching evaluation targets;
  use group/time-aware splits when iid observations are inappropriate [S4].
- The full candidate family and prediction/threshold functions are independent of
  A before its inspection. A selected result may be confirmed with a new inner
  block; all final generalization estimates still use untouched O.
- Region definitions are training-only and minority-region evaluation has enough
  genuinely labelled observations. U supplies no audit labels [S3].
- The learner and generator define the effect. Paired random seeds reduce one
  source of variation but do not supply population guarantees across arbitrary
  training seeds. Assess training variation separately; do not treat repeated
  predictions on the same A as additional independent labelled observations.
- Thresholds, weights, batch counts, masks and margins must be predeclared or
  selected outside A. Global/local AP bootstrap analysis remains exploratory
  unless separately justified; the above bound is for the explicitly paired metric.

### Difference from and proximity to prior art

The prospective difference is a **matched finite-intervention diagnostic** with
ranking benefit beyond threshold/point-weight/count controls and explicit
inconclusive outputs. Ren's validation objective already learns which examples
help; influence functions already estimate impact; Data Shapley already values
groups and suggests acquisition [R §§3.1–3.4; I §§2,7; D §§3.2,4.1,4.5].
Closest published competitor: **group Data Shapley**, with validation reweighting
as the closest training-policy comparator. This is not proven disjoint from
SLEO/SSHR/GDEO or the other inaccessible papers. Do not claim novelty from local
scope, confidence intervals, a FPR constraint, or the three-way interpretation
without a focused additional literature audit.

### Counterexample, falsification and complexity

**Counterexample:** a calibrated/reordered-by-prior score with perfect ranking
but the wrong default threshold can show large recall gain after ROS. The tuned
baseline already fixes it; the ranking contrast is zero. No synthetic-information
contribution should be accepted. Likewise, exact ROS/sample-weight objective
equivalence defeats a supposed discrimination gain attributable to new information
([gap analysis §§1–3](MATHEMATICAL_GAP_ANALYSIS.md)). A region with few minority
audit labels may be useful yet never pass the bound: abstention is not evidence
of irreducibility. Interactions also mean two individually helpful regional batches
need not be helpful jointly [D's coalition-value formulation].

**Falsifiable hypothesis:** conditional regional utility estimates predict
independent finite-intervention ranking gains more reliably than counts, sparsity,
or local uncertainty, and a predeclared selection procedure improves the declared
outer risk/ranking target beyond matched controls without exceeding its majority
harm margin. Reject the usefulness claim if estimates fail to replicate, gains
are explained by tuned thresholds/weights/count search, harm exceeds the margin,
or intervals are generally uninformative at realistic label budgets. No such
result has yet been observed.

For R regions, M counts, G generators and B paired training seeds, K=RMGB
interventions. If C controls are fitted per intervention, a conservative cost is
O(K(C+1)[T_fit(N+m,d)+T_score(n_H+n_A,d)+n_H log n_H]) plus generator cost.
For the concrete proposal, brute-force within-region minority distances add
O(sum_r n_1r^2 d), at most O(n_1^2 d) for a partition, with corresponding
quadratic stored distances; ranking seeds costs O(n_1 log n_1). Reuse across
fixed counts/seeds is permissible when the fitting anchors stay fixed.
Computing AP adds O(n_A log n_A) per scored model; disjoint local pair comparisons
are O(P_r) per contrast. These costs can be lower with shared fixed controls, but
matching stochastic fits/normalization still matters. Storage for candidate
predictions is O(K(C+1)n_A), or less if processed sequentially.

Realistic in scikit-learn: **yes for a small, predeclared study**, using cloned
estimators, sample_weight where supported, threshold tuning, probability/decision
scores and independent splits. This is a future external research harness; it
must not modify the unchanged UGO baseline. This stage grants no implementation
approval.

**Decision:** retain only as an empirical diagnostic hypothesis. Reject a present
claim that it is a novel oversampling method. It is too close to established data
valuation/validation selection until a genuine methodological difference is shown.

## Candidate 2: sensitivity intervals for anchor-compatible graph support

### Exact definition, inputs and outputs

Input: numerical training features for labelled F, unlabelled U and candidate
locations z generated from F; feature-only, training-fitted symmetric nonnegative
graph weights w_ij; labelled anchor constraints; and a predeclared energy radius
rho. No pseudo-label is clamped as if it were an observed label. No H/A/O rows
are used as graph nodes or generation anchors.

Let q_i in [0,1] be an **anchor-consistency potential**, not a calibrated class
probability. On genuine labelled anchors, q_i=y_i. Let L be the weighted graph
Laplacian and define:

$$
E(q)=q^\top Lq=\sum_{i<j}w_{ij}(q_i-q_j)^2,\qquad
\mathcal Q_\rho=\{q\in[0,1]^N:q_{labelled}=y_{labelled},\ E(q)\le\rho\}.
$$

The exact proposed structural interval at z is:

$$
\underline q_z=\inf_{q\in\mathcal Q_\rho}q_z,\qquad
\overline q_z=\sup_{q\in\mathcal Q_\rho}q_z.
$$

If the set is empty, report inconsistent assumptions. If z has no anchor-connected
component, report [0,1]. A proposed structural veto rejects an intended minority
location when its lower consistency bound is below a predeclared beta; it never
asserts that a surviving point has a genuine known minority label.

For a graph where every unlabelled component connects to an anchor, L_uu is
positive definite and the unconstrained harmonic extension is
$h_u=-L_{uu}^{-1}L_{ul}y_l$. For these nonnegative graph weights it lies in
[0,1]. Let delta=rho-E(h)>=0. Completing the square gives the audit identity:

$$
E(q)=E(h)+(q_u-h_u)^\top L_{uu}(q_u-h_u),\quad
|q_z-h_z|\le\sqrt{\delta(L_{uu}^{-1})_{zz}}.
$$

The second bound follows from Cauchy–Schwarz in the L_uu norm. Without box
constraints, both extremes are attained along L_uu^{-1}e_z. With box constraints,
clipping this interval to [0,1] gives a **conservative containing interval**, not
necessarily the exact optimized interval. Computing the exact box-constrained
extrema requires convex optimization. These are proposed sensitivity calculations,
not equations attributed to an inaccessible graph paper or a new theorem claim.

### Uncertainty, location rejection, intra-class imbalance and U

For an exact possible feature-graph specification, first fit scaling on F (and
declared U only if the entire protocol permits it). With fixed k and ell>0,
let N_k(i) exclude i and resolve distance ties by index, and set

$$
w_{ij}=\exp[-\|x_i-x_j\|^2/(2\ell^2)]
1\{j\in N_k(i)\text{ or }i\in N_k(j)\},\qquad w_{ii}=0.
$$

This symmetric nonnegative graph defines L above. Freeze bandwidth/neighbor
selection outside A; changing them after audit inspection invalidates a fixed
candidate-family claim. Ordinary graph constructions are established [S3, S7].
Candidate locations are the fitting-minority interpolation proposals specified
for Candidate 1, or another completely predeclared generator; the graph does
not create independent minority anchors.

The interval width measures **sensitivity of anchor-compatible fields**, not
posterior confidence or a formal aleatoric/epistemic decomposition. Local
classifier disagreement can propose which locations to examine, but does not
enter the energy as an invented score weight. Accept for structural review only
when the exact lower bound exceeds beta, or when the conservative containing
interval's lower endpoint does so. Failing that conservative check is an
uncertain structural rejection, not proof of majority membership. High overlap
requires independent harm/discrimination confirmation regardless of the field.

U supplies graph nodes/connectivity, not clamped labels. Minority subgroups are
examined separately with anchor counts and rejection rates; a poorly connected
rare group can get wide intervals and lose beneficial proposals. No automatic
sparsity compensation is asserted. Question A tests harmful-location rejection,
B tests lost rare-region coverage, and C tests actual U value versus the same
anchors/graph machinery without U. This mechanism cannot solve all three merely
by displaying a consistency score.

### Assumptions, comparison and limits

Graph distances must be meaningful after fitting-only preprocessing. Anchors
must represent relevant modes, and the smoothness radius must be chosen without
audit/test-label access. Relating q to actual minority membership would require
an additional, independently defensible class-smoothness/noise model: observed
Bernoulli labels at anchors are not generally the true eta values. In overlap,
hard anchor constraints need not describe the population posterior at all.
Treat q only as a structural consistency field unless those assumptions are
established. Unlabelled density alone cannot choose a truthful rho or beta.

Closest published competitor: **graph label spreading**, associated in verified
[S7] with Zhou et al., “Learning with local and global consistency” (2004).
Its original paper was not read; [S3, S7] verify implementation and attribution,
not author theorem statements. Harmonic/graph propagation is already implemented;
this proposed interval changes the question from one confident label to sensitivity
over compatible fields. That distinction does not prove novelty: robust graph
learning, harmonic-extension uncertainty and effective-resistance literature
still need direct source review. It also does not defeat a verified SLEO/SSHR
comparison, which remains unavailable.

### Counterexample, falsification and feasibility

**Counterexample:** a dense majority bridge between minority anchors can produce
high harmonic q along the bridge. With rho=E(h), the feasible field collapses to
h and the interval falsely looks certain about actual membership. The non-identifiability
construction in [gap analysis §6](MATHEMATICAL_GAP_ANALYSIS.md) permits the same
unlabelled structure to have opposite labels there. A wider radius may instead
reject a genuine sparse minority island. These are consequences of assumptions,
not guarantees of boundary safety.

**Falsifiable hypothesis:** compared with the same anchor/generator system without
U and with ordinary graph-posterior filtering, sensitivity-based structural
rejection reduces harmful accepted locations without suppressing beneficial rare
regions, on independently evaluated controlled distributions. Reject it if effects
disappear under matched graph/compute controls, U gives no incremental benefit,
shifted/overlapping U produces harm, or it can be reproduced by an ordinary
confidence threshold with no distinct advantage. No experiments support this yet.

With N graph nodes, d features and P locations, brute-force neighbor construction
costs O(N^2 d), with dense O(N^2) storage; sparse kNN storage is O(kN), though exact
high-dimensional construction can retain the quadratic worst case. Dense harmonic
solves/factorization cost O(N_u^3) and O(N_u^2) storage. Bound diagonals can be
obtained by solves for candidate coordinates after factorization; exact constrained
intervals require two optimization problems per location with solver-dependent
cost. Iterative sparse linear solves use O(t|E|) operations per right-hand side
for t iterations; convergence depends on conditioning. Changing the candidate
graph can require rebuilding these quantities.

Realistic in scikit-learn: **partly**. Neighbor graphs and preprocessing are
available; harmonic/sensitivity solves need NumPy/SciPy routines or constrained
optimization outside the standard LabelPropagation API. It is feasible only
for small studies with explicit disconnected/ill-conditioned handling.

**Decision:** reject as a standalone novel oversampling contribution. At most it
is an optional assumption-sensitivity ablation after source review, not a new
pseudo-label authority or proven safety mechanism.

## Excluded proposals and implementation gate

No third candidate is proposed. UGO plus clustering, pseudo-labels, entropy,
boundary distances, count tuning, or multiplied uncertainty/need/support scores
is rejected as the present novelty justification. Relevant ingredients are already
documented [S1, S3, S5, S6], and benefit-driven weighting/data valuation is verified
from full text [R, D, I]. Exact precedence in the six requested papers is still
unknown; this is not an exhaustive proof that every variant is published.

Neither candidate is approved for implementation. First obtain the missing
sources, reconcile original UGO, investigate the named closest literatures, and
pre-register a feasible labelled-validation budget and matched-control study.
Stage 4A ends with a research decision, not a new oversampler.
