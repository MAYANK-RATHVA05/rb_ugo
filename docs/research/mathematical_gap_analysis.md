# Stage 4A: mathematical gap analysis

## Evidence boundary and problem formulation

The six requested publications cannot presently be verified even from abstracts.
The [prior-art matrix](prior_art_matrix.md) distinguishes that missing evidence
from three verified additional full texts and pinned library sources. Citations
R/D/I and S1–S7 below refer to its linked reference register. Equations explicitly
called “audit derivations” are our reasoning, not newly discovered paper equations
or claims that an inaccessible paper contains them. No experiments were run.

The operational question is: **for a fixed learner, generator, fitting set and
evaluation distribution, which proposed local augmentation improves confirmed
discrimination beyond threshold changes, weighting, and matched-count controls?**
It is not “which regions intrinsically need synthetic data?” Data value depends on
the learner, metric and other data, as explicitly formulated in [D, §§1–2, 6](prior_art_matrix.md#d-ghorbani--zou-2019-data-shapley-equitable-valuation-of-data-for-machine-learning).
Training influence depends on curvature and optimization assumptions [I, §2].
Validation-guided weighting is already an objective in [R, §3.1].

Let F be labelled fitting data, H independent threshold/calibration-selection
data, A independent local audit/selection data, and O untouched outer-test data.
U is a separately declared unlabelled fitting pool. O's features as well as its
labels are excluded from graph construction, preprocessing, generation and
selection under this inductive protocol. Any later transductive experiment must
be labelled separately and cannot replace the inductive evaluation [S3, S4].

Define regions G_r using F (and U if declared) before inspecting A or O.
These can be fixed domain strata or a training-fitted partition; clustering is
a measurement tool, not the claimed contribution. Specify whether G_r spans
both classes or only indexes minority anchors. Record n_1r, the number of labelled
fitting minority anchors in each region, but never equate a small n_1r with an
identified population deficit. Region assignments for new observations must be
defined from features alone.

## 1. Threshold error is different from discrimination error

Let s(x) be a minority score and d_tau(x)=1{s(x)>=tau}. Define:

\[
R_{bal}(s,\tau)=\tfrac12\left[
P(s(X)<\tau\mid Y=1)+P(s(X)\ge\tau\mid Y=0)\right].
\]

This is 1 minus balanced accuracy, whose equation is verified in [S2]. It is not
the same as prevalence-weighted total error. With class prevalence pi and
class-conditional densities p_1,p_0, a direct audit derivation gives the balanced
Bayes rule p_1(x)>p_0(x). If eta(x)=P(Y=1|X=x) is calibrated under that prevalence,
the equivalent threshold is eta(x)>pi, not necessarily 0.5.

[S1] explicitly distinguishes score estimation from threshold choice and states
that threshold tuning leaves ROC/PR curves unchanged. Thus better recall at one
threshold need not imply better discrimination or new feature information.

**Counterexample (audit construction, not an experiment).** Suppose all true
minorities have score 0.4 and all majorities 0.1. At threshold 0.5 minority recall
is zero, yet ranking is perfect; threshold 0.25 fixes all decisions without
oversampling. High labelled MMA at 0.4 is therefore insufficient to diagnose a
need for additional samples. In practice these score/label patterns must be
estimated independently, not known by an oracle.

For a fixed set of scored observations, AP is:

\[
AP=\sum_j (Recall_j-Recall_{j-1})Precision_j.
\]

This is the verified non-interpolated [S2] definition. Do not call it trapezoidal
PR-AUC. A strictly increasing score transformation preserves ordering and score
ties, and therefore preserves AP on that set; this follows directly from the
definition. Changing class prevalence can change precision and AP even with the
same class-conditional ranking, so compare on the same natural-prevalence
evaluation observations. AP is threshold-independent but neither calibration nor
a causal measure of sample usefulness. Report a declared AP implementation.

## 2. Oversampling can imitate weighting or threshold shifts

For positive class weights w_1,w_0, pointwise minimization of population weighted
binary log loss gives this audit derivation:

\[
\eta_w(x)=\frac{w_1\eta(x)}{w_1\eta(x)+w_0(1-\eta(x))}.
\]

It is strictly increasing in eta. Thresholding eta_w at 0.5 is equivalent to
thresholding eta at w_0/(w_0+w_1). Under this unrestricted, well-specified
population model, weighting changes decisions but not discrimination. This does
**not** imply that finite, regularized or misspecified classifiers respond to
weighting only through an intercept shift. The distinction requires the tuned
threshold and weighting controls [S1; R, Eqs. (1)–(2)].

For ROS with actual integer multiplicities m_i, the augmented empirical objective
can be written exactly as:

\[
J(\theta)=\frac{\sum_i m_i\ell_i(\theta)}{\sum_i m_i}
+\lambda\Omega(\theta).
\]

This is an audit identity, not a general guarantee about all training code. A
weighted learner using those multiplicities and the same normalized objective
has the same objective. Solver randomness, minibatch sampling and the library's
regularization/sample-weight normalization can break apparent implementation
equivalence. For nonlinear seed selection, matching only global class weights
does not match pointwise multiplicities. Compare region ROS and matched
sample-weight multiplicities where supported, not just class_weight='balanced'.
ROS copying is verified in [S5]. Reweighting via validation is published in [R].

## 3. Synthetic samples are not independently observed minority information

Consider a generator S drawn from a kernel K(S|F,xi), with randomness independent
of an unknown population parameter theta once F is given. An audit factorization
shows:

\[
p(\theta\mid F,S)\propto p(\theta,F)K(S\mid F)
\quad\Longrightarrow\quad p(\theta\mid F,S)=p(\theta\mid F).
\]

The synthetic sample conveys no additional independent population evidence
conditional on its inputs. If U is used, replace F by (F,U); the same statement
holds conditional on both. This argument assumes the generator has no additional
external observations or theta-dependent oracle. A useful generator can still
change the learner's weighting, optimization or inductive bias. Its benefit
must not be described as acquiring new genuine labels. [D, §4.1] studies actual
data acquisition, which is different from interpolating already observed labels;
[S5] describes interpolation and copying mechanisms.

An optional real-labelled acquisition control would separate “a region benefits
from new observed data” from “this synthetic intervention helps this learner.”
In future controlled experiments, reserve that oracle only as a diagnostic arm,
never make its labels available to the proposed procedure.

## 4. High uncertainty and sparse regions have multiple explanations

The supplied/local MMA is 1-2P_hat(true|x); its implementation and unverified
paper alignment are recorded in [the Stage 3 document](../ugo_reproduction.md).
Large values can be consistent with miscalibration, class-prior bias, model
misspecification, noisy labels, underrepresentation or genuine overlap. A low
estimated density can describe a rare coherent mode or an isolated outlier.
No algebraic implication maps either score to a positive augmentation effect.
Verified comparators already emphasize kNN difficulty (ADASYN), boundary points,
or cluster sparsity [S5, S6].

**Irreducible-overlap counterexample (audit derivation).** With balanced evaluation
densities, the optimal expected error is:

\[
R^*_{bal}=\tfrac12\int\min(p_0(x),p_1(x))\,dx.
\]

This follows by choosing the smaller class-specific error contribution at each x.
If p_0=p_1 everywhere, the optimum is 0.5, regardless of sample count. Increasing
predicted positives raises recall and FPR together. Oversampling cannot lower
this true optimum; apparent improvement on a reused selection set can be noise.
An overlapping learner's finite observed failure does not prove Bayes ambiguity:
the model itself may be wrong. The formula is a reference for known synthetic
generating distributions, not an estimator or a claimed theorem from the six
inaccessible papers.

Intra-class imbalance therefore needs **both representation measurement and
confirmed intervention benefit**. Counts, minority-subregion recall and model
influence are complementary measurements, not independent proofs. Group data
value is already explicit in [D, §§3.2, 4.5]; scarcity weighting is implemented
in [S6]. A partition plus UGO does not establish novelty.

## 5. The identifiable target is a conditional intervention contrast

For region r, a fixed generator g, batch size m, learner A, and paired seed xi:

\[
f_0=A(F;\xi),\qquad f_{r,m}=A(F\cup S_{r,m}(F,U;\xi);\xi).
\]

Use H, separately from A, to tune each model's threshold by the same rule.
Candidate comparisons can estimate:

\[
\Delta_R=R_{bal}(f_0,\tau_0)-R_{bal}(f_{r,m},\tau_{r,m}),\quad
\Delta_{AP}=AP(f_{r,m})-AP(f_0),\quad
h_0=FPR(f_{r,m},\tau_{r,m})-FPR(f_0,\tau_0).
\]

For local ranking define, explicitly, a minority-region versus **global-majority**
comparison:

\[
AUC_r(s)=P(s(X_1)>s(X_0)\mid X_1\in G_r)
+\tfrac12P(s(X_1)=s(X_0)\mid X_1\in G_r).
\]

X_1 and X_0 are independent class-conditional draws. This is not within-region
overlap estimation; a separate conditional-majority FPR_r needs majority audit
observations in G_r. Missing class/region observations make that estimate undefined,
not zero. [I, §2] and [D, §§2–3] are prior art for conditional learner benefit;
finite synthetic interventions still need actual retraining and confirmation.

The output categories must be cautious:

- **Threshold-compatible explanation:** tuning without augmentation improves
  decisions, with no confirmed extra ranking benefit. This does not prove that
  all useful augmentation is impossible.
- **Supported benefit for this intervention:** independent comparisons show the
  declared incremental effect beyond controls. It does not prove intrinsic
  regional data limitation, safe labels or benefit with another learner.
- **Observed harm for this intervention:** independently confirmed risk/FPR harm.
  It does not identify irreducible overlap as the unique cause.
- **Inconclusive:** insufficient support or precision. Failure to reject no
  benefit is not proof of no benefit; it is especially common for small disjuncts.

The detailed proposal and bounds are in [candidate_hypotheses.md](candidate_hypotheses.md).

## 6. What unlabelled X can and cannot tell us

U can estimate feature-space support, connectivity, relative mass and potential
distribution shift. It cannot by itself identify eta(x), because:

\[
p_X(x)=\pi p_1(x)+(1-\pi)p_0(x)
\]

does not uniquely determine the two class-conditionals. As an audit construction,
let X be uniform on [0,1] and pi=0.1. One compatible model has Y independent of
X with eta=0.1; another sets Y=1 on a subset of length 0.1. Both have identical
unlabelled distributions and prevalence, with entirely different class support.
Finite observed anchors constrain sampled locations but do not identify the
unsampled label function without assumptions. This construction explains why
[S3] explicitly requires assumptions for semi-supervised gains.

Potential consistency measures include distances/paths to labelled minority
anchors, graph-harmonic agreement, instability under graph perturbations, and
minority-versus-majority anchor competition. These are **structural diagnostics**,
not verified membership probabilities. They require meaningful training-fitted
distances, representative anchors and some label smoothness/cluster alignment
[S3, S7]. In overlap, dense majority bridges, hubness, label noise, unanchored
components or distribution shift, those assumptions can fail. High confident
pseudo-labels may then repeat the classifier's error; confidence is not ground
truth. Self-training as an existing pseudo-label mechanism is verified in [S3].

To test actual value, compare the same fitting labels and compute budget with:
no U, same-distribution U, increasingly larger U, shifted U, feature-destroyed U,
and a supervised/structural control. Artificially hide labels only inside the
fitting allocation and forbid all methods access to those hidden truths. Use
known true labels only in a clearly marked diagnostic arm. Compare graph-only
semi-supervised learning and verified SSHR when its source is obtained. A
feature-only mechanism should be invariant to permuting unavailable U labels;
otherwise labels have leaked. Report both rejected helpful candidates and accepted
harmful ones, not only support scores that look plausible [S3; D, §4].

## 7. Required controls and leakage discipline

These are proposed evaluation requirements, not executed experimental findings.

| Control | Why it is necessary | Evidence/design basis |
| --- | --- | --- |
| No resampling, default threshold | Anchors the ordinary learner's starting point | S1 |
| No resampling, tuned threshold | Separates action-threshold errors from ranking errors | S1; derivation §1 |
| Class weighting, tuned threshold | Tests cheap prior/cost reweighting explanations | R; derivation §2 |
| Original UGO | Separates proposed selection from classifier-guided baseline | Source verification still required; preserve current code unchanged |
| Standard SMOTE | Tests whether uncertainty/region machinery beats ordinary interpolation | S5 |
| Validation-tuned sample count | Tests generic search over amount, with matched search budget | S4; R |
| SSHR and other relevant SSL baselines | Tests whether U adds more than existing SSL | SSHR method unverified; S3 supports self-training/graph comparators |
| Unlabelled-data ablations | Tests whether U contributes beyond anchors/compute | S3; non-identification construction §6 |
| Known-overlap controlled distributions | Distinguishes rare clean modes, noisy islands, bridges and irreducible overlap | Audit Bayes reference §4 |
| Minority-subregion recall | Detects global averages hiding neglected small regions | D's group valuation; predeclared measurement |
| Untouched outer-test evaluation | Prevents selected validation improvements from being mistaken for generalization | S4; D, §4 |

Also include KMeansSMOTE/ADASYN, matched point-weight ROS, and a group-valuation or
influence-guided control when computationally feasible [S5, S6, R, D, I]. Use the
same learner, preprocessing opportunities, threshold tuning, splits, U pool,
random seeds, and comparable tuning/fit budgets. Include region interventions
chosen randomly or by counts/uncertainty/density to assess the added diagnostic.
Do not deprive no-resampling controls of labels used by the proposed method:
include a supervised control that fits all allowed inner labelled data while
keeping its own independent threshold selection.
That full-label-budget control must be assessed on fresh confirmation/outer data;
if it fits A, it cannot be compared on A under the candidate's independence bound.

Preprocessing, region definitions and generator neighborhoods fit on F only;
H/A can guide decisions but never become synthetic anchors. For final outer
evaluation, any permitted refitting protocol must be identical across methods
and predeclared. If regions/preprocessing are refitted on additional inner data,
this creates a new procedure; inner guarantees for the earlier frozen models
do not automatically transfer.

## 8. Repeated validation can manufacture apparent utility

Use outer evaluation containing the **entire** region-search, threshold tuning,
augmentation selection and refitting procedure. Inside each outer-training
allocation, separate F, H and A or use appropriately nested cross-fitting [S4].
Predeclare candidate regions, generator policies, counts, metrics and selection
limits. Candidate predictions/thresholds must not depend on A before a fixed
finite-family bound is applied. Adaptive candidates produced after looking at
A cannot be treated as an independent fixed family. Either use fresh audit
blocks with a declared error-spending schedule or label reused-set analysis
exploratory and confirm on new independent inner data. The outer set is not
that debugging/confirmation resource.

AP and all-pairs AUC have dependent/non-additive contributions; do not apply a
per-sample independent Bernoulli confidence formula to them. Across overlapping
CV folds, repetitions are also not independent. Resample the appropriate
experimental unit (patient/group/dataset), report subgroup counts, and evaluate
the selected **procedure**, not only the winning inner candidate. Predeclare
practical effect margins; wide intervals produce “inconclusive,” not a favourable
safe/beneficial designation. [D, §4] separates valuation from final reporting;
[R]'s validation-objective convergence is not a substitute for this separation.

## Conclusion

There is a defensible **question** about conditional synthetic utility beyond
threshold/weight/count controls. There is not yet a defensible claim that local
data value, validation-guided reweighting, graph support, or their combination is
novel. The verified prior art already addresses much of that conceptual space
[R, D, I, S3, S5–S7]. The six missing publications and original-UGO fidelity must
be resolved before a publishable-gap claim or implementation decision.
