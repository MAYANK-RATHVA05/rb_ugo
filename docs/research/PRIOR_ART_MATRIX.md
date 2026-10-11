# Stage 4A-R: full-text prior-art comparison

Audit date: 2026-10-11. Sources U/L/H/G/B/T below are the six supplied PDFs,
opened and text-extracted in this workspace. Citations give **PDF pages**, sections
and equation/table identifiers. This is a targeted method audit, not an exhaustive
literature search or an independent reproduction of reported experiments.
[REFERENCE_STATUS.md](REFERENCE_STATUS.md) gives exact filenames, metadata and
read scope; [resource inventory](../../data/manifests/resource_inventory.csv)
gives accessible paths, size, SHA-256 and parse status. PDFs stay ignored by Git.
Earlier six-paper UNVERIFIED cells are superseded by the evidence below.

## Core sources and verified scope

| ID | Verified printed bibliography / citation | Full-text sections used |
| --- | --- | --- |
| U | Sağlam & Cengiz, *Uncertainty guided oversampling: A classifier-specific approach for effective handling of class imbalance in machine learning*. KBS 342 (2026) 115886. [DOI](https://doi.org/10.1016/j.knosys.2026.115886) | pp5–8 §§3.1–3.4, Tables 2–4; setup pp8–10 |
| L | Tao, Wang & Wang, *Semi-supervised local entropy-decayed oversampling for imbalanced data*. KBS 333 (2026) 115009. [DOI](https://doi.org/10.1016/j.knosys.2025.115009) | pp4–7 method, Eqs.1–8 and Algorithm 1; printed year 2026, despite DOI 2025 |
| H | Jiang, Zhao, Lu, Zhan & Mao, *A semi-supervised resampling method for class-imbalanced learning*. ESWA 221 (2023) 119733. [DOI](https://doi.org/10.1016/j.eswa.2023.119733) | pp4–6 SSHC/SSHR algorithms, equations; p7 setup; pp11–12 U ablation |
| G | Jiang, Zhu & Liu, *GDEO: Gini-guided discriminative equilibrium oversampling for imbalanced classification*. Neurocomputing 696 (2026) 134121. [DOI](https://doi.org/10.1016/j.neucom.2026.134121) | pp4–6 Eqs.1–9, Algorithm 1, complexity/setup |
| B | Hamilton, *Don't oversample the boundary: A remedy decomposition of the minority deficit in imbalanced learning*. Neurocomputing 704 (2026) 134831. [DOI](https://doi.org/10.1016/j.neucom.2026.134831) | pp4–6 §§3.1–3.5, Proposition 1; pp8–9 §§5.4–5.5; OOF qualification/Appendix B |
| T | Ma & Zhang, *Synthetic Augmentation in Imbalanced Learning: When It Helps, When It Hurts, and How Much to Add*. [arXiv:2601.16120v3](https://arxiv.org/abs/2601.16120v3), 7 March 2026 | Main text pp9–12 Eqs.1–4, Theorems 1–2 / Assumption 1; pp15–19 mismatch/rate claims; pp21–22 VTSS Algorithm 1. Not all 90 pages/supplement proofs audited |

Metadata are verified from the supplied documents, not a new publisher authenticity
check. Negative entries mean **not present/identified in the inspected algorithms
and protocol**, not proof of absence in every related work or supplement.

## Required mathematical comparison

| Source | Quantity optimized/estimated | Seeds / generation | Minority difficulty | Overlap | Intra-class imbalance |
| --- | --- | --- | --- | --- | --- |
| U | Signed MMA and class-mean equality stop [p6 §§3.2–3.3] | Highest mean-uncertainty class; top uncertain clean seeds, ROS/SMOTE [pp7–8] | Class-conditional classifier disagreement | kNN pre-noise exclusion and final relocation | Point emphasis, but no explicit subgroup allocation guarantee [§3.4] |
| L | Local entropy and radius-decayed cluster quantity [pp5–6 Eqs.3–5] | Minority cluster hyperspheres; entropy-based allocation and radial generation [Eqs.7–8] | Local mixed-neighbor entropy | Label-constrained merging and translating majority points out of hyperspheres [Eqs.1,6] | Explicit minority clustering and cluster budget [Eq.7] |
| H | SSHC labelled accuracy/F-measure objective; confidence for U [pp5–6 Eqs.3,7] | Select confident U in minority clusters; fallback within-cluster SMOTE [p5 Algorithm 2] | Cluster/subclass representation and purity | Split impure clusters; remove majority rows in minority clusters; hybrid undersampling | Class decomposition into clusters; minority U acquisition [Algorithms 1–2] |
| G | Gini uncertainty; kernel difference L minimized by gradient descent [p4 Eqs.1–7] | Gini-weighted minority seed, nearby majority partner, Gaussian start, descent [pp5–6] | Gini-weighted ambiguous anchors | Author claims equilibrium/no excessive overlap; no hard minority-posterior constraint in displayed update | Difficult seed weights, not an explicit small-disjunct coverage constraint |
| B | OOB/OOF error triage and threshold-recoverable / estimated data-reducible / estimated irreducible decomposition [pp4–5] | Category-targeted SMOTE-style matched-budget interventions [p8 §5.4]; paper evaluates a remedy menu | Low same-class local representation, classifier errors and threshold response | Estimated boundary errors; posterior interpolation analysis [p5 Proposition 1] | Local ratio normalized by global class prevalence; category 2 scarcity proxy |
| T | Balanced population loss, augmented risk bias/variance; CV balanced validation loss [pp9–12,22] | Abstract synthetic distribution P_syn / supplied generator; size grid | Statistical generator mismatch, curvature and bias, not a region detector | Mismatch can harm; no geometric no-crossing promise | Binary class-size theory; not a local disjunct allocation algorithm |

| Source | Separate U features? | Pseudo-labels? | Downstream influence | Marginal augmentation benefit | Tuned threshold control | Guarantees / limitations |
| --- | --- | --- | --- | --- | --- | --- |
| U | No: original labelled pool, oracle/U pool explicitly removed [p5 §3.1] | No separate-U pseudo-label stage | Yes, internal selected learner retrained/scored [§3.1]; not independent downstream validation | No finite independent benefit contrast; Welch equality heuristic | Not identified in inspected setup pp8–10 | No risk/no-overlap theorem in inspected method. Neighbor filtering only heuristically reduces aleatoric effects [pp6–7] |
| L | **No separate X_U input** in Algorithm 1 [p7]; semi-supervision means labelled class constraints on clustering | No selected U pseudo-label input | Probabilities determine entropy; no finite learner-response selection [pp5–7] | Not identified | Not identified in inspected method | Complexity estimate [p7], not a synthetic-class correctness guarantee; entropy/stopping signs require clarification |
| H | **Yes**, D_L,D_U in Algorithms 1–2 [pp4–5] | **Yes**, selected confident U assigned minority labels [pp5–6] | Cluster objective uses its own prediction/purity; no downstream SVM validation feedback in generation | No independent local finite-batch test; U ablation exists [pp11–12 §4.3.4] | Not identified; p7 uses SVM and resampling comparators | Cluster-smoothness heuristic, complexity argument [p6]; no posterior-membership guarantee; printed metrics and U-split provenance require resolution |
| G | No X_U input [p6 Algorithm 1] | No U pseudo-label stage | LR probability estimate by default; evaluation uses KNN/CART/NB/SVM [pp4,6], so not the downstream learner's finite benefit | Not identified | None identified; defaults used [p6] | Complexity [p6]; no theorem establishes no crossing for literal descent. Own counterexample below |
| B | No distinct X_U in inspected triage/intervention protocol | No separate-U acquisition | RF ensemble drives triage; classifier remedies evaluated [pp4–6] | **Yes at category level:** matched-budget category 2/3/all/random interventions [p8 §5.4]; not an oracle for a new region's causal benefit | **Yes**, cross-fitted BA threshold and all-method threshold parity; ranking controls [pp5,9] | Proposition 1 is posterior/geometric reasoning under continuity or affine/quasiconvex conditions; real-data “irreducible” is an estimate, not known Bayes truth |
| T | No distinct U-set requirement in main theory / VTSS inputs | Not a main-theory pseudo-label mechanism | Risk/loss/learner-specific; classifier fitted in each CV candidate [p22] | **Yes globally:** augmentation-size risk comparisons and CV search | No explicit threshold-tuned no-resampling arm in Algorithm 1; it targets balanced surrogate loss | Theorems require curvature, differentiability, boundedness, covariance and asymptotic/mismatch assumptions; not universal RF/SMOTE safety |

## Author equations and critical source issues

### U: classifier disagreement and stopping, not benefit identification

[U p6 §3.2] $U(x,y)=\max_{k\ne y}P(k\mid x)-P(y\mid x)$.
[U p7 §3.4] chooses $c=\arg\max_l\bar U_l$, then $n_{select}=n^{(c)}\lambda_{select}$,
with $n_{step}=n\lambda_{step}$ fixed before the loop [p8 Table 4].
Welch non-rejection [p6 §3.3] is not proof of equal risks or useful synthetic
information. Source cap conflict and implementation conventions are documented
operation by operation in [UGO_PAPER_CODE_AUDIT.md](UGO_PAPER_CODE_AUDIT.md).
No claim of a calibrated epistemic/aleatoric decomposition is justified: the
paper itself qualifies that interpretation [pp6–7].

### L: genuine subgroup mechanism, but no separate U pool

As printed [L p5 Eqs.3–5]:

$$E(x)=p\log p+(1-p)\log(1-p),\quad E(C)=\sum_{x\in C}E(x),\quad
\psi_j=E(C_j)\exp[-\alpha(r_j-r_0)].$$

Here p is a local minority-membership estimate: kNN, KDE or a classifier-based
estimate are explicitly allowed [L p5 §3.2]. Algorithm 1
[p7] takes X_maj,X_min,alpha,prob_type, not an extra unlabelled feature set.
Equation 1 [p4] limits minority cluster merges using nearest-majority distances.
Majority points inside a minority sphere are translated [p6 Eq.6]:

$$x'_{maj}=c+r\frac{x_{maj}-c}{\|x_{maj}-c\|}.$$

Equations 7–8 [p6] allocate $(n_{maj}-n_{min})E(C_j)/\sum_kE(C_k)$ and generate
$z=c+\xi r\mu/\|\mu\|$. **Critical audit interpretation:** Eq.3 lacks the usual
entropy minus sign (confirmed visually in the PDF); E<=0 conflicts with
Algorithm 1's positive-psi branch. Even positive E times an exponential never
reaches zero at a finite radius without tolerance. Uniform radius with a uniform
direction is not uniform volume in d>1 (ball volume within radius t scales t^d).
Do not silently insert a sign, tolerance or radial transform to claim reproduction.
The p7 approximate O(n log n m) claim treats cluster/overlap iterations as bounded
and does not include arbitrary probability-model fitting cost. It is not a
worst-case guarantee for every learner.

### H: real U selection, with unresolved protocol/metric details

[H p5 Eq.3] $J=\mu Acc+(1-\mu)F_1$ scores labelled SSHC predictions. Equation 4
prints $F_1=2TP\,TN/[TP(TP+FP)+TN(TP+FN)]$, which is not ordinary binary F1.
[H p6 Eq.7] ranks U in minority clusters:

$$C(y_i\mid x;\Theta)=\frac{n_+}{n}
\left(1-\frac{\|x-m_j\|}{\sum_k\|x-m_k\|}\right).$$

Here n_+/n is labelled cluster purity and m_j the associated centroid. Algorithms
1–2 [pp4–5] use real D_U; minority-labelled selected U is not newly observed truth.
The hybrid amounts are $n_U=\lambda(N_--N_+)$ (undersampling, p5 Eq.6) and
$n_O=(1-\lambda)(N_--N_+)$ (U additions, p6 Eq.8). SMOTE fills shortages.
An explicit no-U ablation is reported [pp11–12 §4.3.4], so “nobody tests U value”
is contradicted. These are author-reported experiments, not replicated here.

**Critical audit interpretation:** p6's printed specificity FP/(TN+FP) and
sensitivity FN/(TP+FN) are FPR/FNR; its displayed AUC formula is not ordinary
integrated ROC-AUC. The same page separately describes standard F-measure.
Resolve author code/metric conventions before a faithful implementation or
numeric comparison. The p7 setup delegates to CDBH and does not establish the
exact labelled/U masking and outer exclusion needed for our protocol; do not
invent a fraction or assume absence of leakage. The stated clustering cost is
$O(m(l+u)(t_1K_1|Y|+t_2K_2))$ [p6]; claimed resampling simplification neglects
potential l*n_U/u*n_O terms unless additional bounds apply. K_2 is a cluster
count subscript, not a square. No class-membership/no-overlap theorem follows.

### G: literal descent needs scrutiny

[G p4 Eqs.1–7; p5 Eqs.8–9]:

$$G(x)=2p(x)(1-p(x)),\quad
L(z)=G(x_+)e^{-\alpha\|z-x_+\|^2}-G(x_-)e^{-\alpha\|z-x_-\|^2},$$
$$\nabla L(z)=2\alpha G(x_-)e^{-\alpha\|z-x_-\|^2}(z-x_-)
-2\alpha G(x_+)e^{-\alpha\|z-x_+\|^2}(z-x_+),\qquad
z_{t+1}=z_t-\eta\nabla L(z_t).$$

Seed weight is G(x_i)/sum G(x_j); the start is x_+ plus Gaussian noise, a nearby
majority partner is selected and Algorithm 1 breaks at L<=0 [pp5–6].
**Our analytic counterexample, not a paper experiment:** take x_+=0,x_-=1,
alpha=1, equal G=.375, start 0 and eta=1. A compatible fitted score is
p_1(x)=sigmoid(log(3)(1-2x)), whose boundary is .5. The derivative at 0 is
-.75/e; descent moves toward the majority. Iterates are approximately .2759
and .789, crossing that classifier boundary; stopping at L<=0 can return the
crossed point. A small Gaussian perturbation retains this behavior by continuity.
Thus the literal update does not guarantee the claimed minority attraction or
no crossing. This challenges a general guarantee, not every reported experiment.

Equation 2 uses squared Euclidean distance; p6's complexity prose refers to
Mahalanobis distance. Treat as an unresolved inconsistency, not a license to
change the metric. The stated loop cost before simplification is roughly
O(n_++n_-+n_+(n_-d+n_-log n_-)+(n_--n_+)Td); LR fitting and distance-metric
estimation must be added. No published implementation was run here.

### B: direct novelty threat already covers the proposed framing

[B p4 §3.1 Eq.1] decomposes entropy of mean predictions into mean entropy plus
the residual disagreement. The gating protocol instead uses OOB/OOF predictions
[p4 §3.2]: noisy category if true-class probability <.12; then sparse category
if same-class kNN fraction/global class prevalence <.4; remaining errors are
boundary category. These thresholds are operational, not identified Bayes labels.

[B p5 §3.3] **first tests threshold recoverability**, using five-fold cross-fitted
balanced-accuracy thresholds. Remaining category 2 errors are estimated
data-reducible; remaining category 3 errors estimated irreducible. The paper
explicitly avoids claiming known Bayes irreducibility on real data. [B p8 §5.4]
then matches synthetic budget across category 2/category 3/all-error/random arms;
[B p9 §5.5] checks ROC/PR ranking and threshold parity. These are substantial
precedents for our proposed framing and intervention controls. Fine regional
contrasts are not automatically the same algorithm, but changing resolution or
adding a confidence gate is insufficient demonstrated novelty.

Proposition 1 [pp5–6] uses eta<.5 anchors: continuity ensures sufficiently short local
steps stay majority; whole-segment claims require quasiconvex/affine structure.
For affine eta with endpoints eta_i>.5>eta_j, crossing probability under uniform
interpolation is $(.5-\eta_j)/(\eta_i-\eta_j)$. This is posterior geometry,
not a general guarantee that all oversampling harms. Clause (iii) asserts a
nondecreasing harmful fraction as overlap/seed support shifts; the proof sketch
does not formalize an ordering across arbitrary changing distributions. Do not
promote that claim to a universal monotonicity theorem. Natural-prior eta=.5 and
a balanced Bayes boundary differ under unequal priors: p8's controlled Gaussian
balanced-boundary description must not be conflated with the natural posterior
criterion. Strict OOF robustness is qualified, not universal causal identification.

### T: mismatch theory and size tuning are already published

[T p9 Eq.1] uses balanced loss
$R(\theta)=\frac12E_{P_0}\ell_0+\frac12E_{P_1}\ell_1$.
For N=n_0+n_1+n_tilde, pi_0=n_0/N, pi_tilde=n_tilde/N, Eqs.2–4 [p10] give:

$$\widetilde R=R+(\pi_0-\tfrac12)\varphi+\widetilde\pi\psi,\quad
\varphi=E_{P_0}\ell_0-E_{P_1}\ell_1,\quad
\psi=E_{P_{syn}}\ell_1-E_{P_1}\ell_1,$$
$$b(\theta^*)=(\pi_0-\tfrac12)\nabla\varphi(\theta^*)+
\widetilde\pi\nabla\psi(\theta^*).$$

Theorem 1 [pp10–11] lower-bounds excess balanced risk by
$\mu\|b(\theta^*)\|^2/(2M^2)+o_P(1)$ under its boundedness, unique augmented
minimizer, gradient-Lipschitz and balanced quadratic-growth conditions.
Theorem 2 / Assumption 1 [pp11–12] give a local asymptotic bias/variance expansion
under smoothness, bounded support/moments, strong curvature and positive-definite
covariances. These do not imply finite-sample safe graph labels or universal
oversampling benefit. Supplemental proof completeness is not claimed.

[T p19 Eq.5/Theorem 6] local symmetry means **gradient condition**
$\nabla\varphi(\theta^*)=0$, not sparse feature density. Under Theorem 2's conditions
and $n_0^{-1/2}\ll\|\nabla\psi\|\ll1$, a diminishing size
$\widetilde n=O(\sqrt{n_0}/\|\nabla\psi\|)\ll n_0$ has balanced excess risk $O_P(n_0^{-1})$;
nonvanishing size $\widetilde n\gtrsim n_0$ has balanced excess-risk lower order
$\Omega_P(\|\nabla\psi\|^2)\gg n_0^{-1}$. The conditions must accompany these rates.

VTSS [p22 Algorithm 1] searches gamma in a positive grid centered on 1, sets
$\widetilde n=round[\gamma(n_0-n_1)]$, augments only each CV **training** subset,
fits the learner and minimizes mean class-balanced validation loss. The target
minority count is computed at the input dataset level, so fold-relative scaling
must be declared in a reproduction. Gamma=0 is not explicit in the displayed
grid; automatic avoidance of all augmentation harm is not guaranteed. This
already covers global validation-tuned sample amounts and learner-loss feedback.
It does not alone establish a new region-level intervention or genuine-U mechanism.

## Corrections to earlier research documentation

- Six inaccessible-source cells and prompt-only equations are replaced with full
  source evidence. Older Git versions preserve the access history.
- SLEO's title is not evidence of a separate U pool: Algorithm 1 has none.
- SSHR does use separate U and tests a no-U ablation; pseudo-label confidence is
  not verified membership. Its printed metric/protocol issues remain open.
- The threshold/data/irreducible framing and category-matched interventions are
  **already explicit in B**, not an unverified novelty opportunity.
- T already formalizes mismatch-dependent augmentation harm and validation size
  search. Ordinary count tuning is rejected as novelty.
- UGO's current class-count batch was a demonstrated deviation, now fixed; cap
  conflict is real, not merely an earlier prompt conjecture. See the code audit.
- G's verbal equilibrium/no-overlap account and L's printed entropy/stop signs
  cannot be promoted to verified guarantees without the qualifications above.

## Additional publications verified from full text

Three complete 10-page PDFs were obtained with HTTP 200 from **PMLR's public
GitHub repositories**. Their titles/authors match the official repository's
publication metadata. Relevant mathematical and experimental sections were read;
supplementary appendices/proofs were not comprehensively audited. PDF hashes are recorded below.
They provide additional direct limits on possible novelty claims.

### [R] Ren et al. (2018), Learning to Reweight Examples for Robust Deep Learning

Full-text source: [PMLR PDF mirror](https://raw.githubusercontent.com/mlresearch/v80/gh-pages/ren18a/ren18a.pdf).
[Official PMLR metadata and abstract](https://raw.githubusercontent.com/mlresearch/v80/gh-pages/_posts/2018-07-03-ren18a.md).

In Section 3.1, Eqs. (1)–(2), their bilevel objective is:

$$
\theta^*(w)=\arg\min_\theta\sum_{i=1}^{N}w_i f_i(\theta),\qquad
w^*=\arg\min_{w\ge0}\frac1M\sum_{j=1}^{M}f_j^v(\theta^*(w)).
$$

Their Eqs. (5), (7), and (8) use a one-step perturbed training update and a
validation meta-gradient:

$$
\widehat\theta_{t+1}(\epsilon)=\theta_t-
\alpha\sum_{i=1}^{n}\epsilon_i\nabla f_i(\theta_t),\quad
u_{i,t}=-\eta\left.\frac{\partial}{\partial\epsilon_i}
\frac1m\sum_j f_j^v(\widehat\theta_{t+1}(\epsilon))\right|_{\epsilon=0},\quad
\widetilde w_{i,t}=\max(u_{i,t},0).
$$

Eq. (9) normalizes these weights, handling an all-zero batch explicitly.
This verifies that **validation-guided marginal training influence is already a
published idea**, including class-imbalance settings. It does not establish that
their algorithm is a synthetic generator or uses our proposed local ranking
contrast. Section 3.2, Eq. (12), relates the meta-gradient to training/validation
gradient alignment. Gradient alignment alone is not a new contribution here.

Section 3.4, Lemma 1 and Theorem 2, claim monotone validation-loss behavior and
a stationary-point rate under a Lipschitz-smooth validation objective, bounded
training-loss gradients, a restricted step size, and the full validation batch
condition discussed there. Specifically the lemma gives
$\alpha_t\le2n/(L\sigma^2)$, and Eq. (15) bounds
$\min_{0<t<T}\mathbb E\|\nabla G(\theta_t)\|^2\le C/\sqrt T$.
These are authors' stated **optimization** claims, not guarantees of outer-test
improvement, minority recall, safe synthetic labels, or no false-positive harm.
The setup permits the small validation set to belong to the training set; that
must not be imported as an independence assumption for our proposed audit bounds.
Detailed supplemental proof verification is outstanding.

### [D] Ghorbani & Zou (2019), Data Shapley: Equitable Valuation of Data for Machine Learning

Full-text source: [PMLR PDF mirror](https://raw.githubusercontent.com/mlresearch/v97/gh-pages/ghorbani19c/ghorbani19c.pdf).
[Official PMLR metadata and abstract](https://raw.githubusercontent.com/mlresearch/v97/gh-pages/_posts/2019-05-24-ghorbani19c.md).

Section 2 defines the value function V(S) for a learning algorithm trained on S
and a chosen evaluation metric. Eq. (2) expresses the Shapley value as:

$$
\phi_i=\mathbb E_\pi[V(S_\pi^i\cup\{i\})-V(S_\pi^i)],
$$

where S_pi^i precedes i in a uniformly sampled ordering. Proposition 2.1
characterizes the value up to scale using null contribution, symmetry, and
additivity properties. This is an **axiomatic valuation** result, not a guarantee
that a high-value region benefits from unlimited synthetic interpolation.

Section 3.2 explicitly extends valuation to **groups**. Section 4.1 uses predicted
data value to guide acquisition of new patients; Section 4.5 evaluates group
valuation. Claiming that “local data value,” “subgroup usefulness,” or “marginal
benefit of extra training data” is untouched prior art would contradict these
sections. Actual newly acquired labelled observations are not equivalent to
synthetic examples derived from existing anchors. The acquisition example uses
observables to predict value; this is not a pseudo-labelled semi-supervised
oversampling algorithm.

The exact sum is exponential; Section 3.1 describes Monte Carlo and truncation.
The empirical truncation approximation is not a theorem that all omitted
contributions vanish. Section 4 explicitly separates the data used for V from
another held-out set used for reporting final results. That is a directly
relevant precedent for avoiding valuation/evaluation reuse.

**Critical guarantee check (our mathematical counterexample, not a reported
paper result):** the three displayed properties alone do not establish the
claimed uniqueness. The uniform-subset marginal functional
$\beta_i=2^{-(n-1)}\sum_{S\subseteq D\setminus\{i\}}
[V(S\cup\{i\})-V(S)]$ also satisfies null contribution, symmetry and additivity.
For n=3 and $V(S)=1\{\{1,2\}\subseteq S\}+1\{\{1,2,3\}\subseteq S\}$,
Eq. (2)'s permutation value is (5/6,5/6,1/3), whereas beta is (3/4,3/4,1/4).
They are not proportional even within this single game. An additional condition,
such as efficiency in the usual normalized game formulation, is needed to rule
out this alternative. Thus Proposition 2.1's displayed uniqueness claim should
be recorded as **claimed, not independently established from those three axioms**.
The Eq. (2) value remains a well-defined and relevant published valuation method;
this objection does not erase its group/acquisition prior art. No safety or
generalization guarantee follows from either functional.

### [I] Koh & Liang (2017), Understanding Black-box Predictions via Influence Functions

Full-text source: [PMLR PDF mirror](https://raw.githubusercontent.com/mlresearch/v70/gh-pages/koh17a/koh17a.pdf).
[Official PMLR metadata and abstract](https://raw.githubusercontent.com/mlresearch/v70/gh-pages/_posts/2017-07-17-koh17a.md).

Section 2.1, Eqs. (1)–(2), gives the influence of upweighting an example z:

$$
I_{up,params}(z)=-H_{\widehat\theta}^{-1}\nabla_\theta L(z,\widehat\theta),\quad
I_{up,loss}(z,z_{eval})=-\nabla_\theta L(z_{eval},\widehat\theta)^\top
H_{\widehat\theta}^{-1}\nabla_\theta L(z,\widehat\theta).
$$

The paper calls the query point z_test; here it is renamed z_eval to avoid
suggesting that outer-test labels can guide our synthesis or selection. The
formula requires an attained risk minimizer, twice-differentiable risk, and a
positive-definite Hessian in the stated convex setup. Removal and finite-batch
effects are approximations, not exact derivatives for a large augmentation.
Sections 4.2–4.3 examine damping/nonconvex and smooth-loss approximations;
empirical usefulness there does not restore the original convex guarantees.

This is verified prior art for estimating training-data impact on another
example's loss. Section 7 itself warns about moving from local perturbations to
larger subgroup changes. A finite, group-specific synthetic intervention still
needs direct confirmation rather than an influence-score guarantee.

## Mathematical comparison of verified additional sources

| Source | Quantity | Seeds | Difficulty | Overlap | Intra-class imbalance |
| --- | --- | --- | --- | --- | --- |
| [R], §3 | Validation loss under learnt example weights | No synthetic seeds; minibatch examples | Gradient contribution to validation objective | No explicit geometric overlap guarantee; noisy examples can receive zero weight | Can reweight individual examples; no explicit small-disjunct coverage theorem |
| [D], §§2–3 | Marginal contributions averaged over coalitions | No synthetic seeds; §4.1 acquires valuable new observations | Model/metric-dependent data value, including groups | Low value need not mean overlap; no overlap-removal guarantee | Group valuation is explicit, not limited to global class frequencies |
| [I], §2 | Infinitesimal parameter/evaluation-loss response | No oversampling seed policy | Curvature-adjusted training influence | Helpful/harmful effects are model-dependent, not a geometric safety certificate | Local influence may be aggregated; finite subgroup effects are not guaranteed |

| Source | Separate unlabelled features | Pseudo-labels | Classifier dependence | Marginal additions | Threshold-only comparison | Guarantee scope |
| --- | --- | --- | --- | --- | --- | --- |
| [R] | Algorithm 1 uses labelled training and clean validation; no distinct U input | Not a pseudo-label generator | Yes, through training/validation gradients | Infinitesimal example-weight effects; no synthetic-batch test | Not identified in reported experiments | Smoothness/gradient/step-size-dependent optimization result |
| [D] | Core valuation is supervised; acquisition pool covariates are used in §4.1 | Not in the core valuation algorithm | Yes, learner A is part of V | Yes; finite data/group contributions; acquisition guidance | Not identified in §§4.1–4.5 | Claimed axiomatic uniqueness; counterexample to displayed three-axiom sufficiency above; no synthetic safety guarantee |
| [I] | Labelled training and an evaluation query; not a semi-supervised generator | Not part of its influence formula | Yes, gradients and Hessian | Infinitesimal weight effects; finite changes approximated | Not identified in reported comparisons | Differentiability, optimum and Hessian assumptions |

“Not identified” is limited to the reviewed full-text method/experiment sections,
not a claim about every later implementation or all related work.

## Verified implementation/documentation comparators

These are source-level evidence, **not full-text verification of their cited
original papers**:

- [S1] [scikit-learn 1.5.2: threshold tuning](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/classification_threshold.rst).
  Separates prediction scores from decisions; threshold tuning leaves ROC/PR
  curves unchanged. Warns against fitting and tuning on the same observations.
- [S2] [scikit-learn 1.5.2: evaluation metrics](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/model_evaluation.rst).
  Verifies balanced accuracy and AP equations. AP is not trapezoidal PR-AUC.
- [S3] [scikit-learn 1.5.2: semi-supervised learning](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/semi_supervised.rst).
  Verifies a genuine unlabelled-input API, self-training with model-assigned
  labels, and graph propagation/spreading requiring distributional assumptions.
- [S4] [scikit-learn 1.5.2: cross-validation](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/doc/modules/cross_validation.rst).
  Explains separate validation/test roles and overfitting by repeated selection.
- [S5] [imbalanced-learn 0.12.4: oversampling guide](https://raw.githubusercontent.com/scikit-learn-contrib/imbalanced-learn/0.12.4/doc/over_sampling.rst).
  Documents ROS, SMOTE, ADASYN, boundary variants and KMeansSMOTE; warns about
  outlier-driven generation. Thus clustering, boundary focus, and difficult-point
  emphasis are already established ingredients at the implementation level.
- [S6] [imbalanced-learn 0.12.4: KMeansSMOTE source](https://raw.githubusercontent.com/scikit-learn-contrib/imbalanced-learn/0.12.4/imblearn/over_sampling/_smote/cluster.py).
  `_find_cluster_sparsity` computes mean pair distance^exponent / target-class
  count; eligible-cluster sampling weights normalize these sparsities. The
  source cites Last, Douzas and Bacao, [arXiv:1711.00837](https://arxiv.org/abs/1711.00837).
  Their original paper was not read; do not attribute the version's automatic
  exponent or thresholds to an unverified original equation.
- [S7] [scikit-learn 1.5.2: graph source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.5.2/sklearn/semi_supervised/_label_propagation.py).
  Cites Zhou et al., “Learning with local and global consistency” (2004) for
  LabelSpreading, and Zhu/Ghahramani for LabelPropagation. The publications
  themselves were not obtained; graph implementation evidence does not verify
  their original theorem statements.

## Verified reference register and audit limits

| ID | Bibliography / primary evidence | Verification level |
| --- | --- | --- |
| R | Ren, Zeng, Yang & Urtasun. ICML 2018, PMLR 80:4334–4343. Links above. | Full primary PDF and publisher abstract/metadata |
| D | Ghorbani & Zou. ICML 2019, PMLR 97:2242–2251. Links above. | Full primary PDF and publisher abstract/metadata |
| I | Koh & Liang. ICML 2017, PMLR 70:1885–1894. Links above. | Full primary PDF and publisher abstract/metadata |
| S1–S7 | Pinned library documents/source linked above. | Complete source pages/files relevant to stated implementation claims |
| U/L/H/G/B/T | Supplied full PDFs, DOI/arXiv links and page citations above. | Full text accessible and key method sections inspected; proofs not exhaustively audited |

PDF SHA-256 values:

```text
R: 3d46118cfb8575a12ed3ea2e5c90857db865fb9fc01c7f8b6565090de8ddf276
D: 48979f27f610c834ffda28ab48194c1af13c453d32add965fc817bba1e9b4407
I: 715fb9f935b0af32c7ae1f0d834fd5158edb279a575a4ae91196d1830f281f65
```

Downloaded PDFs/text were kept outside the project under `/tmp`, not committed.
No paywall circumvention or private-data access was used. This is a targeted
counterexample search, not an exhaustive systematic literature review. All
novelty conclusions must reflect that limitation.
