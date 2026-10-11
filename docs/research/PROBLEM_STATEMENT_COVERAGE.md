# Independent coverage of research questions A, B and C

Stage 4A-R research design, not executed experiments. U/L/H/G/B/T citations
resolve to [full-text evidence](PRIOR_ART_MATRIX.md). Real-file facts derive from
[the dataset manifest](../../data/manifests/keel_dataset_manifest.csv).
Full-file parsing and an uncertainty score do not establish coverage of a question.

## A. Boundary proximity with limited harmful overlap

**Existing UGO:** uncertain clean seeds, original-neighbor noise voting and
relocation [U pp6–8]; no posterior-membership or finite-benefit guarantee.
**Related methods:** L's majority translation [L p6 Eq.6], H's impure-cluster
cleaning [H pp4–6], G's Gini/descent objective [G p4 Eqs.1–7] already address
ambiguity. B distinguishes threshold-recoverable errors and category-targeted
benefit [B pp5,8–9]; its Proposition 1 shows overlap risks under stated geometry.
G's literal descent counterexample and L's sign/stop ambiguity prevent blanket
safety claims [matrix].

**Possible addition:** a conditional finite intervention audit with ranking and
majority-harm contrasts beyond tuned thresholds/weights/matched counts. This is
currently an empirical diagnostic, not defensible new methodology.
**Unresolved:** scarce independent minority labels, adaptive validation, target
choice (natural-prior versus balanced boundary), and generator-specific utility.
A negative intervention result does not establish irreducible ambiguity.

**Data:** all seven numerical supplied variants are potential future benchmarks;
Abalone19 needs a declared categorical-distance protocol. None has verified true
boundary or overlap labels. Official folds/provenance are pending for every file.
**Simulation:** equal-covariance two-class Gaussians with separation varied while
counts/noise stay fixed, as defined in DATASET_PLAN A. Use known density ratios
only for held-out diagnostics; natural-prior posterior and balanced Bayes boundary
must stay distinct. Include complete-overlap and separated cases.
**Metrics:** held-out boundary-band minority recall with counts, AP (declared
implementation), balanced risk, majority FPR, expected synthetic mismatch and
ambiguous-region mass. Global recall alone is insufficient [B p9; S1–S2].
**Baselines:** default/tuned no-resampling, tuned class weighting, ROS/SMOTE,
source-audited UGO, matched category/random interventions and VTSS-style count
search [B §5.4; T Algorithm 1].
**Falsifier:** adequately powered confirmation shows no practical discrimination
increment beyond matched controls, or majority harm exceeds its preregistered
margin. Uncertain intervals are inconclusive, not falsification.

## B. Intra-class imbalance

**Existing UGO:** uncertain point ranking within the selected class [U p7 §3.4],
without an explicit rare-subgroup guarantee. **Related methods:** L allocates to
minority clusters [L p6 Eq.7]; H decomposes subclasses [H pp4–6]; B uses local
representation ratio [B p4 §3.2]. KMeansSMOTE explicitly weights cluster sparsity
[S6]. Unequal counts alone are neither novel nor proof that synthesis helps.

**Possible addition:** determine whether a finite rare-region intervention is
beneficial for the declared learner, including interactions and inconclusive
outcomes. Group valuation already exists [D §§3.2,4.5]. **Unresolved:** region
instability, scarce audit labels, useful rare islands rejected as noise, and dense
regions dominating global metrics. No latent subgroup identities are provided
by the eight KEEL files; fitting clusters are estimates, not verified domain groups.

**Data:** all eight files may measure fitting-defined region diagnostics after
fold/provenance review, but simulations are essential to test true subgroup
coverage. **Simulation:** fixed two-component minority evaluation distribution,
unequal fitting component counts at fixed total, plus a separately declared
population-mixture arm [DATASET_PLAN B]. Component IDs never guide the method.
**Metrics:** both scarce/dense subgroup recall, counts, coverage, allocation,
worst-subgroup recall and global risk/FPR/AP; uncertainty by appropriate sampling
unit. **Baselines:** matched clustering/sparsity allocation, random regional budgets,
L/H if reproduction issues are resolved, SMOTE/UGO, group-value controls, tuned
threshold/weights/counts. **Falsifier:** rare-region gains disappear with matched
budgets, another minority region is sacrificed, or utility selection cannot
predict fresh region responses better than simple counts/uncertainty controls.

## C. Unlabelled support without ground-truth pseudo-label claims

**Existing UGO:** no separate U pool; it explicitly removes the oracle/U stage
[U p5 §3.1]. **Related methods:** H truly uses D_U and assigns selected confidence
labels, with a no-U ablation [H pp4–6,11–12]. L's semi-supervised clustering is
not a distinct U input [L p7 Algorithm 1]. Standard self-training and graph SSL
already exist [S3,S7].

**Information limit:** unlabelled X estimates feature support/connectivity under
sampling assumptions. It cannot determine P(Y=1|X), minority membership or truthful
anchors without assumptions; identical p_X can have different class posteriors
[gap analysis §6, explicit construction]. Anchor-consistency fields can estimate
structural compatibility but not genuine membership. Overlapping majority bridges,
missed minority modes, label noise and distribution shift can break smoothness.

**Possible addition:** test structural support as a separately controlled ablation,
not a new pseudo-label authority. **Unresolved:** graph/radius assumptions,
unlabelled provenance, fair label/compute budgets and baseline reproduction.
**Data/simulations:** train-only masked versions of future verified official folds,
known-overlap/subgroup simulations, same-distribution and shifted U [DATASET_PLAN D].

**Exact allocation design:** preserve O as untouched outer test. Allocate genuine
inner labelled H/A first under a declared total label budget; mask only fitting F
using seed 42 at 10%,25%,50%,100%, stratified deterministically where feasible.
Save hidden labels separately as diagnostic-only; the training interface receives
X_U without hidden y. Record actual F/H/A class counts and rounding, not just a
percentage. Match masks, visible labels and eligible U across methods. At 100%
labels with no external U, U is empty. No O features enter any graph/scaler.
Anchor limits and infeasible fixed-k conditions are calculated in DATASET_PLAN;
check actual train/mask counts before any fit and do not silently alter proportions.
No such masking algorithm or semi-supervised sampler is implemented in this stage.

**Metrics/controls:** paired no-U/U changes in AP, balanced risk, subgroup recall,
FPR and harmful locations at equivalent total labels; same-volume shuffled-feature
and shifted-U controls, labelled-only structural graph, ordinary SSL/self-training,
H with declared mask protocol. Hidden truth is evaluation-only and not counted as
training knowledge. **Falsifier:** adequate confirmation finds negligible U benefit
at matched budgets or U systematically increases harm under overlap/shift; a
confident graph/pseudo-label score is not evidence against that finding.

## Shared confirmation requirements

Use fitting F for anchors/preprocessing/regions, independent H for thresholds,
independent A for frozen finite candidate comparisons and untouched O for the
whole selected procedure. Adaptive reuse of A needs fresh confirmation/error
spending or a justified nested protocol [S4; candidates]. Match threshold, label,
U, fit and compute opportunities. Never treat repeated CV folds, repeated seeds
or related variants as independent new labelled units. Preregister practical
margins and power; retain negative/infeasible/inconclusive conditions. None of
A/B/C is experimentally solved by Stage 4A-R.
