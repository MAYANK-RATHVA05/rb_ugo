# Question B: recent-paper extension decision

18 October 2026. **Design and bounded diagnostic; no production sampler or
large benchmark.** Starting branch `question-b-research-proposal`, HEAD
`7d907f4ab18814f2821a784b9e8057f43d85655e`.

## Decision and project state

**Do not implement a new sampler from these candidates.** The bounded diagnostic
confirmed that generated descendants can make deliberately corrupted originals
survive a generic density cleaner. However, original-only support and ordinary
weighted DBSCAN handled the fixture more effectively than the proposed local
parent cap, while retaining the genuine rare observations. There is no demonstrated
incremental need for its extra mechanism. This is not a CRN-SMOTE finding or a
claim that all Question B extensions are exhausted.

The potentially distinctive component was a *local, nonadditive cap on
original-parent contributions*. Its exact prior publication remains unresolved;
global normalization is already ordinary weighted DBSCAN. The two other
paper-specific limitations below support an audit or existing-method comparison,
not a defensible new extension yet. The next research step is to obtain and
verify the unavailable anchor methods before choosing another mechanism, rather
than implement this rejected prototype or add components to rescue it.

The requested undo removed the earlier Question B files from the working tree.
I read their complete historical versions without restoring or changing them:
[pilot/protocol and evidence](https://github.com/MAYANK-RATHVA05/rb_ugo/tree/6107a19/docs/research),
[decision memo](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/c857d43/docs/research/QUESTION_B_DECISION_MEMO.md),
[extension design](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/a42780f/docs/research/QUESTION_B_EXTENSION_DESIGN.md),
and [candidate A review](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/6107a19/docs/research/QUESTION_B_CANDIDATE_A_REVIEW.md).
The four archived CSVs contain 108 aggregate rows, 1,260 paired contrasts,
84 mechanism rows and 17 validation-selection rows. Their Git blobs remain intact.

Those results constrain, rather than exhaust, extension design. Clean RF100
rare recall was 86.2% for reference UGO, 96.4% for validation-tuned UGO, 97.8%
for ROS and 99.6% for EC-EBO. The filter-disabled diagnostic did not establish
an incremental need beyond those controls. Cross-method winners were identified
retrospectively; no validated selector across those methods was demonstrated.
Candidate A remains rejected: established inferred-group robust selection and
unavailable selection-group counts do not become new methodology through a
larger sampler catalog. Neither candidate is revived here. The excluded
provisional PDF and inactive RB-UGO/CSG designs were not used.

## Primary-source evidence and access

Evidence labels: **P** = inspected primary-paper statement; **C** = inspected
implementation; **D** = deduction; **E** = diagnostic outcome; **H** = hypothesis.
PDF pages below are document pages. Publication years follow the actual source;
SLEO has a 2025 DOI and a 2026 issue date.

| Closest source | Verified procedure and limitation | Implication / access boundary |
| --- | --- | --- |
| [CRN-SMOTE, 2025](https://doi.org/10.1371/journal.pone.0317396) | **Not verified:** stage order, clustering inputs, cleaner settings and the reported future-work statement. | PLOS HTML/manuscript/API, DOI and PMC/EuropePMC routes returned proxy CONNECT 403. No fulltext or abstract was obtained here. The user's description is a lead, not a verified quotation or attributed defect. |
| [SMOTE-PCA-HDBSCAN, 2025](https://www.nature.com/articles/s41598-025-97248-0) | **Not verified:** whether PCA/HDBSCAN sees originals, augmented rows or another population. | Nature and alternate primary-text routes were blocked. Do not infer its density behavior from its name. |
| [PPEO, 2026](https://doi.org/10.1016/j.neucom.2026.133891) | Full methods and verified implementation unavailable. | Title/DOI supplied by user; not a completed review. Uploaded `S0925231226015195` is **GDEO**, not PPEO. |
| [UGO, 2026](https://doi.org/10.1016/j.knosys.2026.115886), §§3.2–3.4, pp.6–8; current [code](../../src/ugo.py) and [audit](../ugo_reproduction.md) | **P/C:** fixed original-only prefilter; augmented classifier fitting; generated-row votes against all original observations; relocation toward eligible originals. | Descendants affect the classifier, but **never support one another in the cleaning reference**. This particular feedback mechanism does not apply to current UGO. Filtering a seed does not remove its original training row. |
| [GDEO, 2026](https://doi.org/10.1016/j.neucom.2026.134121), §3, pp.4–6, Eqs.1–9/Algorithm 1 | **P:** original-data logistic Gini scores, Gini quotas, minority/majority potential and gradient steps; no augmented-density cleaner. **D:** written descent/stopping does not establish its no-crossing description. | Full supplied PDF read; author implementation not obtained. Authors' §5 future work concerns multiclass extension, not our deduction. |
| [SLEO, 2025/26](https://doi.org/10.1016/j.knosys.2025.115009), §§3.1–3.3, pp.4–7 | **P/C:** original minority constrained single linkage; summed original entropy; translate majority observations; then generate within centroid spheres. | Full supplied PDF and [paper-linked code](https://github.com/taoliangliang/SLEO/blob/2b4b78e31726aa59e798faf21c8e453093fffa7e/sleo.py) read. Binary cleanup precedes generation. Its labeled `X,y` procedure needs no unlabeled corpus; “semi-supervised” refers to constrained clustering. |
| [Hamilton, 2026](https://doi.org/10.1016/j.neucom.2026.134831), §§3,5.5,6/Table 5 | **P/C:** original/OOB triage; threshold-moving, weighting and sampling controls. Threshold parity depends on learner, with reported exceptions. | Full supplied PDF and [author code](https://github.com/allianceai/eit/tree/0deed3d2bbd951209da99a3845541ed52af35fff) inspected. Not a theorem that sampling never helps; local geometry and triage thresholds remain limitations. |
| [Ma–Zhang, 2026, v3](https://arxiv.org/abs/2601.16120v3), Eqs.2–4, §6/Remark 2/VTSS | **P:** augmentation mismatch and quantity matter; validation-based size selection already exists. | Supplied main-text analysis retained; not an exhaustive audit of the long proof supplement or a guarantee for arbitrary forests. |

Older controls are also substantive. Inspected imbalanced-learn 0.12.4
[SMOTE-ENN](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/0.12.4/imblearn/combine/_smote_enn.py#L149)
generates first, then applies ENN to **all augmented rows**. Its
[default ENN](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/0.12.4/imblearn/under_sampling/_prototype_selection/_edited_nearest_neighbours.py#L156)
uses three neighbors and unanimous label agreement. This is label editing,
not DBSCAN density thresholding. Synthetic rows can change its votes (**D**);
harm is not implied. Historical source: Batista et al. (2004),
[DOI](https://doi.org/10.1145/1007730.1007735); primary article not newly obtained.

[Weighted DBSCAN](https://github.com/scikit-learn/scikit-learn/blob/1.5.2/sklearn/cluster/_dbscan.py#L420)
already sums fixed row weights in each radius neighborhood. GDBSCAN separates
neighborhood and core predicates ([Sander et al., 1998](https://doi.org/10.1023/A:1009745219419);
verified [ELKI implementation](https://github.com/elki-project/elki/blob/main/elki-clustering/src/main/java/elki/clustering/dbscan/GeneralizedDBSCAN.java)).
A lineage predicate specializes that framework; this alone neither proves nor
disproves prior publication of the exact resampling adaptation. Third-party
[DBSMOTE](https://github.com/analyticalmindsltd/smote_variants/blob/master/smote_variants/oversampling/_dbsmote.py)
already fits density structure to originals before synthesis; its original
paper was not obtained. Two public “RN-SMOTE” repositories lacked verified
author linkage and disagreed in procedure; neither is a faithful RN/CRN control.
Failed searches supply **no evidence of novelty**.

## Starting limitation: reliability can depend on synthetic quantity

For an augmented cloud \(Z_q=F\cup\{z_1,\ldots,z_q\}\), ordinary DBSCAN marks
a row core when

\[
C_q(x)=\sum_{z\in Z_q}1\{\|z-x\|\le\epsilon\}\ge m.
\]

At fixed originals, epsilon and minimum support, adding a shared synthetic
prefix cannot decrease an original's count. A sparsely observed mislabeled
region can therefore acquire core status. **This is a deduction about that
specified cleaner**, not a verified CRN procedure. Adding density can also be
intended: it may retain a genuine sparse subgroup or connect a correctly
interpolated interior. The scientific question is whether the change retains
corruption and causes independent prediction loss *after* simple controls.
Neither a larger cluster nor a lower noise-retention percentage answers that.

### Smallest candidate, specified without claiming novelty

Inputs: labeled numerical fitting observations; original IDs; synthetic
coordinates and exact original-parent interpolation coefficients. No true
subgroup/noise identities, validation examples or extra calibration labels.
Meaningful feature distance and interpolation are assumptions, not learned
facts. This would extend an augmented-cloud density cleaner, **not UGO**.

Represent ancestry by a sparse nonnegative matrix \(A\), with each row summing
to one. Original i has \(A_{ri}=1\). For
\(z_r=(1-\lambda)x_a+\lambda x_b\), assign
\(A_{ra}=1-\lambda,A_{rb}=\lambda\); combine repeated parent IDs. Overlapping
pairs share the same original columns, rather than being treated as independent
families. Zero coefficients confer no support. Recursively compose coefficients
if synthetic parents are ever allowed; this study uses originals only.

For \(B(x)=\{r:\|z_r-x\|\le\epsilon\}\), define

\[
S(x)=\sum_i\min\left(1,\sum_{r\in B(x)}A_{ri}\right),\qquad
\operatorname{core}(x)=1\{S(x)\ge m\}.
\]

Retain core points and their immediate epsilon-neighbors; delete unsupported
positive originals **and** synthetic rows in this diagnostic. This is training-row deletion, not UGO seed
eligibility or relocation. Each component has one purpose: ancestry identifies
reused sources; fractional coefficients avoid giving a near-zero interpolation
endpoint a full vote; saturation bounds a source's local contribution. It does
not estimate the probability that a label or subgroup is correct.

```text
generate a fixed original-parent synthetic prefix; record A
construct radius neighborhoods on original and synthetic positive rows
for each query: add coefficients by original ID; cap each ID at one
mark core if summed support >= m; retain cores and immediate neighbors
fit a fresh classifier; select its threshold using validation negatives only
```

Cost for M augmented rows, n original positives and E radius-neighbor
incidences: O(M²d) worst-case distances plus sparse accumulation bounded by
O(E+Mn), with O(Mn+E) storage in a simple small-data implementation. No extra
classifier fit or labeled selection set is intrinsic to the rule. A production
implementation would use sparse ancestry and chunked radius queries.

### What is distinct, and what is already available?

| Component/control | Exact relationship |
| --- | --- |
| Conserved scalar mass | \(c_i=\sum_rA_{ri}\), \(w_r=\sum_i A_{ri}/c_i\), hence \(\sum_rw_r=n\). Applying neighborhood sums of w is **exactly standard weighted DBSCAN**. |
| Local cap | Nonadditive: two same-source ROS copies separately contribute one each, but jointly contribute only one. No fixed scalar weights reproduce every subset. Potential distinction is this lineage-support semantics, not a new clustering framework. Its prior publication remains unresolved. |
| Original-only reference | If only ancestors whose original locations lie inside the radius ball may contribute, originals themselves saturate every eligible column: S becomes the original-neighbor count. Any additional behavior requires distant ancestors to support a query through descendants, an interpolation assumption. Freeze the original graph too; zero-weight synthetic vertices can still change connectivity. |
| Clean first | Removes the same initially unsupported originals before any descendants exist; simpler, but changes the generation pool and can lose genuine rare observations. |
| Threshold/count scaling | Uniform duplication multiplies density; scaling m or assigning inverse multiplicity can undo it. This is a necessary explanation/control, not a new mechanism. |
| SMOTE-ENN / HDBSCAN / RN/CRN | ENN edits augmented neighbor labels; HDBSCAN has different density/connectivity rules. No equivalence to our core predicate, and no faithful RN/CRN or requested HDBSCAN reproduction is established here. |

**Failure behavior:** two copies of a half/half descendant raise S from one to
two before saturation: the cap bounds inflation, not quantity dependence.
Counting distinct parents instead would give an arbitrarily small coefficient
a full vote. A max-per-parent rule would resist exact duplication but is a
different coverage rule; adding it to rescue a failed result is not justified.
A component with r original contributors has S≤r. Thus r<m genuine groups can
never become core from their own descendants. Compact mislabeled groups with
r≥m can pass. Tiny groups, unstable geometry and irreducible overlap are not
resolved by lineage. Identical observed configurations with different latent
truth cannot be distinguished from ancestry alone. This directly limits the
claimed rare-subgroup purpose.

## Bounded diagnostic: frozen before outcomes

**One execution: 130 RF100 fits, 220 records, 19.79 seconds; no failed arms.**
The final protocol and script were hashed before generating data or inspecting
outcomes. Seeds **242–246**, overlap **0/.20**, fitting/validation/test
**600/2,000/10,000**. Observed fitting positives: 40 main, 14 secondary, three
genuine rare, three deliberately false-positive labels; 540 observed negatives.
Evaluation positives have conditional weights 70/25/5 and prevalence .10.
True fitting prevalence is .095 because the three corrupted labels are not
genuine positives. This is a constructed stress test, not a measured noise rate.

Two-feature Gaussian main/secondary/rare populations have centers
(-3,0)/(0,3)/(3.8,3), isotropic SDs .35/.35/.08. Three false-positive originals
lie at (-.45,-.30), (.45,-.30), (0,.48), with SD .02 jitter. Base negatives mix
70% N((0,0),.70²I) and 30% N((3.8,0),.70²I). In overlap, 20% of negatives
instead follow the **rare** positive law. Thus population FPR≥.20×rare recall;
this differs from the archived pilot's aggregate-recall overlap bound and does
not prove any observed worst-group optimum.

Original fitting rows are fixed within each quantity sweep. Existing
`src.generators.generate_batch`, ordinary original-pool SMOTE k=2, supplies one
480-row draw, with exact prefixes **q=0/120/480** and recorded parents/lambdas.
No UGO eligibility intervention is used. All generic cleaners use **epsilon=.45,
m=3**, chosen to permit the compact three-original genuine group. Row-count,
parent-cap and scalar-weight cleaners use the same augmented queries and border
rule. Original-only structure freezes cores on original positive rows, then
retains rows near those cores. Clean-first regenerates from surviving originals;
its altered pool is an explicit difference. Negative originals remain in those
five arms. Shared-prefix SMOTE+ENN can delete both classes.

Every classifier has 100 trees, leaf size 1, paired seed, one worker; no tuning.
Each threshold is the smallest validation-negative score permitting empirical
FPR≤.05 with strict score>threshold. Test FPR is measured. Group/corruption IDs
are evaluator-only. No added selection/calibration label budget is assumed.
q=0 supplies mechanism counts only; the uncleaned q=0 predictor is NORES.
Four post-cleaners are fitted at q120 and q480; clean-first only at q480;
NORES, balanced class weighting, ROS480 and shared-prefix SMOTE+ENN480 complete
the 13 fits per seed/regime. Distinct-parent union is mechanism-only.

Mean corrupted-original retention, **percent**, across five seeds. Descendant
percentages also average per-seed fractions, rather than pooling denominators:

| Cleaner | q=0 | q=120 | q=480 | Noise-descendant retention at q480 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary row-count | 0 | 80.0 | 100 | 100 |
| Fractional parent cap | 0 | 6.7 | 73.3 | 80.0 |
| Original-only structure | 0 | 0 | 0 | 0 |
| Conserved scalar mass | 0 | 0 | 0 | 0 |
| Clean before generation | 0 | 0 | 0 | None generated |
| Unweighted distinct-parent union | 0 | 60.0 | 93.3 | 100 |

All six listed rules retained **100% of genuine rare originals and
their generated descendants** at both positive q levels. This does not mean
all genuine originals survived: original-only retained 54.8/57 genuine positive
originals on average. Positive clouds are identical across overlap regimes, so
these repeated masks provide **five**, not ten, independent mechanism replicates.

At q480, mean **rare recall % / achieved test FPR % / AP**:

| Arm | Retained additions, clean/overlap | No overlap | .20 rare-overlap |
| --- | ---: | --- | --- |
| Row-count cleaner | 480 / 480 | 100 / 4.876 / .98743 | 9.2 / 5.204 / .94677 |
| Fractional parent cap | 476.4 / 476.4 | 100 / 4.273 / .98807 | 10.4 / 5.336 / .94776 |
| Original-only structure | 450.8 / 450.8 | 100 / 2.611 / .99078 | 19.2 / 4.993 / .95559 |
| Conserved scalar mass | 453 / 453 | 100 / 2.271 / .99107 | 18.8 / 4.929 / .95587 |
| Clean before generation | 480 / 480 | 100 / 3.147 / .99137 | 20.4 / 4.938 / .94845 |
| Shared-prefix SMOTE+ENN | 456.4 / 443 | 100 / 4.820 / .98548 | 17.2 / 5.307 / .94994 |
| NORES (q0) | 0 / 0 | 100 / 4.429 / .99458 | 10.8 / 4.820 / .95653 |
| Class weighting (q0) | 0 / 0 | 100 / 4.558 / .99419 | 11.6 / 5.040 / .95722 |
| ROS480 | 480 / 480 | 100 / 5.309 / .98702 | 10.4 / 4.940 / .95072 |

In overlap, parent minus original-only rare-recall differences for seeds
242–246 were **−12,−20,−6,0,−6 percentage points**; AP differences were
**−.01165,−.01173,−.01113,0,−.00464**. There is no isolated positive seed being
promoted into a gap. Original-only test FPR ranged 3.933–5.856%, versus
4.422–5.811% for the cap; these are not identical achieved operating points.
The cap did not supply a favorable recall/AP trade-off in these observations.
ENN retained only one of three genuine rare originals on average in overlap,
but had higher rare recall than the cap: retention and learning remain distinct.
ENN also deleted 44.6/67.4 negative originals on average in clean/overlap, so
it is a full-method comparison, not a density-only ablation.

**Supported:** quantity-induced noise survival is real for the specified
unweighted cleaner; parent saturation still allows it; simple controls suffice
on this fixture. **Not established:** faithful CRN/HDBSCAN failure, a general
noise detector, rare-group prediction failure without overlap, or a uniquely
causal effect of lineage on performance. At q120, original-only overlap recall
was already 22.0% (AP .95798), versus its 19.2% at q480: more generation is not
the remedy. Different retained counts, query graphs and original deletions
remain mediators. No count-matched confirmation, cleaner tuning, UGO/EC-EBO
rerun or real-data benchmark was performed. This is exploratory screening, not
a five-seed superiority test or an equivalence claim.

The ignored local bundle `results/question-b-recent-paper-diagnostic-v1/`
contains the full settings, standalone harness, per-seed arrays/counts/metrics,
paired contrasts and hashes. It can be rerun into a fresh directory with the
same copied frozen protocol/source; the harness refuses an existing outcome
directory. Relevant SHA-256 values:

```text
PROTOCOL.md     7791fe2f0bab86000922a645d16201c82d2fc20973486212062e784cc95073d6
protocol.json   38a52f414b34264c61a6104e43eb39cb4ddf95fa988c58214195e4cbbc36d2d2
run_diagnostic.py 2cca0278fbed6824e8b1ae2761498d7245f12d18ec5285b6c2c320fe34dd387a
records.csv     c622f2a8fca4874014180ca88b387956c67d4ef8602dd3e2a5260c37d4c04b93
```

Checks covered shared prefixes, ancestry coefficients, conserved source mass,
equal q0 masks, truth-blind cleaner inputs, validation thresholds and fit caps.
An independent read of saved records reproduced the displayed means and paired
differences and verified frozen-file hashes, without fitting more models.

## Two other limitations examined

1. **GDEO potential descent versus its geometric description.** For a minority
   seed 0, majority reference 1, equal positive Gini weights c and alpha=1,
   its Eqs.5–7 give
   \(L(x)=c[e^{-x^2}-e^{-(x-1)^2}]\),
   \(L'(0)=-2c/e\), hence the first descent step is \(2\eta c/e>0\), toward
   the majority reference. Algorithm 1 stops at L≤0; here that means x≥1/2.
   This does not establish a no-crossing property. **D**, not the authors'
   stated limitation or evidence of deployed-code loss. Changing the sign,
   projecting a step, or using the existing EC-EBO positive-side rule is a
   repair/adaptation. Safe-level, geometric/local generators and BFCH are close
   controls. No distinct extension is supported by this inconsistency alone.

2. **SLEO centroid regions need not represent nonconvex minority support.**
   A deterministic construction, checked with unmodified pinned author
   `semi_cluster`, puts eight minority observations on a unit octagon and
   majority observations at the origin and a radius-three octagon. Adjacent
   minority distance is .7654, below every original's nearest-majority
   distance 1. The code merges all eight; its centroid is within 7.5e−17 of
   the central majority observation, one unit from the nearest minority row.
   Thus the merge constraint does not certify the centroid's reliability.
   Moving original majority rows and generating within that sphere could
   distort a rare nonconvex mode. The clustering behavior is verified; no
   downstream loss was measured. Replacing a sphere by ordinary local interpolation is already
   familiar machinery, requiring comparisons with EC-EBO, BFCH, A-SUWO and
   KMeans-SMOTE. Likewise, SLEO's summed-entropy quotas preserve a 40:4 group
   ratio when average entropy is equal: changing sum to mean repeats existing
   EC-EBO allocation. Neither observation establishes a new subgroup-allocation
   need. No sampler is selected from these two alternatives.

## Candidate comparison and evaluation gate

| Candidate | Evidence and substantive distinction | Feasibility / main risk | Decision |
| --- | --- | --- | --- |
| Local capped parent support | Exact nonadditive rule; generic dependence demonstrated, but no advantage over original-only or standard weights. Exact prior use unresolved. | Small-tabular prototype feasible in days; independent verification/adapters about 1–2 weeks. Rejects tiny groups and still admits source-supported corruption. | **Reject this variant on present evidence.** |
| Constrain GDEO's potential trajectory | Written rule lacks its claimed geometric protection; no attributed prediction harm. A sign/projection repair is not a substantive new mechanism. | Small repair; faithful author behavior unresolved. Existing safe/local generators may suffice. | Audit/clarify, not a new extension. |
| Replace SLEO's centroid sphere for nonconvex groups | Concrete valid cluster/centroid mismatch; harmful predictions remain hypothetical. Neighbor interpolation is established. | Feasible adaptation in days; must first survive existing-method controls and curved-support prediction tests. | No new method selected. |

No contribution sentence for a **new algorithm** is warranted. The defensible
finding is: **In this controlled stress test, counting synthetic descendants
changed noise survival, but original-only reliability structure removed that
failure without the proposed nonadditive lineage mechanism.** This is an
empirical mechanism result, not a global methodological novelty claim.

A meaningful lineage extension would have to beat the **simplest adequate
original-only/clean-first/weighted control**, not just an intentionally fixed
density threshold. A later confirmation would use a new frozen protocol and
fresh paired seeds 342–361, F600/V2000/T10000 synthetic data initially; no
larger labeled selection budget or true-group input. The present diagnostic
must not become its own confirmation. Real-data transfer needs independently
meaningful subgroup metadata and entity-safe splits, not post hoc clusters
defined using test errors.

Necessary arms: unchanged and validation-tuned UGO, NORES, class weighting,
ROS, ordinary SMOTE, SMOTE-ENN, original-only and clean-first density controls,
conserved-mass DBSCAN, and audited EC-EBO. Add faithful CRN/RN and the requested
HDBSCAN method only after obtaining their full specifications; missing arms
remain missing. Tune neighborhoods and cleaner parameters on the same training
folds/global validation utility with equally sized predeclared grids. No
subgroup labels tune anything. Equal generation proposals, shared coordinates
where applicable, retained-count-matched deletion/ROS, and family-normalized
classifier weighting distinguish geometry from quantity or training mass.
Report requested/achieved additions, original removals, noise-lineage survival,
rare-original retention, runtime, AP and achieved FPR separately.

Necessary ablations would compare the same augmented query locations under
row counts, local caps, conserved scalar mass, and original-evidence-only
counts (original weight 1, synthetic weight 0), alongside the stricter frozen
original graph. A synthetic-only deletion ablation would keep **all original
training rows fixed**; matching final totals alone cannot control which original
observations survive. These isolate source accounting from new query locations,
bridges and original deletion. They were not run in the screening diagnostic.

Primary mechanism contrast would be capped support minus original-only support
in worst true-subgroup recall at the common validation-FPR target .05, on the
predeclared noisy regime. Require a lower 95% whole-paired-seed interval above
**+.05**, AP difference lower limit≥−.005 and FPR increase upper limit≤.005;
report absolute FPR and a maximum .06 per replicate as a guard, not a population
guarantee. Require clean-regime recall noninferiority at −.02. Use the same
paired bootstrap resamples (10,000, RNG42). Intervals spanning the practical
margin are inconclusive; an upper limit below it rejects that benefit.
Threshold controls alone cannot establish ranking benefit; AP and identical
validation-negative threshold opportunities expose this alternative. Matched
retained counts and classifier weights expose ordinary reweighting. Shared
classifier settings, seeds, capacity and tuning budgets expose extra training/search
as explanations; models with different fitting rows are necessarily fitted separately.

For 20 seeds and two regimes this would mean roughly 400–800 final RF100 fits
plus up to 160 four-choice UGO trajectories, depending on the frozen ablation
catalog. Planning estimate: 5–30 CPU minutes, hard cap one CPU hour; adapter
verification is separate work. These are estimates, not an executed benchmark
or a promise for larger real datasets. A future method would need its own
complete manifest before outcomes; this outline is not a frozen launch order.

Do not execute that confirmation automatically. First require a specific
parent-cap advantage surviving the small diagnostic and a closer primary-source
novelty check. Failure against original-only or ordinary weighted controls,
rare-group deletion without recall benefit, purely threshold/count gains, or
frequent inactive mechanisms rejects this mechanism as a new-method target in
the tested setting. Such rejection does not exhaust supervised Question B.

## Plain-language interpretation

The proposed guard would ask how many original observations support a dense
patch, instead of counting every generated descendant as fresh evidence. That
could stop copied mistakes from making themselves look trustworthy. But a few
real rare observations also have few original supporters. The method must show
that its extra bookkeeping helps those real groups more than simply judging
reliability before generation; ancestry alone cannot establish that distinction.
