# Question B: full-paper extension review and decision

18 October 2026. Revision of the [access-limited memo](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/56ce663/docs/research/QUESTION_B_RECENT_PAPER_EXTENSION.md),
using the three newly supplied **complete main papers**. Starting branch
`question-b-research-proposal`, HEAD `56ce6638831b91e7924c860f90e60f331d033cd1`.
Supplementary materials were not supplied. No production sampler or large
benchmark was implemented.

## Decision

**No candidate passes the present gate for implementation as a new extension.**
The full papers establish where descendant-supported cleaning is possible,
but do not establish a need for parent tracking. PPEO supplies a stronger,
explicitly published geometric limitation. I developed one precise directional
extension and screened it against ordinary shrinkage: it achieved its geometric
property, but **90.0% primary rare recall versus 90.4% for both shrinkage
controls** fails the prespecified practical-benefit criterion. Do not promote
its secondary overlap results or add mechanisms to rescue it.

This is a narrower conclusion than “Question B is exhausted.” Novelty of the
specific geometry also remains unresolved. The next prerequisite is a
source-faithful evaluation of a consequential residual failure, not another
sampler assembled from these rejected mechanisms. In particular, clarify PPEO's
missing implementation pieces and subclass-label protocol before treating its
published results as a strictly binary, unknown-subgroup baseline. No additional
experiment is queued by this memo.

Original UGO, the closed filter candidate, Candidate A, archived pilot results
and frozen protocols remain intact in Git history; no earlier files were
restored. The prior [pilot](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/6107a19/docs/research/QUESTION_B_DIAGNOSTIC_RESULTS.md),
[closure](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/c857d43/docs/research/QUESTION_B_DECISION_MEMO.md)
and [Candidate A review](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/6107a19/docs/research/QUESTION_B_CANDIDATE_A_REVIEW.md)
still constrain this work: tuning, ROS, weighting and existing subgroup-aware
methods are necessary controls; zero generation is not failed learning;
validation choices and retrospective test winners are different. The excluded
provisional proposal, inactive RB-UGO/CSG designs and unlabeled-data generation
were not used.

## 1. What the full primary sources establish

**P** = authors' procedure/statement; **C** = inspected code; **D** = deduction;
**E** = our exploratory experiment. Page numbers refer to the supplied PDFs.

| Primary source and access | Exact training populations and rules | Supported limitation and evidence boundary |
| --- | --- | --- |
| Hemmatian et al., [CRN-SMOTE, 2025](https://doi.org/10.1371/journal.pone.0317396), 20 pp.; §§2.3–3, pp.8–10, Algorithm 3/Eqs.2–4 | **P:** SMOTE to balance; DBSCAN separately on each **augmented minority class**; increase epsilon from a small value until **0 < retained cluster count ≤ 1 or 2**; combine with untouched original majority; SMOTE again to rebalance. Originals in the minority cleaning pool can be deleted, and surviving synthetics can enter final generation. | **P:** §5, pp.16–17 explicitly requests study of oversampling–denoising interaction and mutual-neighborhood SMOTE. **D:** the one/two-cluster prior can discard or merge real modes. No known-noise ancestry or held-out within-class subgroup test establishes harm. Minimum support and epsilon increment are not numerically specified sufficiently for exact experimental reproduction; no author-code link found. |
| Nasaruddin et al., [SMOTE-PCA-HDBSCAN, 2025](https://doi.org/10.1038/s41598-025-97248-0), 12 pp.; Methods pp.3–5/Fig.1/Eqs.2–3 | **P:** SMOTE; separate originals from synthetics; standardize/PCA **synthetics only**; HDBSCAN **synthetics only**; combine retained synthetics with **all original observations**. Mutual reachability is max(core-distance p, core-distance q, distance p–q). | **D:** descendants determine their own density and PCA space. Removing synthetic rows does not remove original rare examples. Classwise versus pooled clustering, PCA rank, HDBSCAN settings and how reduced coordinates select/reconstruct original-feature rows are unspecified. No linked implementation. Authors' p.10 limitations are complexity, parameter sensitivity and external validity. |
| Zhao et al., [PPEO, 2026](https://doi.org/10.1016/j.neucom.2026.133891), 18 pp.; §§3.1–3.4, pp.4–7 | **P:** label-constrained hybrid clustering; small prototypes from connected minority components remaining in majority regions; nearest-negative-radius SPSDO for small prototypes, within-prototype SMOTE for large ones; optimize prototype weights using objectives **1−sqrt(TNR × prototype recall)**; maintain an archive and route conflicting predictions by prototype. | **P:** §4.7, p.16 explicitly acknowledges anisotropic hypersphere inadequacy, possible true-boundary crossing and high-dimensional distance concentration. **D:** its mean-unbiasedness theorem is not support preservation. Allocation, prototype recall objectives and specialist prediction are already present; proposing them again is not a contribution. |

CRN's future-work wording is now verified: “analyze how the oversampling
process … interacts with the denoising phase using DBSCAN” (§5, p.16).
This is a research invitation, not evidence that a particular failure occurs.
Its four-dataset/global-metric results supply no rare-mode estimand; Table 8
also shows SMOTE-ENN outperforming it on Blood, despite broad superiority prose.

The HDBSCAN paper evaluates one restricted, 288-row, three-class water-quality
dataset, not independently known within-class subgroups. In Table 5, Clean and
Polluted recalls equal ordinary SMOTE's 28.57% and 61.90%; specificity changes.
Its Clean precision 46.15% and recall 28.57% imply F1≈35.29%, inconsistent with
the reported 91.54%. Eq.6 also prints a sensitivity denominator inconsistent
with its recall prose. These reporting issues limit empirical conclusions;
they neither erase the described synthetic-only stage nor establish its harm.

PPEO's [author-linked repository](https://github.com/zhaoujs/PPEO/tree/4d94423a86631ff1852547a1e9b776b73fe092cc)
is available, but imports absent `SSHR_utils.py`, `sec.py` and `tool.py`.
Its public `mopso_regin.py` evaluates fitness on original training rows and uses
global minority neighbors, differing from the paper's within-prototype rule;
`main.py` scales before cross-validation. These are specific public-source
limitations, not proof of what unshared final experiments executed. The full
method is not verified runnable. The paper's subclass experiment does not
fully clarify whether subclass labels were collapsed before training; our
proposed tests permit only binary observed labels and evaluator-only subgroups.

## 2. Reassessing the starting hypothesis

The concern is structurally applicable to both cleaning papers. For fixed
epsilon/minimum support, ordinary DBSCAN counts can grow when descendants are
added around fixed originals. CRN additionally changes epsilon and resamples
again, so the full pipeline is not the fixed-epsilon intervention. HDBSCAN's
neighbor distances, hierarchy and cluster stability can change with synthetic
quantity; **its final retention need not be monotone**. Refitting synthetic-only
PCA is a separate mechanism. Any direct study must compare a frozen
representation with the full representation-refitted pipeline.

Neither paper measures whether the affected regions are truly corrupt or
whether density changes cause independent prediction loss. Increasing density
can intentionally retain a genuine sparse group. Current [UGO](../../src/ugo.py)
checks generated rows against **original observations only** through
[noise_detection.py](../../src/noise_detection.py); descendants cannot support
each other's cleaning votes. UGO's seed exclusion also leaves originals in
classifier fitting. Do not transfer these papers' row-deletion semantics to UGO.

The earlier 130-fit [generic diagnostic](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/56ce663/docs/research/QUESTION_B_RECENT_PAPER_EXTENSION.md#bounded-diagnostic-frozen-before-outcomes)
remains unchanged. Row-count cleaning retained 100% of corrupt originals at
q480; a fractional parent cap retained 73.3%; original-only structure and
conserved-mass DBSCAN retained none while preserving all rare originals in that
fixture. It is **not** a CRN reproduction: no radius search, cluster ceiling or
second SMOTE. It is **not** the HDBSCAN procedure: it cleaned original positives
too, omitted PCA and used DBSCAN. The source review corrects applicability;
it does not retrospectively change that experiment or its conclusions.

For overlapping parent pairs, the investigated cap used original identity
columns A, with a SMOTE child contributing (1−lambda, lambda) to its two
original parents, combining shared columns. CRN's second generation would
require recursively composing **A_child=(1−lambda)A_parent1+lambda A_parent2**;
synthetic parents are not fresh original contributors. Local support was
**sum_i min(1, sum_{r in neighborhood} A_ri)**. It is bounded, not invariant to
quantity; a genuine group with fewer than m ancestors cannot reach m support.
Global normalization **w_r=sum_i [A_ri/(sum_s A_si)]** is exactly standard weighted
DBSCAN; a corresponding HDBSCAN weight interface was not verified.
Original-only reliability and clean-first controls avoid self-support
without lineage bookkeeping. The full papers supply no new evidence that
parent tracking improves on these alternatives. Keep this candidate rejected.

## 3. Three narrowly defined opportunities

| Candidate limitation | Why consequential / what remains hypothetical | Existing solution or competing explanation | Decision |
| --- | --- | --- | --- |
| Descendants confer apparent reliability | **P/D:** augmented-minority CRN and synthetic-only HDBSCAN use descendant density. **E:** generic dependence exists. Actual paper-specific subgroup prediction harm remains unmeasured. | Original-only structure, clean-first, weighting, SMOTE-ENN; our cap did not add value. | No parent-support extension selected. |
| CRN assumes at most one/two retained clusters per class | **P:** explicit Algorithm 3 stopping condition. **D:** multiple true disjuncts need not share a cluster; a first valid count can hide discarded modes. Consequential held-out loss is unproven. | Unrestricted/variable-density clustering, KMeans-SMOTE, A-SUWO, EC-EBO and PPEO already model multiple regions. Removing the ceiling alone is an adaptation. | No new mechanism justified. |
| SPSDO's empty-negative ball need not respect even anchor-relative class geometry | **P:** PPEO acknowledges spherical/boundary limitations. **D:** conditional mean preservation does not preserve shape or ensure a point stays closer to its source than to negatives. | Half-radius shrinkage already gives the same empirical safety and centering; geometric/Voronoi sampling already exists. | Develop **one** exact directional candidate below, then reject promotion if simple controls suffice. |

These are limitations of specific mechanisms, not automatic evidence for
subgroup quotas, stopping, metric learning or an extra labeled selection set.
The prior GDEO/SLEO audits remain historical context; they are not revived as
additional candidates in this revision.

## 4. The strongest candidate: symmetric directional SPSDO

**Contribution hypothesis, not an accepted novelty claim:** replace SPSDO's
sphere by a centrally symmetric, original-negative-constrained region that
preserves anchor expectation while retaining more directional spread than
uniform radius shrinkage at the same empirical safety level.

Inputs are original numerical fitting features/binary labels and the same
SPSDO anchor schedule and quotas as the comparator. No true subgroup identities,
unlabeled pool, held-out rows or additional calibration labels. In a full PPEO
comparison, PHHC, allocations, large-prototype samples, optimizer and routing
would remain fixed. This extends **PPEO's SPSDO operator**, not UGO's filter.

For a positive anchor x, write v_j=x_j^-−x for every original negative and
r=min_j||v_j||. Original SPSDO draws u uniformly on the unit sphere and
U uniformly on [0,1), then returns x+Ur u. Its theorem establishes E[z|x]=x;
the written law adds covariance **r²I/(3d)**, not shape preservation.

The anchor is no farther from x+h than any negative exactly when

\[
2h^\top v_j\le\|v_j\|^2\quad\text{for every }j.
\]

Requiring the same for both h and −h defines the maximal centrally symmetric
subset of this **anchor-versus-negatives** cell inside the original ball:

\[
K_x=\{h:\|h\|\le r,\;2|h^\top v_j|\le\|v_j\|^2\ \forall j\}.
\]

Sample using the unchanged uniform-direction/uniform-radius convention:

\[
\rho_x(u)=\min\left(r,\min_{j:|u^\top v_j|>0}
 \frac{\|v_j\|^2}{2|u^\top v_j|}\right),\qquad
z=x+U\rho_x(u)u.
\]

Because rho(−u)=rho(u), E[rho(u)u]=0 and E[z|x]=x, even though the final
radius and direction are dependent. This is not uniform-volume sampling or
exact finite-batch centering. Antithetic pairing is unnecessary and is not an
extra proposed component. For r>0, **r/2≤rho≤r**; the ordinary half-radius
ball already satisfies the safety property. The additional behavior is solely
retaining permitted spread between r/2 and r. The guarantee is sufficient,
not necessary, for empirical positive-class 1NN agreement: another positive
can protect a point that violates its own anchor condition. It is not a
population-label or Bayes-boundary guarantee.

```text
for each supplied original SPSDO anchor x and requested sample:
    compute differences to original fitting negatives and nearest distance r
    if r == 0: emit a diagnosed copy; no geometric safety claim
    else:
        draw uniform unit direction u and U in [0,1)
        cap radius by every forward/backward negative bisector
        emit x + U * capped_radius * u
return all unchanged originals plus generated rows; report copies/counts
```

Every part serves the one constraint: original negatives define empirical
separation, the reflected constraints preserve SPSDO's conditional centering,
and the existing radius r bounds expansion without a new scale parameter.
Naive cost is O(n_+ n_- d) initial geometry and O(B n_- d) for B generated rows;
chunking needs O(n_-d) working storage per anchor. No additional classifier or
selection-label budget is intrinsic. A small opt-in operator would take days
to implement/test; full PPEO integration requires separate source clarification.

**Limits:** no covariance estimate or minimum subgroup size is required, so a
singleton can generate when r>0; it could also be a mislabeled singleton.
Coincident opposite labels give copies, not resolved ambiguity. Surrounding
negatives can erase the directional benefit; in one dimension it is exactly
half-radius sampling. The unchanged outer r cannot repair under-expansion of
elongated groups. Symmetry also trims some safe movement away from negatives.
Bad distances, unobserved negative support and irreducible overlap remain.
It does not fix PHHC instability or in-sample evolutionary selection.

For x=(0,0), one negative (r,0), the original ball permits points closer to
that negative; the candidate clips to |h_1|≤r/2 but permits transverse radius r.
Half-radius sampling shrinks every direction. Original SPSDO's uniform-radius
anchor-inconsistency probability is **12.3733%** in this construction; both
guards make it zero. The candidate's feasible area is 60.8998% of the full
disk, versus 25% for half radius. This geometric difference was independently
checked by quadrature; it is not evidence of better prediction.

### Closest work: exact differences and conflicts

| Comparator | Established rule / difference from candidate | Consequence |
| --- | --- | --- |
| PPEO/SPSDO, Eqs.4–5,8–13 | Empty-negative sphere; isotropic direction and uniform radius. No simultaneous directional bisector constraints. | Direct operator extension; allocation and centering are already established. |
| Half-radius SPSDO | Centered sphere r/2 already prevents every observed negative from beating the anchor. | Mandatory simple control; safety plus centering alone adds nothing needed. |
| One-sided Voronoi clipping | Ordinary constraints 2h·v_j≤||v_j||², without reflection. | Tests whether symmetry matters beyond familiar safe-cell clipping. |
| [GeometricSMOTE author code](https://github.com/georgedouzas/imbalanced-learn-extra/blob/9e523b84af629f5a7dc85c419528d835a5774bce/src/imblearn_extra/gsmote/geometric_smote.py) | One selected neighbor/axis controls radius, cap reflection and perpendicular deformation; uniform-volume ball draws. | Candidate is not a literal parameter setting of this inspected code. Both are geometric sampling; this distinction alone proves neither novelty nor utility. [Primary DOI](https://doi.org/10.1016/j.ins.2019.06.007); full article unavailable here. |
| [V_SYNTH, verified third-party code](https://github.com/analyticalmindsltd/smote_variants/blob/c26cdd9d4a2a61886924fc9b081f0480f24b4108/smote_variants/oversampling/_v_synth.py) | PCA/full-site Voronoi faces; interpolate between an opposite-label face and its minority owner. Not centered symmetric radial sampling. | Voronoi-constrained synthesis is established. [Young et al., 2015](https://doi.org/10.1007/s00521-014-1780-0) primary fulltext unavailable; exact fidelity/unexamined Voronoi variants remain open. |
| Safe-Level-SMOTE, DBSMOTE, KMeans-SMOTE, A-SUWO, audited [EC-EBO](../EC_EBO_METHOD_DECISIONS.md) | Existing neighbor-label, density, clustering, sparsity/difficulty allocation and constrained generation mechanisms. | Necessary full-method controls; no new grouping/allocation claim follows from the candidate. Source-level qualifications from earlier audits remain. |

Inspected public CirCover/LR-CO code uses minority-centroid Voronoi partitions
inside pair-centered disks, not all-negative bisectors and central reflection;
peer-reviewed publication status was not verified. Broader Voronoi prior-work
coverage remains incomplete. The candidate uses familiar primitives and has
only a **potentially distinct specific operator**, not established global
methodological novelty. Failed searches were not interpreted as novelty.

## 5. A fresh, bounded experiment rejects promotion

The protocol/source were frozen **before data or outcomes**. One execution:
**140 RF100 fits, 140 successful records, 24.98 seconds**, no tuning or retries.
Fresh seeds **442–446**, F/V/T **600/2,000/10,000**, q **120/480**; primary
cell fixed as **zero added rare-law overlap, q480**. All originals remain identical across
operators/quantities. F contains 540 negatives, 40/14/3 genuine positive modes
and three deliberately corrupted positive labels. Evaluation prevalence .10,
conditional subgroup weights 70/25/5. True IDs remain evaluator-only.

The rare law is N((3,0),diag(.06,.50)²), flanked by majority components at
(2.75,0)/(3.25,0), SD (.06,.60), plus background N((0,0),diag(.80,1.20)²)
with weights .25/.25/.50. Main/secondary positives are centered (-3,0)/(0,3)
with SD .35. Corrupted originals are a fixed jittered three-point fixture in
the background. A second regime replaces 20% of negatives by the rare law;
there FPR≥.20×rare recall in population, not an observed optimum. Full settings
and RNG namespaces are in the frozen bundle. Gaussian supports overlap even
when the added-mixture fraction is zero; the frozen shorthand “non-overlap”
means no additional shared rare-law component, not perfect separation.

Every observed positive anchor receives exactly two/eight descendants.
Six operators use matched counts and paired classifier seeds: original SPSDO,
half radius, candidate, energy-matched scalar jitter, one-sided clipping, ROS;
NORES and class weighting complete the 14 fits per seed/regime. The scalar
control matches **realized squared displacement per anchor**, then independently
rotates its directions to remove alignment with negative geometry. Its q120
coordinates intentionally are not a prefix of q480. Other geometric arms
share directions/U/coordinate prefixes. No true groups determine any quota.

All models use RF100/leaf1. Thresholds use validation negatives only, with
strict score>threshold and empirical FPR≤.05. Primary screening required
**≥+.03 mean rare recall against each shrinkage control**, AP loss≤.01 and
test-FPR increase≤.005. These were screening margins, not confirmatory power
claims. Five paired seeds supply descriptive evidence only.

At q480, means are **rare recall % / achieved test FPR % / AP**:

| Arm | Zero added overlap: primary | .20 rare-law mixture: secondary |
| --- | --- | --- |
| Original SPSDO component | 89.6 / 5.569 / .92365 | 11.2 / 5.620 / .90687 |
| Half radius | 90.4 / 5.493 / .94157 | 10.4 / 5.729 / .92158 |
| Symmetric directional candidate | 90.0 / 5.444 / .93686 | 12.0 / 5.644 / .91795 |
| Energy-matched scalar jitter | 90.4 / 5.609 / .93630 | 10.4 / 5.778 / .91726 |
| One-sided Voronoi clipping | 90.4 / 5.513 / .93722 | 12.0 / 5.609 / .91590 |
| ROS | 83.2 / 5.533 / .94787 | 7.2 / 5.453 / .91909 |
| NORES, q0 | 82.4 / 5.378 / .95832 | 7.6 / 5.382 / .93519 |
| Class weighting, q0 | 82.0 / 5.313 / .95716 | 5.6 / 5.484 / .93471 |

Candidate minus half-radius primary recalls, seeds 442–446, were
**0,+2,−2,−2,0 pp**; versus matched scalar **+2,+2,−4,+2,−4 pp**.
Both mean effects are **−.4 pp**, failing the +3 pp screen. In the same regime,
q120 recalls were 86.8/87.2/88.8% for candidate/half/matched: no favorable quantity
was selected retrospectively. The secondary +1.6 pp overlap mean versus
shrinkage does not rescue a failed primary; one-sided clipping already matched
the candidate's overlap mean. Achieved FPRs are not exactly equal: primary
candidate range 4.967–5.711%, half 4.878–6.056%, matched 5.056–6.089%.

The mechanism **was active**: the cap clipped 22.6/24 rare descendants on
average, kept rho>r/2 for all 24, and retained 1.70 times half-radius
displacement energy. Original SPSDO's “negative closer than own anchor” rate
was 24.17% for rare descendants, versus zero for candidate **and half radius**.
This is not a true-noise rate or necessarily full-class 1NN error. All three
corrupted originals and all 24 corrupted-anchor descendants remained in every
q480 arm. All originals and exactly 480 additions were retained; training
removal, extra rows and extra classifier fits cannot explain operator contrasts.

This is a **standalone component stress test**, with uniform original-anchor
quotas, not full PPEO. It does not show that PPEO would assign these rare rows
to SPSDO; that activity must be checked in any end-to-end study. No tuned UGO,
EC-EBO or G-SMOTE benchmark was run here. The evidence supports rejecting this
operator's promotion in the tested setting, not a universal impossibility or
equivalence claim. Better empirical geometry alone did not establish useful
incremental learning over ordinary shrinkage.

## 6. Selection and implementation/evaluation boundary

| Candidate | Supporting evidence / distinction | Feasibility and main risk | Current decision |
| --- | --- | --- | --- |
| Parent-support cleaning | Generic dependence demonstrated; local cap differs from static weights, but simpler controls suffice. | Small prototype; sparse genuine groups and corrupt anchors remain indistinguishable. | Remains rejected. |
| Remove CRN's cluster ceiling | Exact source prior; multiple-cluster alternatives already exist; prediction harm unmeasured. | Easy adaptation; weak novelty and no attributed residual deficit. | Do not call this a new method. |
| Directional SPSDO region | Published geometric limitation; exact coherent operator differs from inspected code; novelty incomplete. Active probe fails simple-control benefit gate. | Undergraduate-feasible; small-sample geometry is not population support, and shrinkage may suffice. | **Do not implement as project extension on this evidence.** |

If independently established evidence later reopens this mechanism, the
decisive confirmation must first show full-method SPSDO activity, then freeze
the **same training-only prototypes, anchor IDs, quotas and non-SPSDO samples**
across operator swaps. Use new seeds **542–561** or independent data, not these
five pilot seeds. Keep F600/V2000 labeled access; no large selection set.
Primary would be minimum true-subgroup recall at validation target FPR .05,
requiring the lower paired-seed 95% interval limit to exceed a +.05 practical
margin; require AP loss
no greater than .005 and FPR increase no greater than .005. Report absolute
FPR, common-group performance, failures and calibration uncertainty.

Necessary comparisons are half radius, energy-matched shrinkage, one-sided
clipping, ROS, weighting, NORES, ordinary SMOTE, tuned GeometricSMOTE, a verified
Voronoi method, audited EC-EBO, and unchanged/validation-tuned UGO. Full PPEO
requires clarified code and a declared faithful interpretation. Keep data,
generation counts, classifier capacity and tuning opportunities comparable;
report different computation and all shortfalls. A geometry-only safety gain,
recovery by shrinkage/weighting, inactive SPSDO, or no +.05 held-out improvement
rejects further method development. A positive result would still need the
exact Voronoi novelty check and external confirmation.

Planning cost for an operator confirmation is roughly 240–480 RF100 fits
over 20 seeds/two regimes, plus the explicitly budgeted full-method controls;
approximately minutes at this pilot's scale. Full PPEO is much heavier:
15 particles×30 iterations already implies about 450 fitness fits per fold,
before archive/final fits. Do not silently equate those budgets or launch the
confirmation: this pilot failed its prerequisite. Production integration,
unit/failure tests and the full repository suite belong only to an approved
subsequent implementation stage.

## Provenance and preservation

The original 130-fit diagnostic remains under ignored
`results/question-b-recent-paper-diagnostic-v1/`; its record hash is unchanged.
The new ignored bundle `results/question-b-spsdo-geometry-probe-v1/` contains
the pre-outcome protocol/source, all arrays/anchor schedules, per-seed metrics,
paired contrasts, verification and the failed screening decision. Independently
recomputed means/hashes matched; quota, displacement-energy, geometry,
original-retention and validation-threshold checks passed. No further fits
were used for verification. Original implementations, frozen manifests and
historical CSV/JSON artifacts were not changed.

```text
Supplied PDFs, SHA-256:
CRN      1ef0edcba684b618266cbf198db9a9f236b16b09a15f8da8f0feda613f983cf3
HDBSCAN  6680bf5ab6c52367cbe2430d5ffc4a73d8f084fed813f68229c60fc015aa55f0
PPEO     822e5a66ad16c49815b47762125fe81765b9e21de7d9aa270869538cdbac888c
New probe protocol JSON:
0fd1a599b6742246f04df54fd71d10426c49ac182478eb9e7056c2a49a1f10e1
New probe source:
aca67ddd9fe0d24652c99f6cb478bd12751ed7e4e1654502788dfc91f685db48
New probe records:
d163a9bdd713defd86f1a05d092969820a55466307c135c4de4484d748c513aa
```

In plain language, the proposed geometric change would keep synthetic jitter
centered while limiting it in directions blocked by nearby negative examples.
It leaves more room sideways than shrinking every direction equally. That
difference worked geometrically, but it did not improve rare-group prediction
beyond simple shrinkage here. The full papers make the limitation and the
negative result more precise; they do not justify declaring a new sampler.
