# A bounded diagnostic of paired covariance clipping for rare minority subgroups

## Decision and scope

**Decision: do not advance this candidate to a full proposal.** In the registered
diagnostic its additional rare-recall gain over the tuned sphere was only
0.19 percentage points, and paired rejection achieved similar prediction with
less support intrusion. This is an opt-in investigation of the **small-prototype
SPSDO generation component**, not a reproduction of full PPEO or its published
results. Existing UGO, BFCH-SMOTE, EC-EBO, SSHR, historical
results and inactive designs are not changed or reopened.

The initial checkout was clean at `3e5b713c773aa3dbaec8f7224d9cb85b9947b960`.
`Question_B_Paper_Review_and_Extension.md` was absent from the checkout, fetched
references, Git path history and supplied attachments. Its branch/path/text was
requested; its contents and any candidates closed only in that file cannot be
claimed as reviewed. The user's explicit construction, the three supplied PDFs,
and the repository's existing scope rules define this investigation.

## Verified target and what its guarantee means

The [direct source audit](SOURCE_AUDIT.md) records the verified titles, DOIs,
uploaded-file SHA-256 hashes and public-release limitations for:

- PPEO: *Pure prototype-based evolutionary imbalanced oversampling for small
  disjuncts*, DOI `10.1016/j.neucom.2026.133891`.
- CRN-SMOTE: *Addressing imbalanced data classification with Cluster-Based
  Reduced Noise SMOTE*, DOI `10.1371/journal.pone.0317396`.
- SMOTE-PCA-HDBSCAN: *A SMOTE PCA HDBSCAN approach for enhancing water quality
  classification in imbalanced datasets*, DOI `10.1038/s41598-025-97248-0`.

PPEO Section 3.1, pp. 4–5, defines PHHC and its small prototypes. They are graph
components of minority points remaining inside majority-labelled clusters,
not simply clusters below a size cutoff. Section 3.2, p. 5, Eq. (8), draws
`x_new = x + delta*u`, where `delta ~ Uniform(0,r_x)` and `r_x` is the distance
to the nearest original observed majority point. Theorem 1, p. 6, specifies a
uniform surface direction independent of the radius. This is **uniform radial
distance, not uniform volume** in a ball. Sections 3.3–3.4, pp. 6–7, additionally
require evolutionary prototype allocations and a prototype-aware final decision
strategy; neither is evaluated here.

Section 4.7, p. 16, explicitly says that rigid hyperspheres are a poor
approximation in highly anisotropic clusters. They may underestimate minority
regions or intersect the true majority decision boundary. This is the verified
motivation for a directional replacement, not evidence that this particular
replacement improves prediction.

The public PPEO release at `zhaoujs/PPEO`, commit
`4d94423a86631ff1852547a1e9b776b73fe092cc`, lacks the imported PHHC and point
generation helpers. Consequently the reference here is a labelled paper
Algorithm 1 interpretation and paper-specified SPSDO, not verified author code.
The visible release samples anchors uniformly inside weighted prototypes. Here
the prototype map, small flags, one declared allocation and every anchor quota
are frozen across comparisons; evolutionary allocation is intentionally absent.

Four claims must be kept separate:

| Property | What is established |
|---|---|
| Preserve an anchor mean | Original SPSDO preserves it in expectation. Paired emission preserves each realized finite anchor mean. |
| Preserve the minority distribution | Neither construction does this: covariance, support, mode weights and density can change. |
| Avoid observed majority points | A positive-radius SPSDO draw strictly inside its ball avoids those coordinates. The proposed constraints additionally compare distances to every original majority site. |
| Avoid the unknown true majority region | Neither finite observed-site geometry nor a zero perturbation mean provides this guarantee. |

CRN-SMOTE's one/two-cluster denoising premise (pp. 9–10) can be problematic for
genuine additional modes; this is a mechanism concern, not a demonstrated
failure here. HDBSCAN filtering of synthetic data (pp. 3–5 of its paper) also
does not certify labels. Its Table 5, p. 9, contains inconsistent F1 figures:
for example precision 46.15% and recall 28.57% imply about 35.29% F1, not 91.54%.
No printed performance figures from any paper enter this diagnostic's metrics
or validation logic.

## Exact candidate and pseudocode

All preprocessing is fitted on original fitting observations only. For each
small prototype `j`, estimate its covariance using sklearn's established
Ledoit–Wolf identity-target shrinkage estimator, normalize `C_j` to trace `d`,
and use its symmetric positive-definite square root `L_j` so
`L_j L_j^T = C_j`. This is not an author-specified PPEO covariance rule.

For this diagnostic, fewer than `max(3,d+1)` original prototype observations,
deficient centered empirical rank, zero trace, nonfinite geometry or a failed
factorization cause quota-preserving duplication with a recorded reason. The
full-rank gate is a conservative design choice; shrinkage itself does not
require full empirical rank. No synthetic rows update shape estimates.

For each original anchor `x`, with fixed integer quota `q_x`, set

\[
v_0=\delta L_j u,\quad a_y=y-x,\quad
c=\min\left(1,\min_{y:v_0^Ta_y\ne0}
\frac{\eta\|a_y\|^2}{2|v_0^Ta_y|}\right),\quad v=cv_0,
\]

with `eta=0.9`, `delta ~ Uniform(0,r_x)` and uniform unit `u`. If the inner
constraint set is empty, use `c=1`. The algorithm is:

```text
freeze original-only preprocessing, prototype IDs, small flags and anchor quotas
generate the common large-prototype branch once; reuse identical rows
for each small prototype:
    fit and trace-normalize Ledoit–Wolf covariance on its original fitting rows
    for each original anchor x and its fixed quota q:
        if any original majority coordinate equals x:
            append q copies; record unresolved conflicting labels; certify no margin
        else if shape estimation failed:
            append q copies; record the fallback reason
        else:
            repeat floor(q / 2) times:
                draw delta and u independently; set v0 = delta * L * u
                compute one shared c using ALL original majority observations
                append x + c*v0 and x - c*v0
            if q is odd: append x; record an intentional odd-quota copy
```

For each distinct original majority `y`, clipping gives

\[
2|v^Ta_y|\le\eta\|a_y\|^2,
\qquad
\|x\pm v-y\|^2-\|v\|^2
\ge(1-\eta)\|a_y\|^2>0.
\]

Thus both signs are nearer the anchor than every noncoincident original
majority point **in the fitted standardized coordinate space**. This is a centrally symmetric margin core of the
anchor-versus-majority Voronoi halfspaces. It is radial clipping, not Euclidean
nearest-point projection onto that intersection. At `y=x`, the formal `0<=0`
constraint is vacuous: strict proximity is impossible and the conflict remains
unresolved even when the quota is filled with copies.

Every pair sums to `2x`; an odd copy preserves the same mean. Across the paired
small-prototype anchors, however, the generated center is
`sum(q_x*x)/sum(q_x)`, which need not equal the original minority center. The
unchanged large-prototype SMOTE branch has no such exact centering claim. Before clipping,
`Cov(v0)=r_x^2*C_j/(3d)` and `E||v0||^2=r_x^2/3`, matching the original sphere's
average squared radius while changing direction. Clipping changes that radial
law and covariance, often accumulating mass on a constraint boundary. Exact
mean preservation therefore implies neither correct labels nor better recall,
AP, calibration or minority-distribution fidelity.

## Closest precedents and remaining contribution uncertainty

The [prior-art audit](PRIOR_ART.md) records exact accessible source commits and
distinguishes author implementation evidence from full-paper verification.

| Precedent | Established overlap with this idea | Narrow remaining difference to investigate |
|---|---|---|
| G-SMOTE, DOI `10.1016/j.ins.2019.06.007` | Deformable/truncated generation regions and majority-informed region selection | This candidate uses a per-prototype multiaxis covariance and one all-original-majority scale shared by both signs. |
| EMDO, user-specified evolutionary ellipsoidal lead | Ellipsoidal/evolutionary generation is already a precedent; its primary source could not be verified through the available network | No claim that EMDO lacks this construction is justified. Exact comparison remains unresolved. |
| OHIT, author repository `zhutuanfei/OHIT` | Per-mode shrinkage covariance and covariance-shaped generation | Individual fixed anchors, shape-only trace normalization, original-majority slabs and paired radial clipping; OHIT DOI/full-paper details remain unverified. |
| SWIM and Voronoi-region generation | Majority-aware geometry and Voronoi-based oversampling already exist | Analytic symmetric halfspace retraction avoids building a complete tessellation, but that is not proof of originality. |

The precise candidate is a **quota-preserving small-prototype SPSDO replacement
using shrinkage-shaped proposals and shared radial clipping into a symmetric
observed-majority margin region**. Covariance, ellipsoids, shrinkage, symmetric
sampling, majority awareness and Voronoi regions are not novelty claims.
Primary scholarly hosts were blocked by the configured network proxy, so this
is not an exhaustive novelty search and supports no positive originality claim.

Paired rejection already supplies the same feasible region and exact anchor
mean. It conditions raw ellipsoid proposals on feasibility; clipping instead
moves excessive radii to their directional limit. The diagnostic must show
additional value from that transport, rather than from pairing or a smaller
effective radius alone.

## Frozen diagnostic

[protocol.json](protocol.json) was committed as `54f2d49` before classifier
outcomes. Source/runtime/PDF identities are additionally frozen before execution.
The single nonlinear learner is HistGradientBoostingClassifier with 60 boosting
iterations, seven maximum leaves, minimum leaf size eight and fixed seed 42.
No method-specific classifier tuning is performed.

Eight fresh conditions cross elongation ratios `{1,6}`, orientations `{0,60}`
degrees and normalized majority offsets `{2.4,1.2}`. Compact uniform-area
ellipses provide genuine bounded supports: one main minority group and two rare
subgroups, with original fitting counts `80/8/8` and 480 majority observations.
The majority has separate local ellipses on either side of each rare subgroup.
Using the same affine transform for each corresponding minority/majority pair
controls local intersection area independently of elongation and orientation.
At offset 1.2 the overlap fraction is 0.284757 for each neighboring pair, or
0.569514 of each rare support in the union of its two disjoint majority
neighbors; at offset 2.4 it is zero. Non-corresponding cross-class support intersections are checked before fitting.
Validation and test draws are independent, at exactly five and fifty times the
fitting component counts, preserving class and subgroup proportions.

True subgroup IDs and analytic supports are evaluator-only: they do not enter
PHHC, covariance, screening, quotas, sphere selection, thresholds or the learner.
StandardScaler is fitted to original fitting data. Each condition freezes one
paper-interpreted PHHC map and one 384-row allocation from fixed inverse-size
prototype weights, then uniform anchor draws. These diagnostic weights are not
the paper's evolutionary optimum. The common large-prototype SMOTE rows are
identical across all six geometry arms. ROS copies the same complete anchor
quotas; NORES uses originals only.

The eight comparisons are original SPSDO, a validation-tuned smaller sphere,
local ellipsoid, original-only-screened ellipsoid, proposed symmetric clipping,
paired rejection under the identical slab constraints, ROS and NORES. Screening
requires a proposal to be nearer some original minority point than every
original majority point; it never screens against synthetic or held-out rows.
Screening and paired rejection permit 64 attempts before recorded duplication.

Sphere multipliers `{0.25,0.5,0.75}` are selected using **overall validation**
minority recall at the validation-selected FPR threshold, with AP and then larger
radius as tie breaks. The selected fitted model is reused. Every model's
threshold is the lowest score threshold with validation FPR at most 5%; the
achieved independent test FPR is reported, not forced to equal 5%. AP is an
independently implemented tied-score PR step integral, cross-checked in software
tests against sklearn. There are 64 final models and **80 total classifier
fits**, including sphere selection, within the requested limit.

The preregistered conservative development gate requires clipping to beat
**each** of tuned sphere, screened ellipsoid, paired rejection and ROS by at
least three percentage points in mean rare recall, with nonnegative rare-recall
deltas in at least six of eight blocks, no more than one percentage point mean
FPR increase or AP loss, and lower small-branch outside-minority-support rate than
both SPSDO and unscreened ellipsoid. This is a decision rule, not a statistical
significance test. One draw per condition cannot establish general superiority.

## Results and next action

All **80 actual classifier fits and 64 final records succeeded**. No settings,
source hashes or thresholds were changed after outcome inspection. The
[aggregate JSON](results.json), [condition CSV](condition_results.csv) and
[independent audit](validation.json) retain the paired evidence. Individual
anchor records, generated rows and uploaded PDFs remain outside Git.

The following are unweighted means across the eight conditions, with rare
recall averaging the two evaluator-only rare subgroups within each condition.
AP is the mean of condition AP values, not AP from pooled rankings. Outside-
minority and majority-only intrusion rates concern the small branch and include
fallback copies. Timings are descriptive means for the selected generator, in
milliseconds; they include shape fitting and common large-branch generation,
but exclude the separate invariant/support checks and classifier fitting.

A separate zero-classifier-fit replay regenerated all 56 resampling
method/condition outputs, matched fitting-data/prototype/quota hashes, and
independently recalculated support counts and actual squared-distance bounds for
both paired methods. Confusion arithmetic and the independent AP implementation
were also audited. Row-level prediction scores were not persisted, so AP was
not reconstructed again post-run from saved predictions.

| Generator | Rare recall % | Achieved FPR % | AP | Outside minority support % | Majority-only intrusion % | Generator ms |
|---|---:|---:|---:|---:|---:|---:|
| Original SPSDO | 72.25 | 4.79 | 0.9728 | 28.28 | 1.147 | 5.22 |
| Tuned smaller sphere | 73.00 | 4.84 | 0.9742 | 18.68 | 0.456 | 5.21 |
| Local ellipsoid | 72.98 | 4.64 | 0.9727 | 23.73 | 0.375 | 4.88 |
| Screened ellipsoid | 72.88 | 4.83 | 0.9734 | 21.28 | 0.033 | 11.50 |
| Proposed paired clipping | 73.19 | 4.79 | 0.9719 | 19.58 | 0.065 | 6.74 |
| Paired rejection | 72.77 | 4.93 | 0.9712 | 17.69 | 0.033 | 7.81 |
| ROS | 69.56 | 4.16 | 0.9685 | 0.00 | 0.000 | 0.81 |
| NORES | 70.94 | 4.05 | 0.9732 | — | — | 0.00 |

Paired rare-recall results by condition are below; all entries are percentages.
Angle has no distributional effect when elongation is one, so those two angle
rows are independent draws from the same isotropic law, not evidence of an
orientation effect. “Overlap” identifies offset 1.2; “disjoint” identifies 2.4.

| Elongation / angle / overlap | SPSDO | Tuned sphere | Ellipsoid | Screened | Clipping | Paired rejection | ROS | NORES |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / 0° / disjoint | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 98.12 | 97.62 |
| 1 / 0° / overlap | 50.25 | 47.50 | 49.62 | 48.88 | 49.50 | 50.62 | 49.25 | 51.12 |
| 1 / 60° / disjoint | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 92.75 | 92.12 |
| 1 / 60° / overlap | 36.88 | 36.75 | 30.50 | 29.50 | 31.75 | 31.63 | 30.25 | 28.50 |
| 6 / 0° / disjoint | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 90.62 | 94.50 |
| 6 / 0° / overlap | 35.38 | 43.50 | 49.62 | 50.50 | 50.12 | 46.00 | 50.50 | 49.12 |
| 6 / 60° / disjoint | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 91.12 | 98.12 |
| 6 / 60° / overlap | 55.50 | 56.25 | 54.12 | 54.12 | 54.12 | 53.88 | 53.88 | 56.38 |

The registered primary comparisons give the following **clipping minus
control** deltas, in percentage points. FPR is measured on the independent
test set after validation-only threshold selection; lower FPR is better.

| Control | Rare-recall delta | FPR delta | AP delta | Nonnegative rare-recall blocks | Required 3 pp recall gain? |
|---|---:|---:|---:|---:|---|
| Tuned smaller sphere | +0.19 | -0.05 | -0.24 | 6/8 | No |
| Screened ellipsoid | +0.31 | -0.04 | -0.15 | 7/8 | No |
| Paired rejection | +0.42 | -0.14 | +0.07 | 7/8 | No |
| ROS | +3.62 | +0.63 | +0.34 | 7/8 | Yes |

The FPR/AP safeguards and the nonnegative-block criteria pass, but the
additional-recall criterion fails against **three of four primary controls**.
Clipping improves mean rare recall over original SPSDO by only 0.94 pp and over
NORES by 2.25 pp; the latter also has 0.74 pp higher mean FPR. Its advantage over
ROS alone does not isolate a benefit from clipping: all six geometry arms use
a common large-prototype SMOTE branch whereas ROS duplicates that branch too.
The decisive matched-geometry comparisons are the smaller sphere, screening
and paired rejection. Those simpler controls remove the apparent case for a
practically additional predictive benefit.

### Geometry, fallbacks and runtime

- Every resampling arm emitted exactly 384 rows per condition, with identical
  original anchor quotas and reference prototype IDs. Small-branch quotas were
  nonzero for **both** true rare subgroups in every condition; the two rare
  groups account for all 3,000 small-branch rows across conditions.
- All 3,000 clipped small-branch rows, including copies, passed the observed-
  majority distance and paired-mean checks in standardized coordinates. The
  largest anchor-mean error was `1.33e-15`; minimum slab slack was `-2.22e-16`,
  within the declared `1e-10` floating-point tolerance. There were no observed
  exact-coordinate label conflicts; separate failure tests cover them.
- Each covariance arm used **1,492 too-few-observation fallback copies**:
  49.73% of all 3,000 small-branch rows, and **97.39% of the 1,532 rows in overlap
  conditions**. Clipping and paired rejection each also emitted 42 intentional
  odd-quota anchor copies. No rejection-exhaustion or numerical fallback occurred.
- Clipping proposed 733 pairs, clipped 278 of them, and emitted 1,466 non-copy
  geometric rows. Paired rejection needed 1,216 proposals for the same number
  of pairs. The mean generation times were 6.74 versus 7.81 ms, too small and
  hardware-sensitive to support a useful runtime claim. Screened ellipsoids
  averaged 11.50 ms. Selected-model classifier fitting averaged roughly
  16–19 ms; all 80 fits, including sphere selection, totalled about 1.42 seconds
  of measured fit time, excluding prediction/evaluation and test-suite work.
- Clipping generated 575/3,000 small-branch points outside true minority support
  and two in majority-only support. Its geometric certificate therefore plainly
  does not certify true support membership. Paired rejection had 519 outside
  and one in majority-only support; the tuned sphere had 550 outside. The
  percentage table uses mean condition rates rather than these pooled counts.
- Membership in majority support alone includes genuine class overlap: around
  41% of small-branch points satisfy that predicate even for ROS. It is not
  interchangeable with majority-only intrusion or a known incorrect label.

The disjoint geometry arms all reach 100% rare recall, so they cannot establish
a specific advantage for the proposed geometry. In overlap, PHHC fragmentation
leaves too few original points for a meaningful shape estimate in almost all
allocated cases. These ceiling and fallback limitations constrain the strength
of a negative conclusion: this is not proof that clipping can never help. They
are nevertheless part of the requested fixed-reference small-prototype test,
not grounds to change the fallback, allocation or data after seeing results.

### Concrete decision

**Do not develop a full PPEO extension proposal from this result.** The specified
clipping mechanism has not shown the registered additional benefit beyond
validation-tuned sphere, original-only screening and paired rejection. Its
mathematical properties are useful diagnostics, not predictive guarantees or
a demonstrated novel contribution. EMDO primary-source details and the missing
Question B review also prevent a complete novelty/closed-candidate audit.

**One next action:** close this candidate at the diagnostic stage and retain
its frozen negative/inconclusive evidence; do not expand training or alter the
existing methods. A complete proposal and tuned UGO/weighting/EC-EBO evaluation
plan are not justified by the conditional development gate.

### Software validation and reproduction

Focused geometry/reference/runner checks: **68 passed**. Full repository suite:
**806 passed**, zero failures/errors/skips. New code is opt-in; all original
method implementations, requirements, historical manifests and result artifacts
remain byte-identical to the initial checkout.

With the prepared Python 3.11 environment, from the repository root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.runtime/matplotlib" XDG_CACHE_HOME="$PWD/.runtime/cache"
.venv/bin/python -m pytest -q tests/test_ppeo_geometry.py tests/test_ppeo_diagnostic.py
.venv/bin/python -m scripts.diagnose_ppeo_geometry \
  --config docs/ppeo_rare_subgroups/protocol.json \
  --identity docs/ppeo_rare_subgroups/frozen_identity.json \
  --output results/ppeo-rare-subgroup-independent-repeat
```

The runner verifies frozen source/runtime/protocol identities and refuses to
overwrite an existing output directory. It does not reproduce full PPEO. The
command documents an independent future repeat; this task ran the registered
diagnostic once and did not spend additional fits checking repeatability.
Protocol preregistration commit: `54f2d49`; implementation/identity freeze:
`67437b9`. The aggregate artifacts preserve the exact hashes.
