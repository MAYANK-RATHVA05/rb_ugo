<<<<<<< HEAD
# Original UGO reproduction: source-audited baseline with ambiguities

Updated in the explicitly authorized Stage 4A-R recovery task, 2026-10-11.
The original prompt-based Stage 3 record and implementation are preserved in
Git at `2c54d3d`. This document describes the current source-audited revision;
no DRA-UGO mechanism is implemented.

## Evidence and fidelity limits

The supplied full paper [U] is now accessible: Sağlam & Cengiz, *Uncertainty
guided oversampling: A classifier-specific approach for effective handling of
class imbalance in machine learning*, Knowledge-Based Systems 342 (2026),
115886, [DOI](https://doi.org/10.1016/j.knosys.2026.115886).
Methods pp5–8, §§3.1–3.4, Tables 2–4 and experimental setup pp8–10 were inspected.
The [operation audit](research/UGO_PAPER_CODE_AUDIT.md) records matches,
ambiguities, deviations and corrections. Source hashes and portable locations
are in [the inventory](../data/manifests/resource_inventory.csv).

Full-text access supersedes earlier network-only access failures. It does not
resolve contradictory cap formulas or unspecified integer/pool conventions.
Runtime still requires `allow_provisional=True`, emits a warning and records
`source_audited_with_ambiguities` plus revision
`stage4ar_fixed_original_total_batch`. No author code, faithful complete
reproduction or published performance reproduction is claimed.

## Verified equations and declared conventions

MMA is the maximum competing-class probability minus true-label probability
[U p6 §3.2]. For binary probabilities:

$$U(x,y)=1-2P(y\mid x)\in[-1,1].$$

It is signed label-aware disagreement, not entropy or an identified epistemic /
aleatoric decomposition [U pp6–7]. Probability columns follow `classes_`, with
validation for labels, finite values, normalization and reversed order.

Neighbor pre-noise filtering and post-generation voting/relocation follow
[U pp7–8, Tables 2–4]. Index self-exclusion and strict opposing-class majority
for tied votes are explicit conventions. Final voting uses original fitting
rows; detected generated noise is copied to the nearest original clean row of
the selected class. Effective neighbor counts are diagnosed, not hidden.

The class with highest mean MMA can be **the majority class** [U p7 §3.4].
Current scoring/seeds use original clean rows, with stable index tie breaking.
Selection size is `max(1, ceil(lambda_select * eligible_class_count))`;
the paper does not uniquely settle class-count basis or rounding.

The source sets `n_step=n*lambda_step` before the loop [U p7; Table 4 p8].
The corrected implementation freezes the original total n:

$$n_{batch}=\min(B_{remaining},\max(1,\lceil\lambda_{step}n_{original}\rceil)).$$

Earlier code used the current selected-class count. This definite mismatch is
repaired and independently tested. Ceil and last-batch clipping are documented
engineering conventions. ROS draws selected anchors uniformly with replacement.
SMOTE interpolates selected anchors with original same-class neighbors (not
necessarily selected or clean), using k=5 as in [U p8 §4.2]. Eligibility details
remain ambiguous; there is no automatic fallback to ordinary ROS/SMOTE.

Section 3.3 p6 stops at `n_syn >= phi*K*max(original_class_counts)-n_original`.
Table 4 p8 instead loops at `n_syn < phi*max(original_class_counts)`.
`cap_interpretation='section_3_3'` explicitly selects the former by default;
`'table4'` selects the latter. Nonnegative ceil gives the minimal integer reaching
the selected threshold. Historical floor rounding stopped before fractional
thresholds. Both expressions and their disagreement are tested. A final clipped
batch and `max_iter=100` are safeguards, not uniquely published rules.

Welch's binary two-group ANOVA is implemented through the equivalent Welch t
statistic (F=t²). Failure to reject is not proof of equal uncertainties [U p6
§3.3]. Undefined statistics and tiny groups terminate with separate diagnostic
reasons. Prose retraining on augmented data [U p5 §3.1] is followed; Table 4
never explicitly updates X/Y, so original reference pools remain declared choices.

## API, parameters and diagnostics

`UGO.fit_resample(X,y)` returns numerical arrays with original rows/labels first
and unchanged. Inputs and caller estimator are preserved; a clone is fitted each
iteration. Two nonempty classes and finite dense numerical features are required.
Typed categories are rejected; arbitrary numeric category codes are not meaningful
distances. Fit any preprocessing on fitting data only. The API cannot detect a
caller who already mixed held-out rows into X.

| Parameter | Default | Evidence/status |
| --- | --- | --- |
| lambda_select / lambda_step | .5 / .1 | [U p7]; rounding and eligible seed count declared |
| phi | 2.5 | [U p6]; conflicting source cap expressions |
| k_pre / k_after / k_smote | 5 / 1 / 5 | [U pp7–8]; effective clamping recorded |
| alpha | .05 | Declared statistical convention |
| random_state | 42 | Declared reproducibility convention |
| variant | ros | ROS or SMOTE, explicit choice |
| max_iter | 100 | Engineering termination bound |
| allow_provisional | False | Explicit opt-in required |
| cap_interpretation | section_3_3 | Named source choice; table4 also available |

The estimator must implement fit/predict_proba and sklearn cloning. Explicit
estimator/nested seeds are preserved; unset seeds are filled and reported.
`diagnostics_` records counts, original pre-noise, relocation counts, source cap
policy, nominal batch, effective parameters, class means, Welch statistics,
selected class, seed/neighbor/interpolation details, history and termination.
`estimator_` is the last scoring classifier, which may not have fitted a batch
just added; it is not a final evaluation classifier. No performance scores exist.

## Software verification and remaining reproduction work

```bash
.venv/bin/python -m pytest -q
.venv/bin/python main_stage3.py
.venv/bin/python scripts/ugo_smoke.py
```

Tests cover probability ordering, noise/reference/relocation semantics, exact
interpolation, cloning, original preservation, small groups, finite/category
validation, fixed batches, both caps, determinism and termination. The artificial
120-row demo now generates 12 points in each variant and stops at Welch
non-rejection; these are software diagnostics. Historical 3 ROS / 6 SMOTE counts
changed because of the batch correction, not a claimed improvement.

The real-data smoke command currently exits 2 with six blocked rows: supplied
Ecoli2/Yeast3/Ecoli4 archives have no official folds. It never substitutes full
files or generated splits. See [dataset verification](research/DATASET_VERIFICATION.md).
Author code/clarification and source-provided folds remain needed before complete
baseline fidelity or real-fold smoke claims. No benchmark, novel generator,
pseudo-labelling or predictive superiority is implemented in this stage.
=======
# Original UGO implementation and paper alignment

Stage 4R-2 recovers the source-audited batch and cap corrections from
[0866dc9](https://github.com/MAYANK-RATHVA05/rb_ugo/commit/0866dc936f231feef9df5e1538b8df206741735d).
The full uploaded article has been inspected. This remains a **provisional
reproduction with explicit paper ambiguities**, not reproduced benchmark evidence.

See the [reproduction specification](reproduction/SPECIFICATION.md),
[source register](reproduction/SOURCE_REGISTER.json) and
[current core alignment report](reproduction/CORE_ALIGNMENT.md).
The historical Stage 3 prototype is preserved at
[c2fa7f3](https://github.com/MAYANK-RATHVA05/rb_ugo/commit/c2fa7f333689df14c5ab4bb542500bf2676ac897).

## Source and status

Sağlam and Cengiz, “Uncertainty guided oversampling: A classifier-specific approach
for effective handling of class imbalance in machine learning,” Knowledge-Based
Systems 342 (2026), 115886, DOI
[10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886).
Page numbers refer to the uploaded 20-page PDF; its hash is in the source register.
No author implementation or separately supplied supplement is available.

Runtime still requires allow_provisional=True and emits a warning. Diagnostics
report source_audited_with_ambiguities, not unique author-code equivalence.
No RB-UGO/DRA-UGO, held-out batch acceptance, benchmark scoring or performance
claims are included.

## Data and classifier contract

UGO.fit_resample(X, y) accepts fitting data only and returns copied numerical
arrays with original rows/labels first and generated rows appended. Inputs and
the supplied estimator are not mutated or fitted in place.

Features must be finite dense real numerical values with meaningful Euclidean
distances. Missing/categorical features require externally fitted training-only
preprocessing. Typed categorical columns are rejected; integer category codes
cannot be recognized reliably from dtype.

The estimator must be sklearn-cloneable and implement fit/predict_proba. Each
iteration clones it, fits accumulated original plus generated fitting rows, and
scores the original pre-clean rows. Explicit estimator/nested random states are
preserved; unset ones are filled with the sampler's seed and recorded.

The returned estimator_ is the last internal scoring model. If its last iteration
added a batch, it has not fitted that final batch. A later evaluation stage must
train a fresh final classifier. No validation/test arguments exist, but the
software cannot identify held-out rows wrongly passed as fitting data.

## MMA and selection

Section 3.2, p6 defines:

    U_MMA(x,y) = max over k != y of P(k|x) - P(y|x)

For binary normalized probabilities this equals 1-2*P(true), ranging from -1
to 1. Larger values mean stronger disagreement with the known fitting label,
not entropy or a formal epistemic/aleatoric decomposition.

Probability columns use classifier.classes_, including reversed/string labels.
Invalid shapes, labels, nonfinite/out-of-range values and probability sums outside
absolute tolerance 1e-8 are rejected.

Sections 3.3–3.4 and Table 4 choose the class with **highest** mean MMA, which can
be the original majority. The introduction's “lowest” wording conflicts with
those detailed rules. Balanced counts are not themselves a stopping condition.

Uncertainty and seed pools remain fixed original pre-clean rows. Within the
selected class, descending MMA and stable original-index ties choose:

    n_select = max(1, ceil(lambda_select * eligible_original_class_count))

The paper writes n^(c)*lambda_select but does not settle whether that class count
includes noisy rows. Eligible count, rounding and score ties are declared
conventions. Synthetic points do not become scoring points or future seeds.

## Fixed original-total batches

Section 3.4, p7 and Table 4 step 2, p8 define n_step=n*lambda_step, initialized
before the loop. The nominal batch now uses original total size:

    n_nominal = max(1, ceil(lambda_step * N_original))
    n_batch = min(B_remaining, n_nominal)

The nominal batch is fixed across iterations and class choices; only final cap
clipping can shrink it. Ceil rounding, minimum one and clipping are declared
engineering conventions. The later conclusion mentions increasing generation
per iteration without specifying a different equation.

The historical prototype used the selected class's current size. On the existing
120-row example the corrected first batch is 12 rather than 3 for a 24-row class.
This changes generation/stopping trajectories; old generated counts are not
expected to remain identical.

## Noise and generation

Tables 2–3, p8 and Section 3.4, p7 support initial neighbor-vote detection,
generated-to-reference final voting, and relocation to a clean same-class point.

Current explicit conventions:

- Pre-detection uses Euclidean neighbors of original rows, excluding self by
  row index. Distinct coincident rows remain valid neighbors.
- Strict opposing majority marks noise; tied votes retain the assigned label.
- k_pre is clamped to N_original-1; k_after to N_original.
- Pre-noise is computed once. Noisy originals remain in classifier fitting data
  but cannot be seeds, uncertainty-pool rows or relocation destinations.
- Final votes use all original fitting rows. Relocation copies the nearest
  original pre-clean same-class row's coordinates. No eligible destination raises
  an error; samples are not silently dropped.
- Equal distances use original reference index order.
- ROS draws informative seeds uniformly with replacement and copies them.
- SMOTE draws informative seeds, selects among their nearest original pre-clean
  same-class neighbors, and generates (1-r)*seed+r*neighbor with uniform r in
  [0,1). k_smote=5 follows Section 4.2, p8; it is clamped to eligible count-1.
  Fewer than two eligible same-class rows stops with a diagnosed reason.

The paper does not uniquely settle every pool, vote-tie and draw convention.
Section 3.1 supports augmented retraining, while Table 4 omits the corresponding
in-loop X/Y reassignment. Original scoring/reference pools are explicit choices,
not a claim that all author conventions are now known.

Neighbor calculations allocate a full pairwise distance matrix and sort it.
This is suitable for small conformance checks; benchmark resource suitability
must be measured before full-scale experiments.

## Welch and generation stopping

Section 3.3, p6 uses Welch ANOVA and stops when equal class uncertainty means are
not rejected. For two classes, the two-sided Welch t-test has F=t² and the same
p-value, with F degrees of freedom (1, Welch df). A defined p>=alpha stops.

Non-rejection does not prove equal uncertainty or balanced predictive risk.
Too-small/doubly constant groups, invalid degrees of freedom and invalid
statistics produce an explicit undefined status. The loop stops separately with
undefined_welch_safeguard:<reason>, never a fabricated p=1.
One constant group remains valid when the other varies.

The article contains two distinct generated-sample thresholds:

| cap_interpretation | Source expression | Integer budget B used here |
| --- | --- | --- |
| section_3_3 (default) | nsyn >= phi*K*Nmax-N | max(0, ceil(phi*K*Nmax-N)) |
| table4 | Continue while nsyn < phi*Nmax | max(0, ceil(phi*Nmax)) |

N, Nmax and K=2 are frozen from original fitting data. Ceil produces the minimal
integer meeting a noninteger inequality. For counts 90/10 and phi=2.5, the
budgets are 350 and 225. For counts 3/2 and phi=1.1, the Section 3.3 real threshold
is 1.6 and its integer budget is 2, correcting the historical floor budget of 1.

Every batch is clipped to the remaining integer budget. Table 4's pre-batch test
can overshoot, so clipping remains a disclosed deviation. Cap termination uses
the existing provisional_generation_cap reason for compatibility. max_iter=100
is a separate engineering bound; all-noise/no-neighbor paths are also diagnosed.

## Parameters and diagnostics

| Parameter | Default | Meaning |
| --- | --- | --- |
| variant | ros | ROS or SMOTE; no silent fallback |
| lambda_select | 0.5 | Eligible-original seed fraction; rounding convention above |
| lambda_step | 0.1 | Fixed original-total nominal batch fraction |
| phi | 2.5 | Selected source threshold multiplier |
| cap_interpretation | section_3_3 | Named paper expression; alternative table4 |
| k_pre / k_after | 5 / 1 | Initial and final votes |
| k_smote | 5 | Same-class interpolation neighborhood |
| alpha | 0.05 | Defined Welch failure-to-reject threshold |
| random_state | 42 | Local default, not a verified author seed |
| max_iter | 100 | Engineering bound |
| allow_provisional | False | Explicit opt-in to remaining ambiguities |

Diagnostics record implementation_revision=stage4r2_fixed_original_total_batch,
nominal_batch_size, cap policy/budget, effective parameters, original/final
counts, noise/relocation counts, selected classes/seeds, Welch values and status,
drawn seeds/partners/interpolation steps, generated counts and stopping reason.
They contain no predictive-performance metrics.

    from sklearn.linear_model import LogisticRegression
    from src.ugo import UGO

    sampler = UGO(
        LogisticRegression(max_iter=1000, random_state=42),
        variant="smote",
        cap_interpretation="section_3_3",  # or "table4"
        allow_provisional=True,
    )
    # X_augmented, y_augmented = sampler.fit_resample(X_fit, y_fit)

## Validation and remaining reproduction

Run from the repository root using Python 3.11:

    .venv/bin/python -m pytest -q
    .venv/bin/python main_stage1.py
    .venv/bin/python main_stage2.py --data tests/fixtures/toy.csv --target class
    .venv/bin/python main_stage2.py --data tests/fixtures/toy.dat
    .venv/bin/python main_stage3.py

Existing checks cover MMA, votes, relocation, Welch, seed ranking, ROS copies,
SMOTE reconstruction, input preservation, cloning, random-state behavior and
failure paths. New conformance checks distinguish original-total batches from
both minority and majority counts, verify fixed batches and augmented fits across
iterations for both generators, test both cap policies and fractional/integral
thresholds, check clipped final batches, and reject invalid policy without stale
fitted state.

The demo remains an artificial 120-row/four-feature make_classification example
with seed 42 and logistic regression. It does not reproduce either paper
simulation or report held-out scores. Its diagnostics are software evidence only.

Still required: evidence for ambiguous conventions, authoritative missing
generator/comparator implementations, simulations, verified benchmark variants,
classifier settings, training-only preprocessing, evaluation, statistical
analyses and paper-result comparison. Stop after the requested stage; later
stages require separate authorization under AGENTS.md.
>>>>>>> origin/main
