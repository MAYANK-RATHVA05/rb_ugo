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
