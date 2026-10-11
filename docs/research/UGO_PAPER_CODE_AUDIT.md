# UGO paper-to-code audit: Stage 4A-R

Source [U]: Sağlam & Cengiz, *Uncertainty guided oversampling: A classifier-specific
approach for effective handling of class imbalance in machine learning*,
Knowledge-Based Systems 342 (2026), 115886, DOI
[10.1016/j.knosys.2026.115886](https://doi.org/10.1016/j.knosys.2026.115886).
Supplied PDF: `papers/base/1-s2.0-S095070512600612X-main.pdf` (ignored).
SHA-256 is in [the resource inventory](../../data/manifests/resource_inventory.csv).
Page citations are **PDF page numbers**, equal to printed article pages here.
Methodology inspected: pp5–8, §§3.1–3.4, Tables 2–4; experimental setup pp8–10.
No author code or separately supplied supplement was available.

This supersedes the earlier inaccessible-paper assessment. The historical
implementation is preserved in Git at `2c54d3d`; changes below are explicit
source-based repairs, not a new oversampler. Full-source access does **not** mean
all algorithm conventions are uniquely specified or published results reproduced.

## Operation audit

| Operation / source | Documented behavior | Code / test evidence | Finding and significance | Action |
| --- | --- | --- | --- | --- |
| MMA, p6 §3.2 | Highest competing-class probability minus true-class probability, in [-1,1] | `src/uncertainty.py:modified_margin_uncertainty`; exact/reversed-column tests | MATCH for binary classes; 1-2P(true) identity. Label-aware disagreement is not an identified uncertainty decomposition | No equation change |
| Noise pre-pass, pp7–8 §3.4/Table 2 | Neighbor majority determines noise; noisy rows excluded from generation | `noise_pre_detect`; self-exclusion/coincident-row/tie tests | MATCH for odd k and distinct majority; AMBIGUOUS self inclusion and tied votes. Current index self-exclusion and strict opposing majority are explicit conventions | Retain, disclose |
| Scoring pool, p7 §3.4 | Compute uncertainty for non-noisy samples | `UGO.fit_resample`; fixed original-clean pool tests | AMBIGUOUS whether later synthetics enter pool; initial pre-mask versus re-detection is not specified after augmentation | Retain original-only pool; not uniquely verified |
| Class choice, p7 §3.4 | c=argmax_l mean U_l | `UGO.fit_resample`; majority-selection tests | MATCH. Paper does not force minority-only selection | Preserve |
| Seeds, p7 §3.4 | n_select=n^(c)*lambda_select; choose highest U from class c | `informative_seed_indices`; stable ranking/noise-exclusion tests | AMBIGUOUS whether n^(c) counts all original class rows or eligible clean rows after filtering; integer rounding absent | Retain ceil fraction of eligible original rows, documented convention |
| Batch, p7 §3.4 and p8 Table 4 step 2 | n_step=n*lambda_step, initialized before loop | `UGO.fit_resample`; new fixed-total-batch test | DEVIATION in old code: ceil(lambda_step*current selected-class size). This changes intensity and makes it grow by selected class | Correct to fixed ceil(lambda_step*original total); rounding is an engineering convention |
| UGO-ROS, p8 §4.2 / Table 4 | R generates around selected informative points; ROS included as a resampling variant | `_generate`; selected-seed copy tests | MATCH in copying selected seeds; AMBIGUOUS sampling probability/multiplicity details not specified | Retain uniform replacement draws; no ordinary ROS substitution |
| UGO-SMOTE, p8 §4.2 / Table 4 | R receives selected seed rows and X/Y; SMOTE neighbor count 5 | `_generate`; exact interpolation reconstruction | MATCH for interpolation/selected seeds and k=5. AMBIGUOUS whether neighbors must be clean or may include noisy original same-class rows | Retain original pre-clean same-class neighbors; disclose |
| Final noise detection, p8 Table 3 | Neighbors queried from generated X^P to original X; compare selected label S | `noise_final_detect_and_relocate`; reference-spy tests | MATCH for original reference voting; tied/effective-k behavior unspecified | Retain original references and diagnosed k clamping |
| Relocation, p7 §3.4 / p8 Table 4 | Move generated noise to nearest non-noisy row of selected class | Same function; coordinate/destination tests | MATCH. Table 3 alone outputs mask/NN; relocation appears in text/Table 4 | Preserve actual copying of nearest clean anchor |
| Welch stop, p6 §3.3 | Fail to reject equality of class mean uncertainties using Welch ANOVA | `welch_stopping`; two-group F=t²/SciPy tests | MATCH for defined binary test. Non-rejection is not equality. Undefined/too-small handling absent | Preserve separately labelled termination safeguards |
| Cap, p6 §3.3 versus p8 Table 4 | Two conflicting thresholds, detailed below | `generation_cap`; new policy tests | AMBIGUOUS source conflict; old real expression matched §3.3 but floor stopped early for fractional threshold | Default explicitly to §3.3; optional Table 4 policy; ceil is minimal integer meeting threshold |
| Last-batch clipping | Table 4 checks cap before a full batch, potentially overshooting | Cap/truncation tests | DEVIATION retained as declared finite-budget safeguard; not described as a paper rule | Keep clipping and report policy |
| Retraining, p5 §3.1 versus p8 Table 4 | Prose updates labelled data/retrains on augmented set; Table 4 never reassigns X/Y inside loop | Cloned growing-fit tests | AMBIGUOUS pseudocode versus prose. Current augmented fits align with §3.1 | Retain augmented fitting and original reference pools |
| Parameters, pp6–8 | phi=2.5, lambda_select=.5, lambda_step=.1, k_pre=5, k_after=1; SMOTE k=5 | Constructor/default tests | MATCH for stated values. Alpha .05, ceil rounding, max_iter=100 and tie order are separately declared conventions | Do not call every default an author-specified value |
| Epistemic/aleatoric interpretation, pp6–7 | Authors assume limited aleatoric effects after neighbor-noise exclusion; explicitly acknowledge estimates do not separate components | Local MMA/noise implementation | MATCH as a heuristic interpretation; no formal decomposition or safe-membership guarantee established | Correct documentation; do not promote interpretation to theorem |

## Exact cap conflict and selected implementation

Section 3.3, p6 stops when:

$$
n_{syn}\ge\phi K\max_l n^{(l)}-n.
$$

Table 4, p8 instead loops while:

$$
n_{syn}<\phi\max_l n^{(l)}.
$$

These are not generally equivalent. For original counts (90,10), phi=2.5, they
permit 350 versus 225 synthetic points, respectively. Neither is silently called
the definitive author implementation. `cap_interpretation='section_3_3'` is the
default named source choice; `'table4'` allows comparison. Original counts remain
frozen. The minimal nonnegative integer threshold is `max(0, ceil(expression))`:
flooring a real threshold 1.6 would stop at 1, before the stated inequality holds.
The paper does not specify its numerical rounding or batch clipping convention.

## Before/after and validation meaning

Old batches depended on the selected class's current count. New nominal batches
use the original total and are fixed across iterations; only the last budget
clip can shrink a batch. The existing 120-row artificial demo now generates 12
points for each variant, instead of the earlier 3 ROS / 6 SMOTE. Both terminate
at Welch non-rejection. These are **software diagnostics**, not predictive scores
or paper-result reproduction. Historical tests for demo counts were updated;
new independent regressions assert the source's original-total batch basis and
both cap policies. Integer-threshold tests now distinguish ceil from floor.

Public `fit_resample(X,y)` remains unchanged. A new optional constructor parameter
`cap_interpretation` and keyword-only `generation_cap(..., interpretation=...)`
select the documented cap alternatives. Existing defaults use the §3.3 expression.
Opt-in remains required; diagnostics identify `source_audited_with_ambiguities`
and revision `stage4ar_fixed_original_total_batch`. No new DRA-UGO scores, held-out
acceptance or pseudo-labelling were added. The old implementation is inspectable
in Git rather than duplicated as a competing baseline hidden in the code.

## Real-data smoke status and remaining evidence

All eight full files load. However, **no supplied archive contains train/test
folds**. The requested first official training folds for Ecoli2, Yeast3 and Ecoli4
are absent. The offline smoke script writes six blocked rows and exits 2 without
fitting UGO or substituting a full file/generated split. See
[DATASET_VERIFICATION.md](DATASET_VERIFICATION.md). Corresponding test data are
never a tuning source; metadata checks of supplied folds would not be experiments.

Still needed: source-provided folds with provenance, author code/supplement or
clarification of cap, pool, count and neighbor conventions. No unique complete
paper-faithful baseline or published-result reproduction is claimed.
