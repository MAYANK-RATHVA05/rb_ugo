# Stage 4A novelty decision

**Decision: no defensible methodological novelty established. Do not implement
a new oversampler or claim an RB-UGO contribution yet.** This decision is supported
by verified related prior art and substantial missing source evidence; it is not
an exhaustive proof that no useful future contribution exists.

The deliverables are [the prior-art matrix](prior_art_matrix.md),
[mathematical gap analysis](mathematical_gap_analysis.md), and
[two candidate definitions](candidate_hypotheses.md). Reference identifiers below
resolve to the matrix's linked, access-labelled source register. No source or test
code, original-UGO behavior, data, metrics implementation, or dependency was changed.

## A. What is already solved?

Several **ingredients and formulations** are established; the complete requested
problem is not thereby proved solved:

- Threshold selection can change decisions while leaving ranking curves intact
  [S1]. A recall improvement over a default 0.5 cutoff is inadequate evidence of
  better discrimination or the need for synthetic samples.
- Validation-guided learning of example importance, including class-imbalance
  settings, is explicit in Ren et al.'s bilevel objective [R, §3.1 Eqs. (1)–(2)].
- Model-dependent marginal training-data value, **group** value, and guidance for
  acquiring valuable new data are explicit in Data Shapley [D, §§2,3.2,4.1,4.5].
- Curvature-adjusted effects of infinitesimal training-example changes on another
  example's loss are explicit in influence functions [I, §2.1 Eqs. (1)–(2)].
- Difficult-point emphasis, boundary variants, clustering and cluster-sparsity
  allocation already exist in verified oversampling implementations [S5, S6].
- Self-training and graph propagation/spreading already accept genuinely separate
  unlabelled features, under distributional assumptions [S3, S7]. They do not
  make their model-assigned labels observed ground truth.

These verified precedents defeat a broad claim that combining uncertainty,
subgroup need, validation benefit and unlabelled support is sufficient novelty.
They do not prove that any particular combination is identical to one of the
six requested papers, whose methods remain unavailable.

## B. What remains unverified?

1. **All six requested publications**, including their title/DOI metadata,
   abstracts, exact equations, seed pools, pseudo-label use, overlap/intra-class
   treatment, marginal-benefit evaluation, threshold comparisons and guarantees.
   The exact supplied DOI/arXiv URLs returned proxy `403 Forbidden`, not paper text.
2. **Faithful original UGO reproduction.** The existing baseline is unchanged but
   explicitly provisional. Section 3.3's cap versus Table 4, noise reference sets,
   rounding, uncertainty pools and resampling settings still require the original
   source. Preserving code does not validate it as a faithful published baseline.
   See [the prior record](../ugo_reproduction.md).
3. Precedence for robust harmonic-extension intervals, anchor-compatible label
   sets, local augmentation influence, bilevel synthetic-data selection, and
   control-adjusted regional data valuation beyond the three verified papers.
4. Whether local synthetic-utility comparisons are identifiable with affordable
   minority audit counts, stable across seeds/learners, and replicable on new data.
5. Whether U contributes beyond its anchors, sample volume, graph construction
   and extra computation; density alone does not establish class membership
   ([gap analysis §6](mathematical_gap_analysis.md)).
6. Any empirical advantage, novelty, robust safety, or benchmark result. None was
   measured in Stage 4A. R's optimization and D's valuation guarantees do not
   establish an outer-test safety guarantee for our hypothetical generator.
   The audit also finds a concrete objection to the sufficiency of D's displayed
   uniqueness axioms; see the counterexample in the prior-art matrix. The
   published marginal/group-valuation mechanism remains relevant prior art.

## C. Which candidates should be rejected?

Reject **as present novelty arguments**:

- UGO + clustering or count/sparsity-based subgroup weights [S5, S6].
- UGO + pseudo-labelling, entropy or confidence-weighted U [S3].
- UGO + ordinary boundary-distance rejection [S5; exact relevance of the supplied
  boundary paper remains unverified].
- UGO + validation-tuned amount, or a renamed product of established scores
  [S4; R's validation-guided objective].
- “Data-limited region detection” inferred solely from high uncertainty, sparse
  density, no observed gain, or graph support. Counterexamples are derived in the
  [gap analysis](mathematical_gap_analysis.md).
- Candidate 2, anchor-compatible graph sensitivity intervals, as a standalone
  novel oversampler. Its core harmonic extension is established graph machinery
  [S3, S7]; an interval does not remove the unknown label-smoothness assumption.
- Candidate 1, control-adjusted regional intervention contrasts, **as an already
  novel algorithm**. It remains close to group data valuation, influence analysis
  and validation-guided optimization [D, R, I]. A standard statistical gate and
  matched controls are good evaluation practice, not sufficient novelty.

These rejections concern the current justification, not a ban on using established
components in a properly attributed study. Do not market engineering composition
as a novel mathematical principle.

## D. Which candidate, if any, is defensible?

**No candidate is presently defensible as a novel oversampling method.** Candidate
1 is defensible only as a limited, prospective **empirical diagnostic question**:
do training-region intervention estimates predict independent discrimination gain
beyond threshold, point-weight and sample-count controls?

Its output must remain conditional on the learner/generator/data allocation and
include inconclusive outcomes. The candidate's bounded-comparison analysis applies
only to a fixed finite family independent of its audit set; it is not a theorem
about intrinsic data limitation, synthetic correctness or outer-test superiority.
Its distinction from group Data Shapley and ordinary nested validation must be
demonstrated, not asserted. The six missing publications could eliminate the gap.

Candidate 2 may serve as an assumption-sensitivity/unlabelled-data ablation only
after source review. It should not be presented as a second novel component to
rescue an otherwise ordinary validation-selection method.

## E. Single strongest reviewer objection

> This is established model/data selection relabelled as detection of genuinely
> data-limited minority regions. Group data valuation and validation-guided
> reweighting already estimate conditional usefulness; the apparent effect may
> be threshold/weight changes or extra labelled selection budget, not a new
> mathematical mechanism or new minority information.

The direct prior-art basis is [D, §§3.2,4.1,4.5], [R, §3.1], [I, §2] and [S1].
No number of heuristic scores or a new acronym answers this objection. A fair
study must offer controls the same label/tuning/compute opportunities and preserve
the distinction between synthetic training locations and new observed labels.

## F. What result would falsify the proposed contribution?

There is no established contribution to defend; the testable usefulness hypothesis
is Candidate 1's prospective diagnostic/selection benefit. Predeclare target
distributions, learners, practical margins, label budget, seeds and comparison
rules. Evidence against that hypothesis would include:

1. Local utility estimates fail to predict fresh confirmed intervention gains
   better than count/uncertainty/sparsity or group-valuation controls.
2. With adequate power, an equivalence interval places outer discrimination/risk
   advantage within a predeclared negligible margin relative to tuned thresholds,
   weighting, point-weight ROS or validation-count search. Merely failing to reject
   equality in a small study is **inconclusive**, not a falsification.
3. Confirmed majority FPR harm exceeds the preregistered tolerance, or claimed
   subgroup benefits come from sacrificing other minority regions.
4. Matching the total labelled-data/compute budget removes the advantage.
5. The claimed unlabelled component has no replicated gain over no-U and ordinary
   graph/SSL controls, or overlap/shift systematically makes its accepted points
   harmful. This falsifies the claimed U benefit, not necessarily every part of
   the separate local-utility diagnostic.
6. The novelty audit finds the same target, intervention and estimation procedure
   already published. This defeats novelty even if the procedure works.

These are proposed outcomes, not observations. Known class-conditional synthetic
distributions can test overlap assumptions; their true latent classes are reserved
for a diagnostic oracle and never supplied to the proposed method. Confirmatory
measurement must execute the entire selected procedure on untouched outer data
[S4; D, §4].

## G. What evidence is required before implementation?

The following review gates must be met **before any new oversampler code**:

1. Obtain lawful copies or verified abstracts/previews for all six supplied
   publications. Validate identifiers/titles and fill the matrix with section/page
   citations. An abstract may support a broad objective, not an exact negative
   claim about absent controls or a full algorithm equation.
2. Reconcile the original UGO source against the preserved provisional baseline.
   Any subsequent fidelity corrections belong in a separate approved task and
   separate baseline version; never silently alter the comparison baseline.
3. Review the direct competing literature, including group Data Shapley, Ren's
   validation reweighting, influence-based acquisition/synthetic selection,
   KMeansSMOTE/ADASYN, robust graph SSL, and the supplied SSHR/SLEO/GDEO papers.
   The current source set is targeted, not an exhaustive search. Suggested
   queries are a future search plan, not searches claimed to have been executed:
   “validation-guided synthetic data selection”, “group data valuation minority
   subpopulations”, “influence functions synthetic augmentation”, and “harmonic
   extension uncertainty effective resistance”.
4. Write a precise proposed difference from the closest verified competitor.
   If the difference is only a validation loop, confidence threshold, new metric
   or new name, abandon the novelty claim.
5. Specify F/H/A/O and U provenance; group/time constraints; training-only
   preprocessing/regions; independent threshold calibration; a finite candidate
   family or fresh-block adaptive selection schedule; and total label/fit budgets.
   Fresh-block error spending such as delta_t=delta/[t(t+1)] has total at most
   delta, but each block must really be independent of the previous selections.
6. Justify minority audit counts and statistical power. The elementary bounds in
   Candidate 1 will often be wide for small disjuncts; do not promise a practical
   safety guarantee without showing a feasible budget. Resolve AP/all-pairs AUC
   inference separately rather than counting correlated pairs as independent.
7. Pre-register the controls listed in [gap analysis §7](mathematical_gap_analysis.md),
   U ablations, known-overlap regimes, minority-subregion recall, harm margins,
   reportable inconclusive outcomes, and untouched outer evaluation.
8. Obtain user review and explicit approval for the next stage. Stage 4A proposes
   definitions only and does not authorize a pilot experiment or implementation.

## Exact publication-access report

**Full-text verified from lawful publisher-repository PDFs:**

- Ren, Zeng, Yang & Urtasun (2018), *Learning to Reweight Examples for Robust Deep
  Learning*, ICML/PMLR. Methods, objective equations and stated convergence
  assumptions checked; supplementary proof audit remains outstanding.
- Ghorbani & Zou (2019), *Data Shapley: Equitable Valuation of Data for Machine
  Learning*, ICML/PMLR. Valuation equation, group extension, acquisition and
  evaluation separation checked; supplemental approximation details not read.
- Koh & Liang (2017), *Understanding Black-box Predictions via Influence
  Functions*, ICML/PMLR. Influence equations and convex/local assumptions checked;
  original supplemental proofs not read.

**Verified from abstracts only:** none. Publisher abstracts of the three above
were also verified, but they are classified at their stronger full-text level.

**Neither full text nor abstract verified:** UGO, SLEO, SSHR, GDEO, Don't Oversample
the Boundary, and Synthetic Augmentation in Imbalanced Learning at the supplied
identifiers. Their missing evidence is an explicit blocker, not a completed
six-paper verification claim. Pinned scikit-learn/imbalanced-learn source files
were checked as documentation/implementation evidence, not classified as papers.

## Review and synchronization

Only the four requested files under `docs/research/` belong in the Stage 4A PR.
The baseline source/tests/dependencies and previous documentation remain unchanged.
Review the evidence limitations first, then the derivations/controls, then whether
either conditional question merits further source work. If the conclusion remains
“no defensible novelty established,” that is the research result of this audit,
not a reason to rename an ordinary method.

Stop after Stage 4A and wait for user review.
