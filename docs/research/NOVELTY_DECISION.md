# Stage 4A-R novelty decision

**No defensible methodological novelty established.** The six full papers
strengthen that conclusion: B already provides the remedy framing and matched
category interventions; T provides augmentation mismatch theory and size tuning.
No new DRA-UGO algorithm, pseudo-labelling or benchmark was implemented.
Identifiers resolve to [the matrix](PRIOR_ART_MATRIX.md); access limits are in
[REFERENCE_STATUS.md](REFERENCE_STATUS.md). DRA-UGO remains a working title.

## A. What is already solved?

Existing **ingredients and formulations**, rather than the complete general
problem, include:

- Classifier-specific uncertainty generation, noise handling and Welch stopping
  [U pp5–8]; this is the original comparator, not our contribution.
- Minority subcluster allocation / majority overlap cleaning [L pp4–7] and actual
  U confidence acquisition with no-U ablation [H pp4–6,11–12]. L has no separate
  U input; H does. Neither establishes true membership from density alone.
- Threshold-recoverable / estimated data-reducible / estimated irreducible
  diagnosis, targeted matched-budget augmentation, ranking and threshold parity
  [B pp5,8–9]. Claiming this basic distinction as new is contradicted.
- Balanced-risk generator-mismatch bias, conditional augmentation regimes and
  validation-tuned synthetic size [T pp9–12,19,22]. Count tuning is not novelty.
- Validation-guided weighting [R §3.1], group marginal data value/acquisition
  [D §§3.2,4.1,4.5], local influence [I §2.1], and ordinary cluster/graph SSL
  machinery [S3,S6,S7]. No theorem here solves all A/B/C for arbitrary distributions.

## B. What remains unverified?

1. The contents/bibliography of the reported 13-paper collection: attachment
   exceeds 32 MiB. CFS-SMOTE, BFCH-SMOTE, EC-EBO and other listed families remain
   FULL TEXT UNAVAILABLE, not inferred from names.
2. Unique UGO fidelity: cap conflict (§3.3 versus Table 4), exact pools/counts,
   ties/rounding/overshoot and author implementation. Demonstrated batch mismatch
   is repaired, but full experiment reproduction is pending [UGO_PAPER_CODE_AUDIT.md](UGO_PAPER_CODE_AUDIT.md).
3. L's entropy sign/finite stopping, H's printed metrics and U-mask protocol,
   G's distance inconsistency and mismatch between literal descent and verbal
   safety. Source equations are verified; intended author code is not.
4. Full supplementary proof verification, exhaustive direct-competitor/robust-graph
   search, and any finite-data extension of T's asymptotic theory.
5. Official folds, authenticated dataset license/provider semantics and record
   IDs. All eight full files parse, but no official training-fold smoke ran.
6. Practical power and generalization of either proposed diagnostic mechanism.
   No predictive gains, subgroup benefit, U benefit, formal synthetic safety or
   methodological novelty have been measured/established.

## C. Which candidates should be rejected?

Reject UGO plus ordinary clustering, pseudo-labels, entropy weighting, distance
filters, global count tuning or multiplied heuristic scores as present novelty
arguments [L,H,B,T,S3,S5,S6]. Reject high uncertainty/sparse density as an intrinsic
data-limitation estimator; reject non-significant benefit as proof of irreducibility
[gap counterexamples; B p5 explicitly qualifies real-data labels].

Candidate 1 (finite control-adjusted regional contrasts) is rejected **as an
already novel algorithm**: it closely resembles B's intervention question, group
valuation and ordinary nested model selection. Candidate 2 (anchor-compatible
graph sensitivity intervals) is rejected as a standalone novel oversampler:
ordinary harmonic structure and an unjustified smoothness radius do not certify
membership [S3,S7; candidates' overlap counterexample].

## D. Which candidate, if any, is defensible?

Neither is defensible as a novel oversampling contribution. Candidate 1 can be
retained as a **prospective empirical diagnostic**: do fine-region finite response
estimates predict fresh discrimination gains beyond matched controls at realistic
label budgets? Its conservative bound is often uninformative (candidate power
calculation); it must allow inconclusive outcomes. Candidate 2 is at most a
separate structural-assumption/U ablation after source review. Adding it does not
rescue Candidate 1's novelty. No implementation is approved by this decision.

## E. Single strongest reviewer objection

> Hamilton already decomposes threshold recovery, data-reducible scarcity and
> estimated irreducible overlap, and tests matched category-specific augmentation
> with ranking and threshold controls. Your fine-region validation search may be
> that diagnostic plus ordinary data/model selection, with more labelled audit
> data and computation, rather than a new mathematical mechanism.

Direct basis: [B pp5,8–9], with [T p22 Algorithm 1] and [D §3.2] reinforcing the
size-selection/group-value precedents. Finer resolution, intervals, FPR constraints
or a new acronym alone do not answer this objection.

## F. What would falsify the proposed contribution?

There is no established contribution; preregister a diagnostic **usefulness**
hypothesis, practical margins and power before testing. Evidence against it:

- Adequately powered fresh-region confirmation finds negligible predictive value
  for utility estimates beyond counts/uncertainty or category/group-value controls.
- Equivalence intervals place selected-procedure risk/ranking gain within a
  predeclared negligible margin versus tuned thresholds, class/point weighting
  and count search. A small study's non-rejection alone is inconclusive.
- Confirmed majority FPR harm exceeds tolerance, or rare-region gain sacrifices
  another minority subgroup.
- Matching all visible-label, U and fit budgets removes the advantage.
- Claimed U increment fails no-U/ordinary SSL/shift controls; this falsifies the
  U component independently, rather than all possible A/B improvements.
- A direct competitor specifies the same estimand/intervention/selection rule:
  novelty is defeated even if empirical utility exists.

These are planned tests, not observed results. Evaluate the frozen complete
procedure on untouched O, including all selection/refitting [S4; B §5.5].

## G. Evidence required before implementation

1. Supply the collection as individual PDFs/smaller archives, identify exact
   competitors and compare to B/T/D/R/I in a precise difference table.
2. Obtain official folds/provenance and resolve author UGO conventions; finish
   blocked real-fold software smoke checks. Keep every fidelity change versioned.
3. Resolve L/H/G reproduction issues before calling an approximation faithful.
   Do not inherit T's guarantees without its exact learner/loss assumptions.
4. Specify one genuine mechanism or abandon the novelty claim. A standard
   validation loop with a new metric is insufficient.
5. Declare F/H/A/O/U, masks, total visible-label budget, class/group/time rules,
   generator/learner/regularization, cap interpretation, tuning and compute parity.
6. Demonstrate feasible anchor/audit counts and statistical power; justify AP /
   all-pairs AUC uncertainty separately. Adaptive reuse of A needs fresh independent
   confirmation or an explicit justified protocol, not a fixed-family bound.
7. Preregister controls, known-overlap and unequal-subgroup simulations, U ablations,
   practical margins, infeasible/inconclusive outcomes and untouched outer tests
   [DATASET_PLAN.md; PROBLEM_STATEMENT_COVERAGE.md].
8. Obtain user review before another stage. Recommended next work is source/fold
   completion and Stage 4B mathematical/design review, not DRA-UGO implementation.

## Exact publication-access report and stopping point

Newly inspected supplied **full texts**: UGO, SLEO, SSHR, GDEO, *Don't oversample
the boundary*, and *Synthetic Augmentation in Imbalanced Learning* v3.
Previously verified additional full texts: Ren et al. 2018, Ghorbani & Zou 2019,
Koh & Liang 2017. **Abstract-only verified publications: none.** The inaccessible
13-paper collection is not counted as read; no exhaustive proof audit is claimed.

Stage 4A-R creates offline infrastructure, verified manifests, explicit UGO
repairs/tests and source-grounded documentation. It does not run benchmarks or
implement the proposed method. Review this stage and stop for user approval.
