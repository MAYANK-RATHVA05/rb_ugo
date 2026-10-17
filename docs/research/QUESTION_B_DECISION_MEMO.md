# Question B decision: close filter-based method development

**Decision: close this Question B method-development direction within the tested
settings. Select zero further experiments.** Existing tuning and methods recover
the clean-regime deficit; no practically meaningful, recoverable residual beyond
the existing controls has been demonstrated. Overlap leaves an unresolved
subgroup disparity, not an established opportunity for a new sampler. This does
not establish equivalence between methods or solve rare-mode learning generally.

This supersedes the optional count-locked filter follow-up in Section 6 of the
[pilot report](QUESTION_B_DIAGNOSTIC_RESULTS.md). Preserve that report,
[protocol](QUESTION_B_DIAGNOSTIC_PROTOCOL.md), code and results. Filter relaxation
is closed as a sampler candidate here; allocation and stopping have no independent
development case.

## Evidence reviewed and supported conclusions

Reviewed all 108 [aggregate rows](question_b_diagnostic/aggregate_metrics.csv),
1,260 [paired contrasts](question_b_diagnostic/paired_contrasts.csv),
[mechanism traces](question_b_diagnostic/mechanism_summary.csv),
[validation choices](question_b_diagnostic/validation_selections.csv), and
per-seed records for both regimes/learners. The
[audit](question_b_diagnostic/validation.json) records 840 fitting-only jobs,
1,080 pilot records including aliases, and 40 separate independent model replays.
There were no missing or failed comparisons.
True subgroup identities were evaluator-only throughout fitting, generation and
selection.

RF100 means over the ten paired seeds follow. Each cell gives **worst-group
recall (%) / achieved test FPR (%) / AP**. These are separate operating points
selected by the same validation-negative FPR rule, not exactly equal test FPRs.

| Existing arm or labelled diagnostic | Clean | Overlap |
| --- | --- | --- |
| Original UGO / cap-480 k=5 | 86.20 / 4.99 / .98929 | 2.00 / 4.90 / .33096 |
| Validation-tuned UGO k | 96.40 / 4.91 / .99310 | 2.80 / 5.05 / .33180 |
| Generation-eligibility-disabled diagnostic | 95.99 / 5.13 / .99363 | 7.20 / 4.96 / .33632 |
| Full-pre-filter-disabled diagnostic | 94.80 / 5.16 / .99334 | 6.00 / 4.89 / .34444 |
| NORES | 92.80 / 4.90 / .99388 | 5.00 / 4.94 / .33303 |
| Validation-tuned NORES capacity | 92.40 / 4.94 / .99369 | 0.60 / 5.02 / .34180 |
| Class weighting | 94.20 / 4.85 / .99407 | 5.60 / 5.01 / .33981 |
| ROS, 480 additions | 97.80 / 4.93 / .99541 | 9.60 / 5.00 / .33913 |
| ROS matched to generation diagnostic | 97.56 / 4.80 / .99543 | 8.90 / 4.94 / .34108 |
| Matched multiplicity weights | 95.20 / 4.98 / .99405 | 4.60 / 4.99 / .33723 |
| SMOTE | 91.60 / 4.86 / .99070 | 6.00 / 4.83 / .34434 |
| KMeansSMOTE | 85.40 / 4.89 / .98828 | 2.46 / 4.99 / .33692 |
| Audited EC-EBO interpretation | 99.60 / 5.07 / .99749 | 2.60 / 4.91 / .35073 |

Worst recall and rare recall usually coincide, but are different statistics:
overlap KMeansSMOTE rare recall is 6.40%, versus 2.46% worst recall. Clean
generation-disabled and matched-ROS rare recall are 96.00% and 97.60%.

The primary clean diagnostic-minus-UGO worst-recall effect was **+9.79 pp**,
paired 95% interval **[+2.19, +22.80]**; tuned UGO minus reference was
**+10.20 pp [+1.80, +23.00]**. Diagnostic minus count-matched ROS was
**-1.57 pp [-4.96, +1.36]**, and minus matched weights
**+0.79 pp [-3.23, +5.20]**. These intervals do not prove equivalence. They
provide no demonstrated advantage requiring the filter repair. The primary
lower bound missed the frozen **five-point minimum**; some diagnostic test FPRs
exceeded 6% (maximum 6.26%). Its AP gain over reference was .00435, but mean
population error increased from 4.565% to 4.641%. Better rare recall is not
automatically better population utility. Secondary intervals are unadjusted;
all ten-seed simulation intervals remain exploratory.

The mechanism was active in all clean reference runs: all three rare originals
were seed-ineligible but **retained for classifier training**; each seed had
generation rounds where excluded rare points entered the counterfactual ranked
pool. Overlap RF100 entered minority generation in every seed, but only five had
that rare-ranking opportunity. Natural-budget and cap-480 UGO produced identical
fitting arrays in all 40 cells.

Clean diagnostic additions averaged 144 versus reference UGO's 228, with 5.9
versus zero inside rare support. Realized-count ROS and multiplicity weights
control quantity; RF bootstrapping prevents their equivalence. The intervention
bundles anchor/partner access and subsequent trajectory changes. Six tuned clean
runs retained zero rare eligibility/additions yet averaged 95% rare recall.
LogReg had zero rare recall throughout, consistent with capacity misspecification.
Unchanged reference RF scores give 7.2% worst recall at threshold .5 versus 86.2%
after validation selection, at test FPR 0% versus 4.99%. Counts and default-cutoff
recall alone therefore cannot identify a sampler failure.

## What selection and overlap do not establish

UGO k and NORES capacity were **validation-selected** by global balanced accuracy
without group identities. Clean RF chose k=1/3/7 in 4/3/3 seeds; overlap chose
k=1/3/5/7 in 1/4/3/2. NORES chose larger leaves in nine overlap seeds. All
thresholds were validation-selected. **No cross-method selector was evaluated**:
EC-EBO as clean winner or ROS as overlap winner is retrospective test-summary
selection, not a validated deployment policy. Matched ROS also requires a fitted
UGO quota. Competitor-specific tuning and compute were not equal.

Overlap is the principal unresolved observation: even ROS's rare recall was
only 4–18% across seeds, while its common-mode means were 27.44% and 22.12%.
This persists across seeds, rather than being one losing replicate. But the
known law gives only **FPR >= .20 × population-weighted minority recall**:
at population FPR=.05, weighted recall cannot exceed .25. It neither explains
away the group disparity nor proves observed worst recall optimal. ROS minus
reference UGO was +7.60 pp [+5.00, +10.40]; tuned UGO minus its matched ROS
was -5.20 pp [-8.80, -2.20]. Existing duplication already improves this comparison;
neither it nor EC-EBO establishes an overlap solution.

Plausible explanations include only three rare fitting positives, RF score
resolution/regularization, and global versus worst-group selection, alongside
genuine overlap. Capacity tuning improving AP while reducing rare recall
illustrates objective mismatch without establishing its cause. Calibration,
quantity and comparator settings remain alternatives. There is no independent
capacity/learning-curve attribution or prospective method-family selection
study. These exploratory hypotheses require fresh confirmation.

## Literature check and next step

Rechecked primary passages: [Hamilton, Section 5.5/Table 5,
pp.9–10](https://doi.org/10.1016/j.neucom.2026.134831) supports threshold/ranking
and learner controls, with an explicit neural-learner exception, not a universal
no-oversampling theorem. [Ma–Zhang, pp.10–11 and VTSS
p.22](https://arxiv.org/abs/2601.16120v3) separates proportion/mismatch effects
and already tunes quantity; its assumptions do not certify RF. [SLEO,
Eqs.4,7](https://doi.org/10.1016/j.knosys.2025.115009) already allocates by cluster
entropy. The [EC-EBO audit](../EC_EBO_METHOD_DECISIONS.md) retains explicit
paper/implementation discrepancies. These global benchmarks do not establish
this subgroup estimand. KMeansSMOTE author/library evidence and incomplete
primary access for other competitors retain the
[source-ledger qualifications](QUESTION_B_RESEARCH_PROPOSAL.md#4-close-prior-work-mechanisms-and-verification-limits).
No new remedy or novelty claim follows. The old provisional proposal and
inactive designs were excluded.

**Recommend closure.** The scientific question was whether rare-seed exclusion
establishes a need for filter relaxation beyond existing options: it does not in
these settings. Overlap recoverability and real-subtype transfer remain
unanswered, but neither establishes a new-sampler target. Reopening requires
independent, practically consequential held-out harm surviving those controls,
with utility/FPR requirements declared beforehand. Reinspection cannot supply
that confirmation.

Selected next experiments: **0**; incremental experimental cost: **0 fits / 0 CPU
hours**. No new metric, seeds, protocol, sampler or experiment was introduced in
this decision stage. All code, frozen specifications and numerical artifacts
remain preserved.
