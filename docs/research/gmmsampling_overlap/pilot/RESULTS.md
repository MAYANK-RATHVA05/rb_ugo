# GMMSampling competing-minority density gate: bounded pilot results

**Decision: advancement failed. Stop this exact candidate at this pilot.**
Quota protection worked as software, but did not produce the predefined predictive
benefit. Broader evaluation is not justified by these results. This does not prove
that every competing-density method fails; novelty remains unverified.

## What was run

The [final protocol](FROZEN_PROTOCOL.md) and configuration were committed in
`4d95e20ed14aa6909ae2293fa29260428d6ee7d9` **before** performance was observed.
The previous audit/proposed protocol remain unchanged. The user's authorization
replaced the earlier prerequisite to resume literature verification: this tests
usefulness without a new novelty claim or another broad search.

Thirty synthetic cases: deltas 4/1.5/0, minority subgroup proportions 50/50 or
90/10, seeds 42–46. Independent original fitting/validation/test samples contain
1000/100/100, 1000/100/100 and 5000/1000/1000 observations respectively. Two genuine
subgroups occur in each minority class. Their identities are used only by the
evaluator. Gaussian settings and paired random streams are in the protocol.

A is tuned code-conventions GMMSampling; B rejects without refill; C uses bounded
component refill and original-member fallback; D duplicates original hard-MAP
component members with matching quotas. NORES and class-weighted multinomial
logistic regression are simple controls. Logistic C is .1/1/10; A tunes safe k,
B/C tune tau 0/.55/.70. All selection uses independent validation macro AP, ties
by macro F1 then grid order. Test predictions are made after selection; no
validation refit or test threshold tuning. A/B/C have nine candidates each;
D/NORES/WEIGHTED have three each: **1080 classifier fits** and 180 selected results.
There were **254 GMM EM fits (2540 initializations)**, all on original fitting rows.

As instructed, all original observations are retained, including majority rows.
This disables author undersampling in every arm. The addition budget remains
450 per minority class. Largest-remainder quotas, exact Gaussian draw counts,
ID-based neighbor self-exclusion and local random streams are shared engineering
adaptations. A is therefore a documented code-conventions baseline, **not exact
execution of the published full resampling pipeline**. The paper/code differences
remain in [the audit](../AUDIT.md). Pinned author helpers and GMM-selection/final
parameters match exactly on correctness fixtures; identical random sample bytes
with the complete legacy author API are not claimed.

## Main results: partial minority overlap (delta=1.5)

Five-seed arithmetic means; AP/F1 are macro averages over all three classes.
Pair confusion counts true1→pred2 and true2→pred1, divided by all minority test
rows. Worst recall is the per-seed minimum of the four genuine subgroup recalls,
then averaged across seeds. It is not the minimum of the averaged subgroup recalls.

| Subgroup mix | Method | Macro AP ↑ | Macro F1 ↑ | Pair confusion ↓ | Worst subgroup recall ↑ |
|---|---|---:|---:|---:|---:|
| 50/50 | A | .97023 | .93578 | .0962 | .78640 |
| 50/50 | B | .97022 | .93641 | .0953 | .80440 |
| 50/50 | C | .97023 | .93578 | .0962 | .78640 |
| 50/50 | D | .97019 | .93583 | .0962 | .78680 |
| 50/50 | NORES | .97358 | .93645 | .0944 | .79920 |
| 50/50 | WEIGHTED | .96980 | .93664 | .0949 | .79840 |
| 90/10 | A | .94455 | .89051 | .1637 | .77933 |
| 90/10 | B | .94430 | .89197 | .1615 | .78089 |
| 90/10 | C | .94441 | .89164 | .1620 | .78022 |
| 90/10 | D | .94455 | .89039 | .1639 | .77778 |
| 90/10 | NORES | .94526 | .89531 | .1570 | .81511 |
| 90/10 | WEIGHTED | .94519 | .89525 | .1571 | .81578 |

C matches A exactly in every 50/50 partial-overlap seed. In 90/10, C changes only
seeds 43/44; confusion reductions are .0055/.0030, but AP changes are
-.000120/-.000583. The remaining three seeds tie A. Thus confusion improves in
0/5 and 2/5 seeds, below the required 4/5 in each condition.

Across the ten partial-overlap cases, C's paired mean AP changes are
**-.000070 vs A**, +.000060 vs B, and **-.000054 vs D**. It does not beat A or D on
the primary measure. Compared with B, its tiny AP increase comes with slightly
worse F1 and pair confusion in both subgroup conditions. The differently realized
sample sizes also prevent interpreting C–B alone as a quota-specific benefit.

In 90/10, C–A mean pair-confusion change is -.00170 (descriptive 95% paired t
interval [-.00479,+.00139]); AP change -.000141 [-.000455,+.000173]; worst-group
recall change +.000889 [-.000910,+.002688]. C–B worst-group recall change is
-.000667, far below the required +.03. Full paired values, SDs and descriptive
intervals for A/B/D and simple controls are saved below. Five correlated seeds
and related conditions do not support statistical certainty.

## Coverage, quotas and duplication

The partial-overlap subgroup safeguard passes: 50/50 is unchanged; 90/10 C–A
mean recalls for groups (class1 common/rare, class2 common/rare) change by
+.004667, +.010000, -.002000 and approximately zero. There is a small class2
common-group tradeoff, below the .02 tolerance; no minority-to-majority error
increase masks the confusion reduction. This small gain does not meet the primary
AP or confusion thresholds. NORES/WEIGHTED meet the predefined simple-control
coverage test in both partial-overlap cells.

Mean additions/proposals per selected method, summed over both minorities:

| Condition | Method | Requested | Gaussian proposals | Accepted | Rejected | Fallback | Realized additions |
|---|---|---:|---:|---:|---:|---:|---:|
| Partial 50/50 | A | 900 | 900 | 900 | 0 | 0 | 900 |
| Partial 50/50 | B | 900 | 900 | 873.2 | 26.8 | 0 | 873.2 |
| Partial 50/50 | C | 900 | 900 | 900 | 0 | 0 | 900 |
| Partial 50/50 | D | 900 | 0 | 0 | 0 | 900 | 900 |
| Partial 90/10 | A | 900 | 900 | 900 | 0 | 0 | 900 |
| Partial 90/10 | B | 900 | 900 | 813.6 | 86.4 | 0 | 813.6 |
| Partial 90/10 | C | 900 | 1050.8 | 900 | 150.8 | 0 | 900 |
| Partial 90/10 | D | 900 | 0 | 0 | 0 | 900 | 900 |

All methods retain 1200 originals, so total fitting rows add those originals to
these realized additions. Requested/generated/accepted/rejected/fallback counts
for **every class and fitted component**, including unselected settings, are in
[component_counts.csv](evidence/component_counts.csv).

Validation selected C's bypass tau0 in all ten weak-overlap cases, all five
50/50 partial cases and three of five 90/10 partial cases; the other two selected
tau.70. **No fallback duplicates occur in selected C's partial-overlap cases**;
fallback therefore cannot explain the lack of benefit there. D has comparable
AP despite being entirely duplicates. This weakens a claim of a useful added
Gaussian generation mechanism on this family.

At delta0, selected C has mean fallback fractions 11.58% (50/50) and 4.91%
(90/10). Its pair confusion remains near .5 (.5019/.4949) and worst-group recall
can be very low (.0552/.0060). Identical minority distributions have no intrinsic
class separation; fitted rejection scores can manufacture apparent confidence.
These diagnostics cannot certify generated labels. Weak-overlap AP/F1 safeguards
pass, largely because validation turns the gate off. Analytical true-density
rival-dominance/ambiguity/soft rare-coverage diagnostics are evaluator-only and
saved for accepted Gaussian additions and duplicates separately.

## Frozen decision, validation and limitations

[decision.json](evidence/decision.json) records every gate. Accounting, subgroup/
other safeguards, weak-overlap controls and fresh-generation fractions passed.
Both required confusion reductions, primary AP benefit vs A and D, coverage gain
vs B, and distinction from the simple controls failed. **Do not advance or alter
these criteria after viewing results.** No larger or real-data experiment was run.

The full repository suite passed **945 tests**, including nine focused new tests.
The read-only saved-evidence audit checked 30 cases, 1080 validation files,
180 selected test files and **358998 consumed proposals**, including quota bounds,
class labels, original retention, fallback parent/component membership, numerical
gates, validation selection and reconstructed selected test probabilities. No
classifier was refitted during this audit. Tau0 matches the k7 baseline exactly.
All 30 cases and 1080 candidates completed, with **zero recorded failures**.
The measured run took **44.50 seconds** on this environment; per-case, density,
sampling and classifier timings are saved. Peak-memory usage was not instrumented.

This is one two-dimensional Gaussian family with a linear classifier, a largely
separated majority and a small tuning budget. Macro AP includes an easy majority
class; minority-only AP, class recalls and complete confusion matrices are also
saved. No broad literature search was repeated and the source-access/novelty
uncertainties from the audit remain. Failure to advance here is a bounded
usefulness finding, not a universal impossibility result.

## Saved evidence

Tracked [evidence/](evidence/README.md) contains per-seed results, paired effects,
intervals, selected parameters, all candidate validation scores, class/component
counts, original assignments, evaluator density diagnostics, runtime, source
versions, frozen decision and hash manifests. Raw probabilities/partitions/proposals
are preserved under ignored `results/gmmsampling-overlap-pilot/`, following the
repository's local-artifact policy. Six portable bundles (about 14–15 MiB each)
are saved at `/workspace/gmmsampling-overlap-evidence/`; their hashes are tracked.
The tracked report/CSVs do not by themselves contain full probability vectors.

Use `python -m scripts.audit_gmmsampling_overlap_pilot --output
results/gmmsampling-overlap-pilot` to verify the raw evidence without training.
See the pilot README for fresh-run instructions; the runner refuses overwrites.
