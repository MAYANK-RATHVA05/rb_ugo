# Saved-prediction diagnostics: standard versus original-support HS

This is an exploratory follow-up to the completed
[bounded pilot](RESULTS.md). **The original `advancement_trigger_not_met`
decision is preserved.** No classifiers or preprocessing were refitted, no
new synthetic observations were generated, and no method/grid/dataset was
changed. All findings below use already saved selected-model test scores.

## Artifact availability and verification

All 30 case JSON files and 30 prediction NPZ files are available and match
the previously published checksums. Both standard HS (B) and original-support
HS (C) have aligned test labels/probabilities in every case: 60 vectors total.
Synthetic cases additionally save known test subgroup identities. There are
no missing artifacts needed for fixed-threshold or test-curve diagnostics.

**Validation probability vectors were not saved.** Consequently this analysis
cannot select or assess validation-calibrated decision thresholds. Test-curve
sweeps below are retrospective diagnostics only. They must not be reported as
deployment thresholds or replacement headline performance.

The original decision's SHA-256 remains
`5593e7e8d031211514ab0d5451f235608e143e0023b4971e65fce947062a5199`.
Both raw and published decision files agree. The new computations recompute
the original saved AP/F1 values before adding diagnostics.

## Paired metrics at the original .5 threshold

Each entry below is a five-seed mean. P=minority precision, R=minority recall,
F1=minority F1, AP=average precision, FPR=FP/(FP+TN). Values are fractions,
not percentages. No metric is averaged together with another metric.

| Setting | Method | P | R | F1 | AP | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ecoli4 | B: standard | 1.0000 | .4833 | .6333 | .5902 | .0000 |
| Ecoli4 | C: original | 1.0000 | .4833 | .6333 | .5902 | .0000 |
| Glass6 | B | .7333 | .7317 | .7273 | .6620 | .0373 |
| Glass6 | C | .7067 | .7317 | .7095 | .6620 | .0424 |
| Yeast4 | B | .2720 | .5517 | .3470 | .3443 | .0913 |
| Yeast4 | C | .2447 | .6925 | .3400 | .3543 | .1293 |
| Synthetic c=0 | B | .3451 | .7452 | .4716 | .4580 | .1417 |
| Synthetic c=0 | C | .2946 | .8040 | .4292 | .4605 | .1977 |
| Synthetic c=.5 | B | .3709 | .6720 | .4775 | .3682 | .1137 |
| Synthetic c=.5 | C | .3337 | .6620 | .4430 | .3855 | .1317 |
| Synthetic c=1 | B | .4223 | .4528 | .4354 | .3397 | .0634 |
| Synthetic c=1 | C | .3598 | .4440 | .3965 | .3210 | .0797 |

The complete **paired per-seed** report is
[per_seed.csv](exploratory_v1/per_seed.csv): 30 rows containing both methods'
precision, recall, F1, AP, FPR, confusion counts, differences, subgroup recall,
selected lambdas and prediction transitions. These are paired on the same
test observations; they are not pooled confusion counts across seeds.
[summary.csv](exploratory_v1/summary.csv) contains paired means, SDs,
descriptive t(4) intervals and win/tie/loss counts for every metric.
Intervals describe variation across overlapping holdouts/related synthetic
settings, not independent-dataset inference or significance.

## Is rare recall accompanied by more false positives?

Yes, especially in the c=0 setting that drives the original aggregate rare
recall gain. Every synthetic test set has 5000 negatives and 100 rare positives.

| Concentration | B rare recall | C rare recall | Mean B FP | Mean C FP | Mean extra FP | Precision change | F1 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .460 | .694 | 708.6 | 988.6 | +280.0 | -.0505 | -.0425 |
| .5 | .560 | .568 | 568.6 | 658.6 | +90.0 | -.0372 | -.0345 |
| 1 | .506 | .506 | 316.8 | 398.4 | +81.6 | -.0624 | -.0388 |

At c=0, rare recall improves in four seeds and ties in one; false positives
increase in **all five**. Across the five paired test evaluations, C adds 147
true positives and 1400 false positives relative to B, with none removed.
These totals illustrate transitions on the existing test sets; they are not
independent draws or a population ratio. The mean rare-recall difference
is +.234, paired interval [-.0507,+.5187]; mean FPR difference is +.056,
paired interval [.0118,.1002]. These descriptive intervals do not replace
the previously failed advancement test.

For transparency, here are all five c=0 pairs:

| Seed | B rare recall | C rare recall | B FP | C FP |
| --- | ---: | ---: | ---: | ---: |
| 42 | .48 | .77 | 701 | 1122 |
| 43 | .59 | .59 | 643 | 671 |
| 44 | .55 | .71 | 746 | 1019 |
| 45 | .63 | .75 | 704 | 908 |
| 46 | .05 | .65 | 749 | 1223 |

At c=.5, four seeds improve rare recall, but only three improve total minority
recall; false positives increase in four seeds. Mean total recall actually
falls by .010. At c=1, mean rare recall ties while total recall falls .0088
and FPR increases in all five seeds. Thus extra positive predictions are not
consistently directed at genuine minority observations.

The real-data Yeast4 gain also comes with more false alarms: recall rises
.5517 to .6925, FPR .0913 to .1293, precision falls .2720 to .2447, and F1
falls .3470 to .3400. No genuine subgroup identities exist for these real
inputs, so this is total minority recall, not rare-group recall.

## Selected regularization and concentration

These are validation-selected settings from the original run; nothing was
retuned. Use relative lambda for comparisons: B's raw lambda multiplies
the augmented root count and C's multiplies the original root count. A
smaller raw C lambda alone does not imply weaker effective local shrinkage.
Local parent counts also differ. The fixed structure and raw predictions
remain shared between B/C.

| Setting | B median relative lambda (range) | C median relative lambda (range) |
| --- | --- | --- |
| Ecoli4 | 0 (0–1) | 0 (0–.3) |
| Glass6 | 1 (0–30) | .001 (0–.3) |
| Yeast4 | 1 (.003–30) | 1 (.001–10) |
| Synthetic c=0 | 1 (.3–100) | 30 (.3–30) |
| Synthetic c=.5 | .1 (.03–.1) | .03 (.003–.1) |
| Synthetic c=1 | .03 (.03–.1) | .001 (.001–.03) |

At c=0, standard HS chooses its upper grid endpoint 100 in two seeds; C
chooses 30 in three seeds. This is a limitation of the existing finite search;
the grid was not extended. At high concentration, C selects far smaller
relative lambdas. The interaction of original support with selected lambda
is substantial, rather than a stable improvement across concentration levels.
Individual settings, raw lambdas, validation AP/F1 and test metrics are in
[selected_parameters.csv](exploratory_v1/selected_parameters.csv).

AP changes by +.00243 / +.01724 / -.01871 across c=0/.5/1. Medium concentration
has a modest ranking benefit (four AP wins out of five), but its paired AP
interval includes zero and F1 decreases in all five seeds. Strong concentration
reverses the AP benefit (four losses out of five). This does not establish
consistent robustness to local synthetic concentration.

## Exploratory recall under shared precision requirements

For each saved test score vector, enumerate all deterministic thresholds at
distinct score values, retaining tied score groups together. For each common
minimum precision floor, report the greatest attainable recall. If no nonempty
prediction set reaches that floor, record infeasibility; do not manufacture
precision=1 for an empty prediction set or replace missing recall with zero.
No interpolation, tie splitting or extrapolation is used.

Fixed diagnostic floors are .25/.50/.75/.90. Additionally, in each seed use
the previously observed B precision at .5 as a common floor for both curves;
repeat symmetrically with C's precision. Both methods get their own retrospective
curve envelope, so even B's envelope recall can exceed B's frozen-threshold
recall. These comparisons condition on test labels and are **exploratory**.
No threshold is proposed for future data or used to revise headline scores.

Exact precision matching is generally impossible with tied/discrete tree
scores. These are matched **minimum-precision constraints**, not promises of
identical achieved precision. All actual precisions, FPRs, rare recalls,
thresholds and infeasible cases are saved in
[matched_precision_per_seed.csv](exploratory_v1/matched_precision_per_seed.csv).

At each seed's B precision floor:

| Setting | Paired feasible seeds | Achieved B/C precision | B/C minority recall | B/C rare recall |
| --- | ---: | --- | --- | --- |
| Ecoli4 | 5/5 | 1.0000 / 1.0000 | .5333 / .5333 | unavailable |
| Glass6 | 5/5 | .7967 / .7967 | .7317 / .7317 | unavailable |
| Yeast4 | 5/5 | .2895 / .3213 | .6907 / .6740 | unavailable |
| Synthetic c=0 | 5/5 | .3481 / .3467 | .7628 / .7692 | .548 / .542 |
| Synthetic c=.5 | 5/5 | .3718 / .3739 | .6748 / .5884 | .562 / .508 |
| Synthetic c=1 | 3/5 | .3957 / .4015 | .4853 / .3447 | .510 / .400 |

The c=0 rare-recall advantage **disappears** under this common precision
constraint: difference -.006 rather than +.234. Its total-recall difference
is only +.0064, paired interval [-.0106,+.0234]. At c=.5 total recall is
.0864 lower for C, and at c=1 C cannot even attain B's floor in two seeds.
The three feasible high-concentration pairs must not be generalized to all
five. Slightly different achieved precisions also limit interpretation.

The symmetric C-floor check gives c=0 rare recall .690/.694, only +.004
for C; at c=.5 rare recall is .598/.570, and at c=1 .628/.552. Thus the
large fixed-threshold c=0 advantage is not robust to controlling precision.

The fixed floors show some localized benefits and limitations:

- At precision >=.25, c=0 total recall is .8412/.8432 and rare recall
  .740/.756: a small residual rare gain (+.016), rather than +.234. At
  c=.5 rare recall improves .676/.702 while total recall slightly falls.
- At precision >=.50, c=0 total recall is .4244/.4328, a small +.0084
  gain; both rare recalls are zero. All five pairs are feasible.
- At c=.5 and precision >=.50, B is feasible in 2/5 seeds and C in 4/5;
  only two seeds are jointly feasible. On those two, total recall is
  .264/.272 and rare recall .325/.325. This is localized positive ranking
  evidence, with too little common coverage for a general conclusion.
- At c=1 and precision >=.50, each model is feasible in one seed, but
  **different seeds**, so there are zero valid paired comparisons.
- Higher floors are frequently infeasible. No synthetic case reaches .90;
  floor .75 has only one paired c=0 seed and no paired c=.5/c=1 seeds.

[matched_precision_coverage.csv](exploratory_v1/matched_precision_coverage.csv)
includes every setting/floor, even zero-pair combinations. Summaries in
[matched_precision_summary.csv](exploratory_v1/matched_precision_summary.csv)
always state paired sample size, individual feasibility and achieved precision.
Unattainable cases are not silently scored as successes.

## Interpretation

The method changes local shrinkage and sometimes ranking, so it is not
literally a uniform probability shift. B/C test-score Spearman correlation
is 1.0 on Ecoli4/Glass6 and approximately .9952 on Yeast4, .9972/.9916/.9910
across synthetic concentrations. Identical rankings can still produce different
decisions at .5; AP is then unchanged while precision/F1 change.

At the original cutoff, the clearest rare-recall benefit predominantly comes
from classifying more observations as minority and accepting more false alarms.
Predicted-positive fraction rises .1966 to .2528 at c=0, .1645 to .1799 at
c=.5, and .0988 to .1128 at c=1. Precision decreases in all 15 synthetic
pairs. At high concentration, even total recall declines despite more positive
predictions. This is not a consistent increase in useful minority recognition.

Do not infer that every probability is shifted upward: mean negative-class
probability changes +.0382 / -.0179 / -.0135 across c=0/.5/1, while FPR
increases in every concentration mean. Redistribution around the .5 cutoff,
local rank changes and the originally selected lambdas all matter. This
analysis cannot causally separate the count rule from its interaction with
validation-selected lambda or assert improved probability calibration.

There are small, distinct ranking/curve benefits in some settings, notably
medium-concentration AP and a few precision floors. They are insufficient
to establish a reliable advantage over tuned standard HS. The dominant
large rare-recall gain is an operating-point tradeoff, and does not survive
the shared-precision checks as a comparably large benefit. **The failed
advancement decision remains appropriate and unchanged.**

## Reproduction

The analysis script imports metrics/statistical utilities only; it never calls
model fitting, preprocessing fitting, SMOTE or the pilot evaluator. It reads
the frozen raw artifacts and writes only the new exploratory report tables.
No graph or expanded benchmark was generated.

Five focused correctness tests passed for confusion metrics/AP, tied-score
operating points, infeasible precision floors, recall-envelope selection,
undefined precision and missing artifacts. Repeating the complete analysis
with classifier fitting, scaler fitting and SMOTE APIs deliberately blocked
reproduced every CSV byte for byte. All original raw/published artifacts,
including the failed advancement decision, retained their original hashes.

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_original_support_hs_predictions --output results\hs-prediction-diagnostics
```

The original raw case/prediction directory is required. Pulling the branch is
sufficient to read the committed analysis tables but does not download raw
NPZ files. Use a fresh output directory to preserve evidence. See
[verification.json](exploratory_v1/verification.json) for artifact coverage,
precision-floor policy, zero-refit declaration and preserved decision hash.
