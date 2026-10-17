# Question B diagnostic pilot: existing tuning and methods recover the observed deficit

**Conclusion: Existing tuning or methods resolve the failure, within this
exploratory artificial-data pilot.** Default UGO exhibited a recoverable average
rare-mode deficit in the clean RF100 setting. The evidence does not support a
new sampler: validation-only UGO tuning, ordinary duplication and an existing
subgroup-aware method matched or exceeded the diagnostic recovery. This is not
a universal resolution claim or an equivalence test across real problems.

The intervention demonstrates sensitivity to the **generation eligibility
policy**, including downstream trajectory/quantity changes. It does not establish
that restoring rare anchors specifically is necessary or uniquely responsible.
The prespecified strong promotion criteria were not met, so no new-method
development is recommended from this pilot.

## Frozen protocol, scope and validation

- [Protocol and completed source-evidence ledger](QUESTION_B_DIAGNOSTIC_PROTOCOL.md),
  [frozen JSON](../../data/manifests/question_b_diagnostic_protocol.json),
  implementation freeze commit `d986740` made before inspecting pilot outcomes.
- Fitting/validation/test: 600/2,000/10,000 independent feature draws with fixed
  strata. Target positive prior .10, minority mode weights 70/25/5. Ten paired
  seeds 42–51, clean and known-overlap regimes, LogReg and RF100.
- 840 fitting-only records completed first; 960 final classifiers were then
  fitted. Including 120 validation-selected aliases, all **1,080 pilot result
  records succeeded**. There were no failed, unavailable or silently skipped
  jobs. UGO performed 1,624 internal scoring fits, separately from final fits.
- All original UGO training rows/labels were retained exactly. Every resampling
  arm passed the original-row prefix check. Preprocessing and resampling used
  F only; tuning/thresholds used V only; true mode identifiers were evaluator
  metadata. All learned models were fresh final fits after resampling.
- **756 tests passed**, including 18 focused diagnostic/report checks. Baseline
  UGO numerically matches its instrumented form without an ablation. Tests
  cover exception-safe filter restoration, group-blind APIs, validation before
  test construction, inactive generation, cache integrity, threshold ties,
  missing-group metrics and independent AP calculation.
- An independent post-run audit checked all record/array checksums and refitted
  **40 prespecified clean RF100 models** from cached fitting arrays. Independently
  calculated AP, achieved FPR, thresholds and subgroup recalls matched. The
  LogReg multiplicity-weight control also matched the corresponding duplicated
  data metrics in all 20 regime/seed comparisons.
- Exact reruns produced byte-identical manifests, mechanism summaries, per-seed
  metric CSV and numerical summaries; cached results were reused. Additional
  direct RF checks against unmodified UGO matched the traced fitting arrays.

Scientific scope: controlled software/simulation evidence only. No real-data
study, new sampler, claim of exact author-result reproduction or novelty claim
is made. Existing source-audit ambiguities and missing primary sources retain
the qualifications in the protocol and research proposal.

## 1. Did UGO reach a relevant generation mechanism?

Yes in the clean regime. Both original natural-budget UGO and the cap-480
reference entered minority generation in every seed for both learners. The
same-size unfiltered ranking probe contained excluded rare points in every
clean seed. Thus these were active opportunities, not hypothetical exclusion
in runs that stopped before generation.

| Regime / learner | Seeds entering minority generation | Total minority rounds | Seeds with excluded rare points in counterfactual ranked pool |
| --- | ---: | ---: | ---: |
| Clean / RF100 | 10/10 | 38 | 10/10 |
| Clean / LogReg | 10/10 | 70 | 10/10 |
| Overlap / RF100 | 10/10 | 11 | 5/10 |
| Overlap / LogReg | 10/10 | 74 | 10/10 |

In all reference runs, all three rare original rows were seed-ineligible but
still present in classifier fitting. In overlap RF100 seeds **44,45,46,49,51**,
generation occurred, but excluded rare points never entered the observed
trajectory's counterfactual same-size ranked pool. Do not attribute their
outcomes to lost *ranked rare-anchor access*. This is narrower than saying all
generation was inactive. The probe is fitting-only and does not establish an
identical counterfactual future trajectory.

Natural-budget UGO and cap-480 UGO had identical resampled arrays in all 40
dataset/learner cells: the larger cap produced no extra rows. Termination
reasons still differed in four overlap/LogReg cells: natural UGO stopped on
Welch while controlled UGO hit its cap after the same additions. Natural
phi=2.5 was retained as an original reference, not silently replaced by the
controlled phi=1 configuration.

## 2. Clean RF100 held-out results

Means across ten paired seeds, at each arm's validation-selected FPR operating
point. G0/G1/G2 are independently known evaluation modes; G2 is the 5% rare
mode. Numbers in recall/FPR columns are percentages. AP is score-based.

| Arm | G0 recall | G1 recall | G2 recall | Achieved test FPR | AP | Minority additions | Additions inside rare support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NORES | 100.00 | 99.88 | 92.80 | 4.90 | .99388 | 0 | 0 |
| Class weighting | 100.00 | 99.96 | 94.20 | 4.85 | .99407 | 0 | 0 |
| Original UGO / cap-480 k=5 | 100.00 | 99.72 | 86.20 | 4.99 | .98929 | 228 | 0 |
| Validation-tuned UGO k | 100.00 | 99.84 | 96.40 | 4.91 | .99310 | 204 | 2.3 |
| Generation-eligibility-disabled diagnostic | 99.99 | 99.88 | 96.00 | 5.13 | .99363 | 144 | 5.9 |
| Full-pre-filter-disabled diagnostic | 99.93 | 99.84 | 94.80 | 5.16 | .99334 | 204 | 13.1 |
| ROS, 480 additions | 100.00 | 100.00 | 97.80 | 4.93 | .99541 | 480 | 23.9 |
| ROS matched to generation diagnostic | 100.00 | 99.96 | 97.60 | 4.80 | .99543 | 144 | 7.7 |
| Matched multiplicity weights | 100.00 | 99.88 | 95.20 | 4.98 | .99405 | 0 | 0 |
| SMOTE | 100.00 | 99.92 | 91.60 | 4.86 | .99070 | 480 | 9.9 |
| KMeansSMOTE | 99.66 | 99.68 | 85.40 | 4.89 | .98828 | 482.4 | 174.2 |
| Existing EC-EBO | 100.00 | 100.00 | 99.60 | 5.07 | .99749 | 478.4 | 21.5 |

All additions above were minority-labelled; no majority additions occurred in
these clean RF100 arms. EC-EBO shortfalls and KMeans per-cluster ceiling effects
were retained. KMeans/EC-EBO are full-method comparisons with different
generation rules; they cannot isolate filtering. Support-region counts are
evaluated locations, not seed ancestry or a general proof of synthetic labels.

Raw values for all seeds, arms and regimes remain in ignored
`results/question-b-diagnostic-v1/metrics.csv` and checked job records. Reviewed
[aggregate metrics](question_b_diagnostic/aggregate_metrics.csv) additionally
include macro/worst recall, F1, balanced accuracy, population error, FPR ranges,
eligibility and generated counts for every arm.

## 3. Paired effects and what the controls establish

Exploratory percentile bootstrap 95% intervals resample whole paired simulation
replicates, not folds or correlated test rows. N=10 for each contrast. Secondary
intervals are unadjusted and are not confirmatory discoveries.

| Clean RF100 contrast | Mean worst-recall difference | 95% interval |
| --- | ---: | ---: |
| Generation eligibility disabled minus cap-480 k=5 | +9.79 pp | +2.19 to +22.80 pp |
| Full pre-filter disabled minus cap-480 k=5 | +8.60 pp | +1.80 to +17.80 pp |
| Validation-tuned k minus cap-480 k=5 | +10.20 pp | +1.80 to +23.00 pp |
| EC-EBO minus cap-480 k=5 | +13.40 pp | +3.60 to +27.40 pp |
| Generation eligibility disabled minus matched ROS | -1.57 pp | -4.96 to +1.36 pp |
| Generation eligibility disabled minus matched weights | +0.79 pp | -3.23 to +5.20 pp |
| Generation eligibility disabled minus NORES | +3.19 pp | -1.41 to +7.00 pp |

For the primary contrast, achieved FPR changed by **+0.142 pp** (95% interval
**-0.056 to +0.371 pp**) and AP by **+.00435** (**+.00144 to +.00885**).
The AP change cannot be caused by changing a decision threshold alone. The
same RF capacity, fitting originals, SMOTE helper, batch rule and cap were
used. This supports an effect of the generation-eligibility intervention in
this controlled setting, without identifying a unique rare-anchor mechanism.

However, the lower confidence limit did **not** exceed the frozen five-point
practical-effect margin. Nor was absolute FPR<=6% maintained in every primary
ablation replicate: its range was **4.22–6.26%**. The tuned UGO range was
**3.89–6.09%**, and EC-EBO **4.39–6.77%**. Empirical validation FPR control is
not a test/population guarantee. Count-matched ROS stayed at **4.21–5.46%**.
These results do not satisfy the stronger conditions for promoting a residual
failure into a new-method program.

The primary ablation used **fewer**, not more, additions on average (144 versus
228). Equal caps are not equal realized counts. Count-matched ROS/weights
account for realized per-class volume and already match or exceed recovery;
there is no measured unique benefit requiring a new geometric generator.
They do not form an exact count-locked UGO-versus-UGO replay. Thus filtering's
total policy effect, adaptive stopping/quantity and changed anchor/partner
composition cannot all be separated into independent causal effects here.

The safety-oriented subgroup objective is also different from ordinary
population error. Mean population error was **4.565%** for reference UGO and
**4.641%** for the primary ablation: +0.076 pp (95% interval -0.107 to +0.291 pp).
Improved rare-mode recall is not proof of better prevalence-weighted utility.

## 4. Three distinctions resolved by the evidence

**Seed exclusion versus training removal.** Reference UGO's mean eligible
minority counts were 41.8/12.5/0, yet all 600 original training rows, including
all three rare observations, remained. The generation-only ablation expanded
generation eligibility to 42/15/3 while retaining the original scoring mask
and post-generation correction. The full-pre-filter ablation additionally
changed scoring and relocation-reference access and has a broader interpretation.

**Few additions versus prediction loss.** NORES achieved 92.8% rare recall with
zero additions. Default UGO achieved 86.2% with zero rare-support additions;
their difference was not a proof that zero augmentation is inherently harmful.
KMeansSMOTE's large mean rare-support count did not establish high recall.
In **six of ten validation-selected UGO runs**, rare eligibility and rare-support
additions both remained zero, yet rare recall averaged **95.0%**. For example,
seed 49 improved from 34% to 96% using selected k=7, with zero rare eligibility
in both configurations. This is direct evidence against treating rare-anchor
restoration as necessary for prediction recovery.

**Filtering versus other causes.** The controlled within-RF intervention fixes
classifier family, inputs and generator, but changes later dynamics. Count and
threshold controls prevent attributing all changes to a novel seed mechanism.
At the default .5 threshold, reference UGO's worst recall was only **7.2%** with
test FPR **0%**; validation threshold selection raised it to **86.2%** at **4.99%**
FPR without changing a single fitted model. All principal comparisons therefore
use the same validation operating-point rule. The two thresholds are not fair
comparisons at the same achieved FPR.

LogReg's rare recall remained **0% in every evaluated arm** in both regimes.
Linear capacity cannot represent the concentric compact supports; its failure
is not evidence that an oversampler should create equal group counts or solve
the representation problem. RF100 is the relevant nonlinear control here.

## 5. Known-overlap negative control

In the overlap regime, majority conditional density contains .20 times the
minority feature law, so population **FPR >= .20 * weighted minority recall**.
At population FPR=.05, weighted minority recall can be at most .25. This bounds
aggregate recovery; it does not prove every individual group's observed deficit
is irreducible or optimally distributed.

| RF100 arm | Rare recall | Worst recall | Achieved test FPR | AP |
| --- | ---: | ---: | ---: | ---: |
| NORES | 5.0% | 5.0% | 4.94% | .333 |
| Original UGO | 2.0% | 2.0% | 4.90% | .331 |
| Validation-tuned UGO | 2.8% | 2.8% | 5.05% | .332 |
| Generation eligibility disabled | 7.2% | 7.2% | 4.96% | .336 |
| Full pre-filter disabled | 6.0% | 6.0% | 4.89% | .344 |
| ROS | 9.6% | 9.6% | 5.00% | .339 |
| EC-EBO | 2.6% | 2.6% | 4.91% | .351 |

None establishes a general fix for overlap. In half the original RF trajectories
the rare-ranking opportunity probe was absent despite generation. The poor
outcomes cannot uniformly be attributed to the rare-seed filter, and AP, worst
recall and the global mixture bound answer different questions.

## 6. Next research step and limitations

Use existing validated recipes rather than implement a new sampler. If pursuing
the scientific attribution further, freeze a fresh confirmation with independent
seeds/test draws and a **count-locked, fixed-schedule generation-pool diagnostic**
to separate quantity from eligibility effects. Keep it group-blind, clearly
modified, and separate from original UGO. Do not use this pilot's test results
to tune a revised rule and then claim independent confirmation.

The present evidence supports recovery by existing options but leaves important
limits: ten simulation seeds, one rare-mode size and geometry, one nonlinear
learner configuration, finite validation-FPR calibration, coupled anchor/partner
eligibility, and no noise-corruption or real-domain subtype study. Matching four
validation choices for NORES controls selection opportunity, not all internal
UGO computation. Comparator-specific neighbor/generation/count rules remain
different. The current evidence neither confirms a new mechanism nor proves
that all rare-mode problems are solved.

The primary five-point minimum benefit is not established, and there is no
demonstrated residual advantage beyond duplication/weights/existing methods.
Therefore the next stage is optional independent diagnostic confirmation and
source completion, **not new-algorithm implementation**.

## Reproduce and inspect

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.run_subgroup_diagnostic --phase mechanism \
  --output results/question-b-diagnostic-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.run_subgroup_diagnostic --phase pilot \
  --output results/question-b-diagnostic-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.report_subgroup_diagnostic \
  --input results/question-b-diagnostic-v1 \
  --output docs/research/question_b_diagnostic --replay-primary
```

The same source/protocol reuses checked fitting arrays and terminal records;
changed identities require a fresh output directory. Raw artificial job records
remain ignored. Reviewed evidence:
[metrics](question_b_diagnostic/aggregate_metrics.csv),
[paired contrasts](question_b_diagnostic/paired_contrasts.csv),
[mechanism summary](question_b_diagnostic/mechanism_summary.csv),
[validation choices](question_b_diagnostic/validation_selections.csv), and
[audit/provenance](question_b_diagnostic/validation.json).

Run ID: `76cf95f6ee7a40395358152c922ffae332f4c0cc138b5d0ef5e0826644f95335`.
Protocol SHA-256: `98ab2d9ee2c48dffcfb5df7c4496d5e7ec3acf7885ad94c7130b401aac8d10b9`.
