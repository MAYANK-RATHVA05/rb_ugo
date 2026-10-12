# Stage 4R-6: measured simulation pilot

The selected simulation pilot is complete: **300/300 fold jobs succeeded**, with
zero failed or unavailable jobs. This means fitting, prediction and scoring
completed, not that every UGO run satisfied its statistical stopping condition.
No real benchmark dataset was executed and this is not an exact reproduction
of the published results.

## Fixed matrix and provenance

The [frozen configuration](../../configs/stage4r6_pilot.json) uses seed 42,
LogReg/GNB/KNN, NORES/ROS/SMOTE/UGO-ROS/UGO-SMOTE, ten shared stratified folds
and one native computation thread. The three classifiers are disclosed sklearn
approximations, not verified author packages. Five unavailable paper classifier
implementations were outside this selected pilot; they are not counted as passed.

The linear input has 1,050 rows, observed labels 985/65, positive label 1 and
15 majority-to-minority relabels. The nonlinear input has 1,000 rows, observed
labels 925/75, positive label 0 and 17 relabels. The linear normal parameter is
interpreted as standard deviation. The author does not supply exact coordinates,
seed, all distribution choices or folds; Figure 7's nonlinear example instead
shows 915/85. See [the input reconstruction choices](EXPERIMENT_SETUP.md).

Preprocessing is fitted on original training partitions and final classifiers
are freshly fitted after resampling, following [the evaluation contract](EVALUATION.md).
UGO uses the declared Section 3.3 cap, the existing source-audited defaults and
the engineering limit of 100 iterations. No parameters were tuned to these
outer-test results, and no data/model/method was substituted to improve scores.

## Observed outcomes

These are descriptive mean MCC scores over the same ten successful folds per
dataset/classifier. They are not independent-dataset statistical evidence.

| Simulation | Classifier | NORES | UGO-ROS | UGO-SMOTE |
| --- | --- | ---: | ---: | ---: |
| Linear | LogReg | 0.5445 | 0.6402 | 0.6348 |
| Linear | GNB | 0.6478 | 0.6300 | 0.6328 |
| Linear | KNN | 0.6412 | 0.6305 | 0.6448 |
| Nonlinear | LogReg | 0.0000 | 0.0004 | 0.0602 |
| Nonlinear | GNB | 0.3024 | 0.5854 | 0.5992 |
| Nonlinear | KNN | 0.7285 | 0.7072 | 0.7132 |

UGO therefore does not uniformly improve NORES in this pilot. LogReg improves
on the linear input and GNB improves on the nonlinear input; other comparisons
are mixed. The [complete paired scores](pilot_stage4r6/paired_scores.csv) include
all five methods, IAM/MCC/F1, paired counts, baseline means, paired mean
differences, sample SDs and explicit undefined-score counts. No significance
test or confidence claim is made from these two simulations.

Of the 120 UGO jobs, **19 stopped at the generation cap**:

- Nonlinear LogReg: ten UGO-ROS and eight UGO-SMOTE folds.
- Nonlinear KNN: one UGO-SMOTE fold.

The other 101 stopped when the Welch test failed to reject. No job reached the
100-iteration safeguard. Linear LogReg/GNB/KNN used on average 95–294.5 added
rows under UGO, versus 828 for ordinary balancing. Nonlinear LogReg UGO-ROS
added an average of 3,262.5 rows, exceeding the 765 needed for numerical balance;
the method can select the majority class under its uncertainty rule. These
cap-limited outcomes must stay visible in later reports rather than being
described as statistical convergence or universal generation savings.

Nonlinear NORES LogReg predicts only the majority class in all ten folds. Its
minority F1 is zero and MCC uses the declared zero-denominator zero convention,
with all ten undefined flags retained. Other degenerate predictions remain
counted in the exported summaries.

## Comparison with the article

The [audited reference values](../../data/manifests/ugo_simulation_reference.json)
contain 90 selected cells from Tables 9–10 (pp13–14). All were checked against
independently parsed PDF table text; Table 10's rendered page was also inspected
to confirm negative signs. The [paper comparison](pilot_stage4r6/paper_comparison.csv)
records each published value, our paired-fold mean, difference, source table/page
and fidelity limits. Differences are descriptive, not evidence of exact agreement
or acceptance targets for tuning.

For example, linear LogReg UGO-ROS MCC is **0.6402 here versus 0.574 in Table 9**.
Nonlinear GNB UGO-SMOTE MCC is **0.5992 here versus 0.706 in Table 10**.
Seeds/coordinates, class counts, fold assignments, preprocessing, estimator
defaults and ambiguous UGO conventions prevent isolating the causes of these
differences from the available sources alone.

There is an additional **F1 averaging ambiguity**. Section 4.3 gives a
positive-class F1 formula, but Table 10 reports LogReg NORES IAM=-0.084, MCC=0
and F1=0.478. An always-majority predictor on the illustrated 915/85 counts
would have minority F1=0, majority F1 approximately 0.956 and macro F1
approximately 0.478. This suggests an averaging mismatch; it does not prove
which implementation the authors used because their fold predictions are
unavailable. Our explicit minority-positive definition remains fixed. All F1
paper comparisons are marked as averaging-unresolved. Author code or fold-level
outputs are needed before claiming that their F1 values have been reproduced.

## Measured resources and resume

The first complete invocation ran all 300 jobs, with no cache hits, in
**25.11 seconds**. Linux/x86_64 process-lifetime peak resident memory was
**176,619,520 bytes = 168.44 MiB**. It includes interpreter/library imports,
simulation inputs, native allocations and retained records. It is not a
per-classifier/per-fold peak or an incremental allocation measurement. OpenMP
and both OpenBLAS pools reported one thread while the pilot ran.

Replaying the exact matrix took **1.50 seconds**, reused all 300 records and
performed zero new fits. The original measured execution metadata and compact
exports were preserved. Cached replay time is not reported as training runtime.
The [summary JSON](pilot_stage4r6/summary.json) records exact manifests/folds,
input/source hashes, configurations, classifier fidelity, record hashes,
diagnostics, original measurement and native library context. Detailed prediction
records remain in the ignored local `results/stage4r6-pilot/` directory.

This small two-feature/classifier subset cannot predict the resource needs of
74 datasets, 500-tree RF, Bayesian models, dense MCSMOTE matrices or LEAO.
Continue sequential execution until those workloads are separately measured;
do not extrapolate a full-matrix completion time from this pilot.

## Reproduce and verify

From the repository root at this implementation revision:

    .venv/bin/python main_stage4r6.py --report-dir docs/reproduction/pilot_stage4r6
    .venv/bin/python scripts/pilot_paper_comparison.py --summary docs/reproduction/pilot_stage4r6/summary.json --output docs/reproduction/pilot_stage4r6/paper_comparison.csv
    .venv/bin/python -m pytest -q

The first command resumes existing matching records. For a fresh measurement,
use `--output results/stage4r6-fresh --report-dir /tmp/stage4r6-fresh-report`.
Changed code/configuration/input identities require a new output directory.
The exporter refuses to replace reports belonging to different run identities.
Native runtimes/memory can vary; recorded source/configuration/score identities
permit auditing the measured invocation.

The full suite passed **322 tests**, with no failures or skips. Independent
checks verified every one of the 300 record hashes, training/test disjointness,
saved IAM calculation, sklearn MCC/minority F1 and exact report recomputation.
Tests also verify paired coverage under failures/missing scores, unsupported
memory status, preserved measurements on resume and missing reference handling.

## Remaining work

This completes the authorized simulation pilot. The later dataset collection
prepared all 74 inputs, with 56 metadata matches and 18 explicitly accepted
approximations, and the classifier-completion stage added all fourteen
interfaces. Author preprocessing, F1 averaging, folds, several exact package
choices, and cap conventions remain unresolved for exact claims.
Stage 4R-7's additional analyses and Stage 4R-8's final report require separate
authorization under AGENTS.md.
