# Focused Gaussian exposure mechanism diagnostic

The initial grid does **not justify an adaptive extension**. Threshold matching
and weighted PL match the budget within the preregistered practical margins in
both adequately active cells. The budget lowers expected endpoint risk versus
count matching in one cell, but does not translate that change into better MCC,
G-mean or AP. This is a limited negative mechanism result, not a universal
impossibility claim or a statistical equivalence test.

## Frozen scope and evidence interpretation

[Preregistration](EXPOSURE_MECHANISM_PREREGISTRATION.md) and
[configuration](../../configs/exposure_mechanism_v1.json) were committed and pushed
as `23b2afe` before this grid was drawn. The fixed-trace prototype, previous runner,
matched controls and earlier evidence remain byte-identical; hashes are checked
before execution. No adaptive algorithm was implemented.

The earlier pilot's interpretation was confirmed against its saved manifest and
aggregate: no realized lineage-error reduction, one-endpoint Ecoli2 diversity,
inactive overlap/Ecoli4 cases, and no unique predictive benefit. It motivated
this separate experiment but is not counted as additional independent seeds.

The new grid uses separation 1/3, labeled minority count 8/32, U minority prior
.02/.20, 200 labeled majority rows, 400 U rows, and 4,000 independent test rows
at minority prior .20. There are two equally weighted Gaussian minority modes.
Each cell uses independent dataset draws for seeds 42–51, with algorithm seed
42. All 80 cases and **560/560 jobs completed**, with zero failures. The six
requested conditions and NORES use identical observed L; the U conditions share
one guide, eligible bank, anchor trace, interpolation randomness and candidate
check. Threshold .9, budget rate .025 and endpoint cap five were never adjusted.

Oracle probabilities, hidden U labels and latent modes enter only the evaluator,
after all seven models and generation decisions are frozen. Oracle endpoint and
child risks use the cell's U prior; this varies independently of the fixed test
prior. No oracle prior correction is given to any method. L-only standardization
and the original guide calibration remain unchanged. Raw draws, fitted arrays,
ledgers, probabilities and individual job records stay under ignored
`results/exposure-mechanism-v1-reviewed/`.

## Activity and endpoint diversity

Counts below retain all ten seeds per cell. Eligible endpoints are the distinct
union across minority rounds. Binding and restrictive rates have **usable
minority-generation rounds**, rather than all cases, as their denominator.

| Separation | L minority | U prior | Active budget cases | Mean eligible endpoints | Binding / usable rounds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | .02 | 0/10 | 0 | undefined, 0 usable |
| 1 | 8 | .20 | 0/10 | 0 | undefined, 0 usable |
| 1 | 32 | .02 | 0/10 | 0 | undefined, 0 usable |
| 1 | 32 | .20 | 0/10 | 0 | undefined, 0 usable |
| 3 | 8 | .02 | 0/10 | 0 | undefined, 0 usable |
| 3 | 8 | .20 | 0/10 | 0 | undefined, 0 usable |
| 3 | 32 | .02 | 5/10 | .6 | 2/10 = 20% |
| 3 | 32 | .20 | 8/10 | 10.6 | 14/14 = 100% |

Overall, 13/80 cases exercised the budgeted U branch. Of 67 inactive cases,
38 had no minority-generation round and 29 had no eligible endpoint. There
were no additional candidate-rejection-only or integer-allocation-only inactive
cases. The existing guide/UGO decisions therefore gate most of the grid; those
cases cannot test descendant allocation. Their classification outcomes remain
in unconditional paired comparisons, while lineage/child errors are undefined.

The exposure constraint was binding and restrictive in 16/24 usable rounds.
Restrictiveness was checked against a loose-exposure continuous solution with
the same objective. Conservative flooring loss and integer count reduction were
recorded separately; they were not treated as proofs that exposure was binding.

In the rare-U active subset, each method used a mean 1.2 distinct endpoints
per case. In the .20-U active subset, budget used 4.5 distinct endpoints with
effective endpoint count 4.18; count matching used 6.0 and effective count 6.00;
threshold matching used 1.75 and effective count 1.75. Unconstrained generation
used 13.25 distinct endpoints. The first active cell is severely diversity-limited.

## Mechanism outcomes

Values are means of **active dataset-seed error rates**, not pooled descendant
trials. Rare-U has five paired active seeds; .20-U has eight. Weighted PL has
endpoint multiplicities but creates no child vectors. Original UGO has no U
endpoints, so neither receives an invented U-child error value.

| U prior | Method | Realized lineage error | Oracle expected endpoint error | Oracle U-child error |
| ---: | --- | ---: | ---: | ---: |
| .02 | Budget | .200000 | .026714 | .080262 |
| .02 | Unconstrained | .200000 | .026714 | .079844 |
| .02 | Count matched | .200000 | .026714 | .080262 |
| .02 | Threshold matched | .200000 | .026714 | .080262 |
| .02 | Weighted PL | .200000 | .026714 | undefined |
| .20 | Budget | .000000 | .002735 | .016066 |
| .20 | Unconstrained | .000000 | .005709 | .018801 |
| .20 | Count matched | .000000 | .006427 | .019951 |
| .20 | Threshold matched | .000000 | .003024 | .019654 |
| .20 | Weighted PL | .000000 | .002735 | undefined |

The rare-U realized error reflects one all-wrong-endpoint case among five,
not a calibrated 20% error probability. The budget did not reduce this error
relative to any matched control. In the .20-U cell every selected endpoint's
realized label was minority, but oracle error is positive: zero observed errors
must not be interpreted as zero underlying risk.

For .20 U, budget minus count matching oracle endpoint error is
**-.003691 ± .000826 SE**, with seven favorable seeds and one tie. Oracle child
error changes by **-.003885 ± .002342 SE**, with five favorable, one tied and
two unfavorable seeds. Against threshold matching, these effects are
**-.000289 ± .000541 SE** for endpoints and **-.003588 ± .004512 SE** for
children. These are descriptive SEs across independent dataset seeds, not
confirmatory intervals. The primary endpoint differences are below the
preregistered .01 practical margin.

Endpoint risk and interpolation risk differ substantially. In both active cells,
budgeted child risk exceeds endpoint risk. At matched total fitting weight,
budget minus weighted PL expected error of **all retained added minority mass**
is +.012808 ± .002697 SE in rare U and +.003597 ± .001298 SE in .20 U. All five
and all eight active seeds, respectively, favor weighted PL on that measure.
This diagnoses a possible interpolation cost, not evidence that endpoint
allocation alone solves synthetic-label error.

## Minority modes

Coverage uses the preregistered rule: a mode must receive at least one oracle
expected minority unit of added training mass. This uses fractional mixture
posteriors; it does not assign generated points verified latent labels.

Rare-U U-child mode coverage is .50 for budget, count and threshold matching.
In the .20-U active subset it is .9375 for budget, .8750 for count matching and
.8125 for threshold matching. Including supervised fallback, all six UGO-based conditions
have identical mean added-mass mode coverage within each active cell: .90 over
all ten rare-U cases and 1.00 over all ten .20-U cases. These describe added
mass, not total L coverage. NORES adds no mass and has coverage zero by definition.

Higher child-mode coverage did not produce better test-mode recall. For .20 U,
mean recalls across all ten seeds were:

| Method | Minority mode 0 recall | Minority mode 1 recall |
| --- | ---: | ---: |
| Original UGO | .85338 | .83834 |
| Budget | .84775 | .83469 |
| Count matched | .84823 | .83496 |
| Threshold matched | .84725 | .83445 |
| Weighted PL | .84703 | .83272 |

The CSV evidence contains mode masses, exact test-count summaries and variation
across seeds, including the inactive cases.

## Classification and paired effects

The following .20-U cell means include all ten seeds, with the same test data
within each seed. The remaining active cell and all inactive cells are fully
reported in [method_summary.csv](exposure_mechanism_v1/method_summary.csv).

| Method | MCC | G-mean | AP |
| --- | ---: | ---: | ---: |
| Original UGO | .837479 | .907525 | .945611 |
| Budget | .837199 | .905586 | .945814 |
| Unconstrained | .836827 | .901408 | .946555 |
| Count matched | .837455 | .905791 | .945865 |
| Threshold matched | .836954 | .905385 | .945795 |
| Weighted PL | .837255 | .905043 | .945812 |
| NORES | .824331 | .879314 | .946842 |

Conditional paired effects below use the eight active .20-U seeds. Each is
**budget minus control, mean ± SE**. Higher classification metrics are better.

| Control | Delta MCC | Delta G-mean | Delta AP |
| --- | ---: | ---: | ---: |
| Original UGO | -.000350 ± .000606 | -.002423 ± .000746 | +.000254 ± .000204 |
| Unconstrained | +.000465 ± .001459 | +.005223 ± .001595 | -.000927 ± .000814 |
| Count matched | -.000320 ± .000156 | -.000256 ± .000125 | -.000064 ± .000031 |
| Threshold matched | +.000307 ± .000258 | +.000251 ± .000171 | +.000024 ± .000040 |
| Weighted PL | -.000070 ± .000294 | +.000680 ± .000195 | +.000002 ± .000028 |

Budget matched or lost MCC to count matching in all eight active seeds and lost
AP in six. Original UGO had higher G-mean in all eight. There is a G-mean gain
relative to unconstrained generation, but it trades against AP and does not
survive the simpler matched alternatives. In rare U, budget, count and threshold
matching produced identical predictions; differences versus weighted PL were
well inside the .005 downstream practical margins.

Pooled summaries first average the eight cell effects **within each seed**, then
summarize ten seed-level averages. For budget versus original UGO, pooled mean
changes are MCC -.0000685, G-mean -.0003930 and AP +.0000477. For budget versus
count matching, they are -.0000320, -.0000256 and -.00000645. These small pooled
effects partly reflect 67 inactive cases; conditional results above are essential
to the mechanism interpretation. No significance or equivalence claim is made.

## Runtime, verification and recommendation

The final measured run took **11.41 seconds**, including local artifact writing,
with one numerical thread and sequential case execution. Shared preparation
(UGO, guide and bank) averaged 21.74 ms per case, totaling 1.74 seconds. Additional
activity diagnostics totaled .040 seconds. Mean incremental replay/fit/predict
times were 1.59 ms for parent, 1.86 for budget, 1.87 for unconstrained, 1.84 for
count matching, 1.80 for threshold matching, 1.82 for weighted PL and 1.36 for
NORES. Shared preparation was reused, so these are **not standalone algorithm
benchmarks**. Serialization, evaluation and shared work account for the remaining
wall time; timing is environment-dependent.

The preregistered descriptive rule rejects the current added-value claim **on
this grid**: threshold matching and weighted PL each match primary error and all
three downstream metrics within the declared margins in both cells with at
least five paired active seeds. Six cells remain under-identified for allocation.
No new seeds, threshold changes or favorable subset searches were performed.

Recommendation: **stop expansion to the adaptive algorithm**. Preserve this as
negative/limited evidence. If revisiting the research question in a separately
specified study, first distinguish eligibility/membership limitations, multiplicity
allocation and interpolation cost. A more complex adaptive loop would not answer
the present failure to outperform matched controls. This does not rule out all
budgeting methods, other distributions or differently justified hypotheses.

An independent audit rechecked all 560 metric records, Gaussian-density oracle
risks, realized errors, lineage coordinates, mode masses, counts, weights and
budget feasibility. The focused suite passed 10 new tests and the full suite
passed **766 tests, zero failed and zero skipped**. The
[validation record](exposure_mechanism_v1/validation.json) preserves these checks.
An initial run was retained, followed by the same frozen-grid
replay after a reporting-only correction to secondary-error win directions and
additional mode summary fields. All 80 fitting-decision hashes and all 560
probability vectors matched exactly. This was not a seed/parameter search.

## Reproduce and evidence files

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m scripts.exposure_mechanism_diagnostic --output results/exposure-mechanism-new
.venv/bin/python -m scripts.report_exposure_mechanism --run results/exposure-mechanism-new --output /tmp/exposure-mechanism-new-report
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

Use fresh output directories; the commands refuse to overwrite prior results.
The [cell summary](exposure_mechanism_v1/cell_summary.csv),
[method summary](exposure_mechanism_v1/method_summary.csv),
[paired effects](exposure_mechanism_v1/paired_effects.csv),
[pooled seed effects](exposure_mechanism_v1/pooled_seed_effects.csv) and
[run evidence/decision](exposure_mechanism_v1/run_evidence.json) retain denominators,
seed IDs, means, SDs, SEs, ranges, wins/ties/losses, source identities and runtime.
Raw individual job records and oracle-bearing input arrays are not committed.
