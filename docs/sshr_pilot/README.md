# SSHR Stage 4 pilot artifacts

These are descriptive exports from the two-dataset controlled study documented
in [SSHR_SMALL_STUDY.md](../SSHR_SMALL_STUDY.md). The registered local source
files are Ecoli4 and Yeast4; this is not the authors' benchmark or evidence of
general superiority. Algorithm interpretations and paper/code verification
status are recorded in [SSHR_METHOD_DECISIONS.md](../SSHR_METHOD_DECISIONS.md).

## Files

| Artifact | Ecoli4 | Yeast4 | Contents |
| --- | --- | --- | --- |
| Scores | [scores.csv](ecoli4/scores.csv) | [scores.csv](yeast4/scores.csv) | Held-out metric summaries, shared-fold and individual-method scopes |
| Coverage | [coverage.csv](ecoli4/coverage.csv) | [coverage.csv](yeast4/coverage.csv) | Every configured main-study method/classifier, including failures and missing jobs |
| Diagnostics | [diagnostics.csv](ecoli4/diagnostics.csv) | [diagnostics.csv](yeast4/diagnostics.csv) | Row accounting, pseudo-label auditing, geometric proxies, quotas, stops and duplicate counts |
| Clusters | [clusters.csv](ecoli4/clusters.csv) | [clusters.csv](yeast4/clusters.csv) | SSHR cluster/SMOTE-pool records for each repeat, fold and classifier |
| Unlabelled-data ablation | [ablation.csv](ecoli4/ablation.csv) | [ablation.csv](yeast4/ablation.csv) | Paired SSHR with-U minus without-U comparisons |
| Complete aggregate envelope | [summary.json](ecoli4/summary.json) | [summary.json](yeast4/summary.json) | All the tables, identities, policies and ablation coverage |

The CSV files use standard CSV quoting. List/dictionary cells, including
`fold_ids`, `totals`, `issues` and nested proxy counts, contain JSON; parse them
with `json.loads` after reading the CSV. Undefined scalar values are blank CSV
cells and JSON `null`. An empty dictionary means no available fields, not a
measured zero for every possible field.

This directory's `.gitattributes` preserves CSV/JSON artifact bytes without line
ending conversion. The byte hashes in the audit therefore remain valid in
GitHub and in Windows or Linux checkouts; the rest of the repository retains
its existing CSV line-ending policy.

[validation.json](validation.json) preserves the separate read-only audit of
all 800 saved jobs, source/run/artifact hashes, frozen-design reconstruction,
metric and pseudo-label recomputation, accounting assertions and explicit limits.
It contains no individual feature, gold-label, prediction or probability arrays.
Its assertion total counts repeated checks over rows and artifacts, not pytest
tests or independent experiments. `controlled_runner_file_sha256` identifies
`scripts/run_sshr_study.py`; `study_runner_file_sha256` identifies the fixed
`scripts/run_sshr_small_study.py` driver. The audit script itself remains local
in `/tmp/audit_sshr_pilot_saved_results.py`, with its hash recorded. Generated
coordinates are not saved, so independent per-row proxy recomputation is outside
the audit's scope; saved proxy aggregation and training-only code scope were checked.

## Metric schemas and comparison scopes

`scores.csv` has one row per dataset, classifier, method, scope and metric.
Metrics are ordinary minority F1, G-mean, MCC, probability ROC-AUC and
`fpr = FP/(TN+FP)`. Probability ROC-AUC is distinct from the SSHR paper's
printed AUC formula. `mean` and `sample_sd` describe fold scores;
`n_folds` and `fold_ids` give their denominator. A fold ID is the zero-based
pair `[repeat, fold]`, not just a fold number reused across repetitions.

- `all_methods_shared_success` is the canonical comparison: separately for
  each classifier, intersect successful finite-score fold events across all
  six selected methods. Every method's canonical mean uses that same set.
- `all_method_success` is a supplementary **individual-method** summary: use
  all successful fold events for that method, even if another method failed.
  Different methods can therefore have different denominators in this scope.

The serialized policy's phrase “all-method-success summaries” refers to the
second scope, not to another all-method intersection. Check `scope` explicitly
when selecting rows; the two scopes must not be counted as separate experiments.
`n_paired_with_nores`, `nores_mean` and `paired_mean_difference` use only selected
fold events that also succeeded for NORES. The difference is method minus
NORES on matched fold events, not a difference between unmatched means.

Each dataset has five repeats of five outer folds, two classifiers (`LogReg`
and `GNB`), and six main-study methods. All **600 main-study jobs succeeded**:
every method/classifier coverage row reports 25 successes, zero failures,
zero unavailable jobs and zero missing jobs. The canonical intersection has
25 fold events in every condition; the two score scopes consequently agree
numerically for this saved pilot. Each dataset exports 120 score rows,
12 coverage rows and 12 diagnostic rows.

The empty-U runs add **200 successful jobs**: NORES and SSHR on the identical
folds, masks and classifiers. `summary.json` → `ablation.coverage` records
25 paired successful SSHR fold events and 25 identical saved NORES controls
per classifier, with no issues. It is the coverage record for the ablation;
`coverage.csv` describes the main with-U run only. `ablation.csv` has ten rows
per dataset, one per classifier/metric. Its difference is **with U minus
without U**, and `paired_fold_event_counts` covers exactly those matched SSHR
events. The ablation measures access to U as a whole, including clustering
and selection changes; it does not isolate one component.

The repeated folds share observations and overlapping training sets. Their
sample standard deviations, including the SD of paired ablation differences,
describe spread; they are not standard errors, confidence intervals,
independent replications or significance tests. SD is undefined when fewer
than two fold events are available. This pilot supports no superiority claim.

## Diagnostics and denominators

SSHR selects **existing unlabelled observations and assigns pseudo-labels**.
Those rows are not new synthetic feature vectors. Only its within-cluster
SMOTE fallback creates vectors. `totals.selected_pseudo_labeled` and
`totals.generated_synthetic` keep these operations separate. Baseline
generation is also separated into `generated_minority` and
`generated_majority`: UGO can generate rows assigned to either class.

Diagnostic totals are sums of **successful fold events per classifier**.
They include L/U pool sizes, retained rows, majority/minority removals,
selection, generation, requested quotas and unmet quotas. A source row can
appear or be selected in multiple repeats. SSHR's clustering/resampling is
classifier-independent, so its matching diagnostics in LogReg and GNB are
copies of the same training-fold operations; do not pool them to claim twice
as many unique observations. The same cautions apply to cluster records and
ablation row counts.

`pseudo_label_event_weighted_accuracy` is `sum(correct)/sum(selected)` from
the saved retrospective hidden-label audit. It is undefined when no rows were
selected. `pseudo_label_mean_nonempty_fold_accuracy` instead averages each
nonempty fold's accuracy without selection-count weighting; its denominator
is `pseudo_label_nonempty_fold_events`. Hidden gold labels are evaluator-only
after resampling, never a clustering, confidence or generation input. The
reporter verifies count arithmetic and U-pool membership; hidden gold-label
correctness requires the original local dataset for an independent audit.

`sampler_status_counts` and `sampler_stopping_reason_counts` retain successful
no-op and partial-generation outcomes. A classifier job can succeed despite
an unmet resampling quota. In this pilot SSHR reported three shortfall fold
events on Ecoli4 and one on Yeast4 per classifier; EC-EBO reported partial
generation on eleven Yeast4 fold events per classifier. These are preserved
in the diagnostics rather than relabelled as balanced outputs.
`shortfall_causes_all_saved_jobs` includes available failure diagnostics too.
`failed_job_removal_totals` and their reporting counts retain available
removals from unsuccessful jobs; absent fields remain unknown.

`training_overlap_proxy` counts additions whose nearest original observed
majority distance is strictly smaller than their nearest original observed
minority distance. Every method uses the same original reduced L reference,
before sampler-specific removals. Rate denominators are added rows, ties are
separate, and zero-row rates are undefined. SSHR has `pseudo_labeled` and
`smote` groups; supervised baselines have `synthetic_all`,
`synthetic_minority` and `synthetic_majority`. Generated majority rows must not
be interpreted as minority intrusions. This training-only distance proxy does
not establish true overlap or class membership.

`coordinate_duplicate_fold_event_counts` concerns exact feature equality,
not source-row provenance. Per-field `coordinate_duplicate_reporting_coverage`
states how many successful events supplied the count and how many did not;
missing counts are not filled with zero.

`clusters.csv` identifies each final surviving SSHR cluster by its persistent
`cluster_id`, repeat/fold, classifier and run. Cluster `size` is retained
observed L plus its original assigned U after cleaning/weighted removal,
before adding pseudo-labelled or SMOTE rows. `observed_minority_count` and
`observed_majority_count` use observed labels only. `unanchored` means no
observed L anchor; `minority_anchored` means at least one observed minority
anchor, which alone does not establish eligibility. `smote_pool_size`,
`smote_requested`, `smote_generated` and `smote_shortage_reason` describe
available fallback allocation records; blanks mean no reported allocation.

## Traceability and local reproduction

The [source register](../../data/manifests/sshr_pilot_sources.json) preserves
portable input paths and SHA-256 hashes. Each summary retains the source-file
hash, processed-input hash, source/software fingerprint, run IDs, fold/mask
hashes, fixed configuration, feature/provenance information and frozen design.
Hash verification identifies the supplied files; it does not prove identity
with the original authors' files, folds or unlabelled-data protocol.

The full local evidence is under
`results/sshr-small-study/<dataset>/{with-u,without-u}/`: `manifest.json`
defines the exact run, and atomic `job-*.json` envelopes retain predictions,
probabilities, lineage and diagnostics with integrity hashes. Those job
records and raw `.dat` files are ignored by Git. Supplied archives and papers
remain local and are not part of these committed artifacts. Shared summaries
include row-index/group design metadata for traceability, but no raw feature
vectors or individual held-out predictions/probabilities/gold-label arrays.

With the registered local raw files and the same tested Python dependencies,
run from the repository root:

```bash
python -m scripts.run_sshr_small_study --output results/sshr-small-study --report-dir docs/sshr_pilot
```

The runner reuses existing terminal jobs when run identities match. A changed
dataset, source fingerprint, configuration or software identity requires a
new output directory; existing evidence is rejected rather than overwritten
as if it were the same experiment. GitHub report files alone cannot resume
training because the ignored local job records are required.

To recheck and re-export existing evidence without classifier fitting, use
[sshr_study_report.py](../../src/sshr_study_report.py):

```python
from pathlib import Path
from src.sshr_study_report import (
    summarize_controlled_run, summarize_unlabeled_ablation, export_study_reports,
)

for dataset in ("ecoli4", "yeast4"):
    run = Path("results/sshr-small-study") / dataset
    report = summarize_controlled_run(run / "with-u")
    ablation = summarize_unlabeled_ablation(run / "with-u", run / "without-u")
    export_study_reports(report, ablation, Path("docs/sshr_pilot") / dataset)
```

The reporter checks manifest/job hashes, expected job identities, masks,
held-out metric arithmetic, row/lineage/quota conservation and matching
ablation identities. It also compares duplicated NORES scores, predictions
and probabilities. A corruption or mismatch fails loudly; incomplete coverage
is reported explicitly.
