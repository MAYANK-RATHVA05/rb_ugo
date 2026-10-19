# Rare-subgroup pilot runner

Status: runner implemented and checked on reduced artificial software fixtures. The frozen 30-case research pilot has **not** run. Execution remains disabled in the configuration because this stage prepares software; there are no new research scores to present.

## Safe local check now

No external datasets or new dependencies are needed for this controlled synthetic pilot. From the existing checkout:

```powershell
cd C:\ugo\ugo_base
git switch rare-subgroup-research
git pull --ff-only origin rare-subgroup-research
.\.venv\Scripts\python.exe -m scripts.run_rare_subgroups_pilot --check-only
.\.venv\Scripts\python.exe -m pytest -q tests\test_rare_subgroups_pilot.py tests\test_aw_smote.py
```

Check-only prints **30 cases, seven methods, at most 420 classifier fits, and 210 score rows**. It generates no arrays, fits no scaler/sampler/classifier, creates no output directory, and reads no raw datasets. Omitting both mode flags also performs only this check.

The later experiment stage must enable `execution_enabled` in `configs/rare_subgroups_pilot_v1.json` and freeze that configuration and the runner in a clean commit before using `--run`. The CLI rejects `--run` while execution is disabled, or while tracked changes are uncommitted. These are the existing staged protocol requirements; the current stage does not run the pilot.

## What the runner implements

The numerical design is unchanged from [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md): two features, two minority components, six settings and seeds 42–46. A semantic SHA-256 freezes the scientific design fields. Changing a seed, grid, distribution, convention or metric makes this version reject the configuration rather than silently run a different study. Progress metadata can change without changing the scientific design.

| Module | Responsibility |
| --- | --- |
| [`src/rare_subgroups_protocol.py`](../../../src/rare_subgroups_protocol.py) | Frozen-design loading, independent keyed Gaussian partitions, array fingerprints, negative-only threshold calibration and held-out metric arithmetic. |
| [`src/rare_subgroups_study.py`](../../../src/rare_subgroups_study.py) | Fitting/selection on X/y only, ordinary baselines, shared AW generation, validation AP selection, and evaluation after selection locks. |
| [`src/rare_subgroups_report.py`](../../../src/rare_subgroups_report.py) | Complete planned coverage, identical-seed summaries and the predeclared descriptive screening rules. |
| [`scripts/run_rare_subgroups_pilot.py`](../../../scripts/run_rare_subgroups_pilot.py) | Preflight, explicit execution, source/environment identity, complete-case checkpoints, predictions and report files. |

`StandardScaler` fits once on original fitting features. The `train_methods` API receives only scaled fitting X/y, selection X/y, fixed configuration and seed. It cannot receive calibration/test arrays or per-row subgroup IDs. Only after training and selection finish are calibration/test features transformed and scored. Hidden identities enter the evaluator and retention logger, not the two-feature matrices or selection objective.

The seven treatments are NORES, class-weighted forest, ROS, SMOTE, AW-Remove, seed-filter-only retention and validation-tuned AW-Remove. All forests use the fixed protocol settings; no classifier hyperparameters are selected. Retention restores originals around the actual AW-Remove synthetic rows, guaranteeing shared generation rather than relying on a redraw. The fixed default AW model is reused in the nine-setting search. Exact AP ties prefer the default, then ascending MinPts and percentile. The chosen model is reused without fitting on validation/calibration data. Unselected candidates never predict the test set. If tuned AW chooses the default, the final predictions are also reused.

Every reported treatment gets its own calibration threshold from the same independent negative-only calibration partition. The rule uses `score >= threshold` and handles score ties conservatively. If no finite candidate meets the empirical FPR target, a null threshold plus `threshold_is_infinite=true` records the positive-infinity threshold. That is a legitimate all-negative operating point, not a failed metric or a population-FPR guarantee.

## Files produced by a later run

The default output location will be `results\rare-subgroups-pilot-v1`. Nothing is written there by check-only.

| Path under the output directory | Contents |
| --- | --- |
| `manifest.json` | Full configuration, semantic/file fingerprints, source-file hashes, Git commit, platform/Python/package versions and planned jobs. |
| `job-<hash>.json` | Integrity-checked completed case: seven method statuses, scores or failure reasons, all nine selection candidates, scaler parameters, input/output hashes, timings, actual classifier-fit attempts, group retention and AW provenance. |
| `predictions\case-<hash>.npz` | Compressed held-out y/group arrays and positive-class probability vectors for successful methods, with checksums. No model pickle is needed. |
| `reports\scores.csv` | All 210 planned rows, including failed, unsupported and missing rows with blank metrics. `fit_rare_count` identifies the fitting setting; `rare_count` is the held-out metric denominator. |
| `reports\coverage.csv` | Successful/unsupported/failed/missing seeds for each setting and method, plus the shared-success count. |
| `reports\summary.csv` | Mean and sample SD per setting, method and metric on identical successful seeds across all seven methods. A single seed has null SD; no shared cases give null means. |
| `reports\screening.json` | The predeclared practical-effect rules, paired differences, actual FPR differences, original group losses and ordinary-control comparisons. |
| `reports\status.json` | Overall completion, saved/planned cases, successful score rows, known completed-case fit attempts and summary scope. |

Metrics include rare/common recall, overall positive recall, actual test FPR, F1, noninterpolated AP and worst-group recall. Score rows also include calibration FPR, confusion counts and denominators. Unlike metrics are never averaged together, and related synthetic settings are not described as independent real datasets.

The maximum of 14 fits per case counts six fixed treatments and eight additional grid candidates. Row-level timing/fit metadata for tuned AW refers to its reused candidate; the case-level fit-attempt counter counts each actual fit only once. The 420-fit plan assumes an uninterrupted traversal: rerunning an unfinished case can add work, and the completed-checkpoint counter does not include unfinished attempts. Group retention describes original observations; generated seed/partner provenance does not certify the true class of a synthetic location.

## Failures, interruptions and resuming

Unsupported AW cases do not substitute ROS, retain all seeds, reduce k or get a zero score. Expected sampler failures and classifier/prediction errors keep structured statuses and reasons. The tuned search considers only candidates with successful fitting and finite validation AP. If all candidates fail, tuned AW fails too. Test prediction failure never triggers selection of another candidate.

One process should write a result directory at a time. A case is checkpointed only after its method evaluations and prediction artifact finish. After an interruption, repeating the same later `--run` command reuses intact completed cases and reruns the unfinished case. Source/configuration/environment changes reject that directory. Saved record hashes and prediction checksums are verified before any new fit; corruption is reported rather than silently recomputed. Reports include all durable cases and mark unfinished cases missing. A checkpoint's top-level `status=success` means the case record was completed; `case_status` and individual method statuses determine whether model evaluations succeeded.

No automatic retries with new seeds/settings occur. `--run` returns 0 for complete method coverage, 1 for completed traversal with partial coverage, 2 for a preflight/identity/configuration error, and 130 for a controlled keyboard interruption. Unexpected programming errors stop the command with a traceback; they are not converted into successful model evaluations.

Screening uses the protocol's practical thresholds, with a 1e-12 numerical tolerance at inclusive floating-point boundaries. Its original-loss condition follows the protocol's "a group loses originals" wording; common and rare losses are reported separately. A qualifying association does not prove rare-row deletion alone caused the effect, particularly when final class totals differ. Statuses such as `trigger_not_met`, `simple_controls_suffice`, or `follow_up_required` do not establish statistical significance, noise robustness or novelty. Incomplete method coverage gives `incomplete_coverage`.

## What was checked

Software tests cover independent keyed streams and paired prefixes; unchanged fitting/selection when calibration features, test labels or group IDs are perturbed; original-training-only scaling; default and nondefault AP selection; no post-selection refit; paired AW generation; hand-derived calibration/F1/AP/recall/FPR arithmetic; explicit unsupported/prediction failures; same-seed summaries; screening boundaries; source/design rejection; and interruption/resume/checksum behavior. A small real forest fixture exercises the complete computation without running the research design.

Cloud verification with Python 3.11.17 and the repository environment: **23 runner tests passed, 76 combined runner/AW tests passed, and 824 tests passed in the full suite**, with no reported failures or skips. The real check-only command, scientific-design fingerprint, local document links, JSON metadata and absence of a pilot output directory were also verified. These are software checks, not pilot performance results.

The next stage is to enable and run the frozen pilot, inspect its coverage and diagnostics, and interpret the measured results before deciding whether any new method is warranted. This runner does not modify the previous UGO/BFCH-SMOTE/EC-EBO/SSHR implementations or their 30-dataset reports.
