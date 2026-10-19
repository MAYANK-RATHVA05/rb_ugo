# Contamination-study runner and software checks

Status: the first frozen experiment completed with **90 cases, 630 successful method evaluations and 1,260 classifier fits** on run commit `975020a`. Read the [audited results](CONTAMINATION_RESULTS.md) for the unmet screen and strong ordinary controls. Tests use reduced artificial fixtures; their numbers are not research evidence. The [prospective protocol](CONTAMINATION_PROTOCOL.md) and its scientific SHA-256 are unchanged from protocol commit `bff7ce5`.

## What you can run locally now

No new data downloads or dependencies are required. In the existing Antigravity PowerShell terminal:

```powershell
cd C:\ugo\ugo_base
git switch rare-subgroup-research
git pull --ff-only origin rare-subgroup-research
.\.venv\Scripts\python.exe -m scripts.run_rare_subgroups_contamination --check-only
.\.venv\Scripts\python.exe -m pytest -q tests\test_rare_subgroups_contamination.py tests\test_rare_subgroups_pilot.py tests\test_aw_smote.py
```

Preflight prints **90 cases, 630 method evaluations, 810 AW candidate records and at most 1,260 classifier fits**. It generates no arrays, trains no model and creates no output directory. Omitting both mode flags also performs only preflight. Execution metadata is now enabled for the authorized experiment; `--run` additionally requires a clean committed configuration and implementation.

The measured report, compact tables and figures are included in Git; viewing them requires no retraining. For a new local numerical run, use a fresh output directory, for example `results\rare-subgroups-contamination-v1-local`. The original cloud checkpoints belong to run commit `975020a` and will reject a different publication-commit identity.

## Implementation and isolation

| File | Responsibility |
| --- | --- |
| [`src/rare_subgroups_contamination.py`](../../../src/rare_subgroups_contamination.py) | Strict frozen-design loading, original Gaussian stream reuse, nested label-only corruption, independent tie uniforms and calibration/metric arithmetic. |
| [`src/rare_subgroups_contamination_study.py`](../../../src/rare_subgroups_contamination_study.py) | Observed-X/y training, reference-library provenance, reused AW fits/selection vectors, post-selection hidden-truth diagnostics and held-out scores. |
| [`src/rare_subgroups_contamination_report.py`](../../../src/rare_subgroups_contamination_report.py) | Planned coverage, identical-seed summaries, clean/noisy paired changes and all four predeclared screen outcomes. |
| [`scripts/run_rare_subgroups_contamination.py`](../../../scripts/run_rare_subgroups_contamination.py) | No-training preflight, explicit execution gate, committed-source/run identity, checksummed vector artifacts and resumable case reports. |
| [`tests/test_rare_subgroups_contamination.py`](../../../tests/test_rare_subgroups_contamination.py) | Artificial software cases, independently hand-derived arithmetic, isolation and failure/recovery checks. |

Original fitting X, true labels, true groups and row order are shared across each noise triplet; only observed majority labels change. Corruption uses stable pre-shuffle IDs and a separate keyed permutation. Scaling fits on original fitting X only. The training API takes fitting X/observed y, clean selection X/y, design and seed. It cannot accept hidden fitting truth, corruption flags, calibration/test data or tie uniforms. Selection locks before hidden fitting diagnostics or held-out scoring.

The observed original count difference determines each generation budget. AW uses the existing audited implementation; its retention control restores originals around exactly the same generated rows. The default AW fit is reused inside the nine-candidate grid, and the selected fit is reused without refitting. Unselected candidates never predict calibration/test data. If default AW is selected, the held-out predictions are also reused.

SMOTE instrumentation observes the interpolation arguments after the library draws them and delegates numerical generation to the library. It draws no additional random numbers. ROS uses the reference library's `sample_indices_`. Tests verify exact output equality, repeated calls, duplicate-feature behavior and interpolation reconstruction. These hooks target the pinned imbalanced-learn API; a dependency update requires revalidation. They do not change the historical sampler implementations.

After selection, each fitted model uses the protocol's independently keyed uniform tie values to select the 100th largest calibration key. This gives 100 false positives among 2,000 calibration negatives when keys are unique. Duplicate keys fail explicitly without redrawing or trying another seed. Actual test FPR is measured separately. Selection AP uses unmodified probabilities, not tie values.

## Run outputs and published copies

Original cloud output root: `results\rare-subgroups-contamination-v1`. Compact measured copies are under [contamination_v1](contamination_v1). Freeze the enabled configuration in a clean commit before any new execution. One process may write a directory at a time.

| Relative path | Contents |
| --- | --- |
| `manifest.json` | Complete configuration and semantic/file hashes, source hashes, clean Git commit, environment versions, jobs and run ID. |
| `job-<hash>.json` | Seven method statuses, metrics/errors, all nine candidate records, selection decisions, original-row retention, parent/cluster traces, scaler, partition/vector hashes and actual classifier-fit attempts. |
| `predictions\case-<hash>.npz` | Selection y and every successful candidate's probabilities; calibration y and successful methods' probabilities; test y/groups/probabilities; calibration/test tie uniforms; hidden fitting labels/groups/corruption/stable IDs for audit. Numeric arrays only; loading uses `allow_pickle=False`. |
| `reports\scores.csv` | Every planned method row, including failures, unsupported and missing rows with blank metrics; noise level, generation/final counts, group retention, denominators, thresholds and all seven metrics. |
| `reports\coverage.csv` | All completion statuses and shared-success seed counts for each of the 18 settings and seven methods. |
| `reports\summary.csv` | Per-setting means and sample SD for performance and original-group retention on identical successful seeds across all methods. `n_values` excludes undefined fractions; zero-noise retention has null mean/SD. One seed has null SD; no common seeds gives null means. |
| `reports\paired_changes.csv` | Every method's per-seed noisy-minus-clean metric differences, with explicit incomplete-pair status and null differences where needed. |
| `reports\screening.json` | Rare-protection and contamination-specific AP conditions for all 12 contaminated cells; paired AP interactions; control coverage at all noise levels; final descriptive status. |
| `reports\status.json` | Durable cases, success coverage, known completed-case fit attempts and screening status. |

With zero corrupted rows, noise-retention fraction is null. Interpolation-parent categories describe exposure to corrupted labels, not oracle synthetic classes. Removal changes final training counts relative to retention. The AP interaction compares contamination-induced changes against the matching zero-noise case; retaining rare observations alone is insufficient to pass the screen.

Sampler trace IDs index the shuffled original fitting arrays. Map them through saved `fitting_stable_ids` to obtain the pre-shuffle identities used to choose corrupted observations.

Saved selection/calibration vectors allowed the completed independent audit to recompute all 810 candidate AP values, 630 tie cutoffs/calibration counts and all held-out metrics, addressing the first pilot's audit limitation. The [audit helper](../../../scripts/audit_rare_subgroups_contamination.py) can repeat these checks from complete raw checkpoints without classifier refits; it optionally compares zero-noise probabilities with the first pilot. The [plot helper](../../../scripts/plot_rare_subgroups_contamination.py) needs only the published scores CSV and fits no models.

## Resuming and failures

Execution requires the metadata flag to be enabled for a separately authorized experiment stage, unchanged frozen design, and tracked source/configuration in a clean commit. The runtime configuration must match the file on disk. Any source, configuration, Git commit or environment change produces a different manifest and rejects an existing result directory. The first pilot's checkpoints are never accepted as this study's cases.

Before fitting a new case, the runner verifies every durable case record, NPZ file checksum, vector-column schema, individual array fingerprints and saved truth/partition alignment. Corruption fails loudly. A checkpoint is durable only after its vectors and full record are written. An interrupted unfinished case is rerun; intact complete cases are reused. A partial case with failed methods is still a completed checkpoint: it is reported and reused, not automatically retried or replaced.

Unsupported AW candidates remain unsupported without ROS/cluster fallback. If all selection candidates fail, tuned AW fails while fixed methods can still be reported. A held-out failure never triggers a different candidate selection. Missing/failed/unsupported metrics are not zeros. The descriptive screen reports `incomplete_coverage` before any favorable conclusion if planned method coverage is incomplete.

Exit codes: 0 for preflight or full method coverage; 1 for traversal with partial method coverage; 2 for configuration/identity/preflight errors; 130 for a controlled interruption. Unexpected programming errors propagate with a traceback. Known sampler, classifier and scoring errors have method/candidate statuses. The 1,260-fit maximum assumes no unfinished-case reruns; case counters do not include work lost before a durable checkpoint.

## Verification and next stage

The focused checks cover keyed label-only pairing and zero-noise stream equivalence; exact library-output/provenance equivalence; hand-derived tie calibration, F1/AP/recall and AP-interaction signs; four/five-seed and tolerance boundaries; distinct-base counting; same-control coverage across noise levels; hidden-truth/held-out isolation; default/nondefault fit reuse; unsupported/selection/scoring failures; full audit-vector retention; corruption detection; changed-identity rejection; and interruption/resume. A tiny real forest fixture exercises the complete computation without running a research cell.

Before research execution, cloud verification with Python 3.11.17 and the pinned environment found **51 new contamination tests passed; 127 combined contamination/pilot/AW tests passed; 875 tests passed in the full suite**, with no failures or skips reported. The real CLI preflight, frozen design fingerprint, document links, prior paper/result fingerprints and then-absent contamination research output directory were also checked. These historical counts establish software behavior, not research performance or efficacy.

The first experiment and its audit are complete. No new sampler, additional classifier, real-data study or novelty claim is included here. Further experiments require a separately specified stage.
