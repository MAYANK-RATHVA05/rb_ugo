# Dataset provenance and local preparation

This repository shares dataset manifests, source hashes and measured aggregate
reports. **Raw datasets, processed data and detailed experiment job records are
not uploaded to GitHub.** Place or copy the required files into each machine's
local checkout; Antigravity does not receive the cloud's raw files through Git.

The current [project scope](docs/PROJECT_SCOPE.md) covers published UGO,
BFCH-SMOTE, EC-EBO and SSHR reproductions. Source identity, paper metadata
compatibility and exact author-data equivalence are different claims. A matching
name, row count or imbalance ratio does not prove matching feature values,
preprocessing, class grouping or folds.

## UGO's historical 74-entry collection

The publisher supplement was inspected during the reproduction work. Its final
74-entry dataset metadata and the independently prepared input choices are
recorded in [ugo_74_approximate.json](data/manifests/ugo_74_approximate.json) and
the [collection report](docs/reproduction/DATASET_COLLECTION.md):

- 56 source names/aliases and class counts matched.
- 18 variants were explicitly accepted approximations or substitutions.
- 16 inputs had raw feature-count differences from the supplement.

The manifest preserves actual dataset IDs, relative paths, hashes, targets,
positive labels, feature metadata and preparation choices. Keep substitutions
visible in results; for example, a white-wine substitute must not be reported as
the original red-wine source. This collection does not establish exact author
rows, preprocessing or evaluation folds.

The historically prepared directory was `data/raw/ugo74_approx/`. It is ignored
by Git and absent from the current cloud checkout. A new clone contains the
manifest, not those 74 files. The [benchmark execution record](docs/reproduction/BENCHMARK_EXECUTION.md)
documents the interrupted run: 28,919 saved jobs, with 18 completed datasets
and 27,610 successes/110 failures in those completed datasets. The full benchmark
was not finished, and its historical `results/stage4r7-full/` cache is also absent
from this checkout. Restore the raw files and matching cache from a separate
local copy before attempting that exact run's resume.

The older [benchmark_candidates.json](data/manifests/benchmark_candidates.json)
is an eight-file historical audit, not a complete or authoritative replacement
for the 74-entry manifest.

## BFCH-SMOTE and EC-EBO pilot inputs

The [BFCH study](docs/BFCH_SMALL_STUDY.md) and
[EC-EBO study](docs/EC_EBO_SMALL_STUDY.md) used these local KEEL variants:

| Relative source path | Rows | Majority / minority | Study feature policy |
| --- | ---: | ---: | --- |
| `data/raw/ugo74_approx/winequality_red_8_vs_6.dat` | 656 | 638 / 18 | All 11 numeric features |
| `data/raw/ugo74_approx/vowel0.dat` | 988 | 898 / 90 | Ten acoustic features; speaker groups stay together across train/test |

SHA-256 values:

- Winequality: `6b556e909321b1ce3476506e188aabeaa3dea4a36cfb800d66b6f0a7375c4356`.
- Vowel0: `478dab0fcc87f8932e9481af3dd66dfa99601ba8b72c49d2525cba04b7f6cf76`.

Vowel's source has 13 numeric-stored columns. Its `TT`, `SpeakerNumber` and `Sex`
fields are excluded from interpolation; `SpeakerNumber` is used for fold grouping.
The ten-feature study variant differs from the source table. Numerical storage
does not by itself establish continuous feature semantics.

## SSHR pilot inputs

The complete SSHR implementation is on `sshr-implementation`; `main` does not
yet contain its runner. The
[SSHR source register](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/data/manifests/sshr_pilot_sources.json)
and [measured report](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/SSHR_SMALL_STUDY.md)
preserve the supplied files, archive hashes, feature columns and fixed protocol.
Place the extracted original files at these repository-relative paths:

| Relative source path | Rows | Numeric features | Majority / minority | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `data/raw/ecoli4.dat` | 336 | 7 | 316 / 20 | `1890c32ba2550cfe106d0524ebf6a42c2b50730f6521bd4f22eb58e7bda86bcb` |
| `data/raw/yeast4.dat` | 1,484 | 8 | 1,433 / 51 | `2e50d1b43c97875d2071859428d718487086e695a1a6ec559336ccadb40c1ff0` |

For the existing Windows clone, the full paths are
`C:\ugo\ugo_base\data\raw\ecoli4.dat` and
`C:\ugo\ugo_base\data\raw\yeast4.dat`. Keep the files unchanged after extraction.
An optional ZIP copy under `data/incoming/` is not a substitute for the extracted
`.dat` path. If an archive is present, the runner also verifies its registered
hash and member bytes. Do not commit the archives or papers.

To inspect the file hashes in PowerShell:

```powershell
Get-FileHash data\raw\ecoli4.dat -Algorithm SHA256
Get-FileHash data\raw\yeast4.dat -Algorithm SHA256
```

The pilot preserves all raw rows. Yeast4 has 31 same-label duplicate coordinate
pairs; groups remain intact across outer train/test and observed/unlabelled
training pools. Ecoli4 has no exact coordinate duplicates. Conflicting-label
coordinate groups are rejected rather than assigned an invented consensus label.
Both sources have no missing/nonfinite values, but their numeric declarations
do not prove physical plausibility of interpolated biological measurements.

The controlled study hides a documented subset of **outer-training labels only**.
SSHR receives the resulting unlabelled feature pool without its hidden labels;
all baselines receive the same reduced labelled set. Hidden labels are retained
only by the evaluator for retrospective pseudo-label auditing. Test features and
labels never enter that pool. No separate naturally unlabelled data source was
used in this pilot, and the paper's original unlabelled-data construction remains
unverified. Existing selected observations are pseudo-labelled rows; only the
SMOTE fallback generates new feature vectors.

Run manually from the full branch after verifying both files:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_sshr_small_study --output results\sshr-small-study --report-dir docs\sshr_pilot
```

Linux uses `.venv/bin/python`. A new machine needs its own dependencies, raw
inputs and, for cache resume, the matching ignored job records. The
[shared artifact guide](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/sshr-implementation/docs/sshr_pilot/README.md)
explains the CSV/JSON summaries and their limitations.

## Adding other external data

1. Obtain the exact variant from a verified source and check its licence and
   access conditions. Record the provider, source location, release/version,
   access date and file hash; do not guess missing provenance.
2. Preserve the original under `data/raw/` and record any required compatibility
   copy or transformation separately. Do not edit a file merely to make a parser
   accept it. Preserve official folds, source partitions, groups and timestamps.
3. Declare the target, original label meanings, binary class grouping and feature
   semantics. Numeric-looking category codes are not continuous measurements.
   Define the positive outcome from the research problem rather than using
   outer-test class counts to guide training.
4. Decide fitting, selection and outer-test boundaries before learned operations.
   Fit preprocessing on training partitions only. Resample training data only;
   never put held-out features into an unlabelled pool.
5. Commit non-sensitive provenance and permitted aggregate documentation only.
   Raw/processed files, credentials, virtual environments, caches and detailed
   generated outputs stay local. Do not force-add ignored datasets.

KEEL `.dat` files can declare attribute types, inputs and outputs; CSV requires
an explicit target and does not provide a semantic feature schema. The loader
supports UTF-8 binary CSV and a documented KEEL subset, preserves input row order
and rejects unsupported targets or malformed records. `main_stage2.py` reports
descriptive counts without fitting preprocessing or models. For example:

```bash
.venv/bin/python main_stage2.py --data data/raw/ecoli4.dat
.venv/bin/python main_stage2.py --data tests/fixtures/toy.csv --target class
```

`tests/fixtures/toy.csv` and `toy.dat` are tiny artificial software fixtures,
not research benchmarks or performance evidence. The optional Stage 2 JSON
reports contain source paths, labels and counts; review them before sharing.
They are written under ignored `results/stage2/` and never overwrite an existing
report.
