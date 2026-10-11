# Actual resource and KEEL verification: Stage 4A-R

Verified 2026-10-11 from supplied files, not dataset-name assumptions. All eight
ZIPs pass CRC checks and contain exactly one full `.dat` each, with **no predefined
training/test folds**. The full files load in the existing Stage 2 parser; they
are not training partitions for the requested official-fold UGO smoke checks.

## Accessible inventory and missing resources

[resource_inventory.csv](../../data/manifests/resource_inventory.csv) lists each
of the 14 accessible uploads (six PDFs, eight ZIPs): original filename, actual
accessible cloud path, portable imported location, bytes, SHA-256, parse status
and intended use. The fifteenth offered resource, the reported 13-paper ZIP,
is inaccessible because the attachment transfer limit is 32 MiB. Its checksum,
size and members remain unknown. See [REFERENCE_STATUS.md](REFERENCE_STATUS.md).

User ZIPs were copied unchanged to ignored `data/incoming/`, extracted without
renaming to ignored `data/raw/keel/<dataset>/`, and checksum-verified. UGO is
in ignored `papers/base/`; the other five supplied PDFs in `papers/additional/`.
No real data/PDF is Git tracked. Scripts, manifests and documents are tracked.
Actual cloud attachment paths are inventory evidence, not universal local paths.

## Verified full-file statistics

| Relation / filename | Rows | Features (numeric/categorical) | Majority / minority | IR | Missing | Feature / complete duplicates | Fold pairs |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: |
| ecoli2 / ecoli2.dat | 336 | 7 (7/0) | 284 / 52 | 5.461538 | 0 | 0 / 0 | 0 |
| glass6 / glass6.dat | 214 | 9 (9/0) | 185 / 29 | 6.379310 | 0 | 1 / 1 | 0 |
| yeast3 / yeast3.dat | 1484 | 8 (8/0) | 1321 / 163 | 8.104294 | 0 | 31 / 31 | 0 |
| page-blocks0 / page-blocks0.dat | 5472 | 10 (10/0) | 4913 / 559 | 8.788909 | 0 | 79 / 69 | 0 |
| ecoli4 / ecoli4.dat | 336 | 7 (7/0) | 316 / 20 | 15.800000 | 0 | 0 / 0 | 0 |
| yeast4 / yeast4.dat | 1484 | 8 (8/0) | 1433 / 51 | 28.098039 | 0 | 31 / 31 | 0 |
| segment0 / segment0.dat | 2308 | 19 (19/0) | 1979 / 329 | 6.015198 | 0 | 224 / 224 | 0 |
| abalone19 / abalone19.dat | 4174 | 8 (7/1) | 4142 / 32 | 129.437500 | 0 | 0 / 0 | 0 |

Every file declares target `Class` with original labels `negative` and `positive`.
Here `negative` is majority=0 and `positive` minority=1. All have zero missing
feature values and zero rows containing missing features. Feature names and types,
exact relation, target vocabulary, file/ZIP hashes and exact unrounded IR are in
[keel_dataset_manifest.csv](../../data/manifests/keel_dataset_manifest.csv).
Abalone19 has nominal `Sex`, not an arbitrary integer distance; the other seven
are entirely numeric. Header spellings such as `Lenght`, `Vegde-sd` and
`Saturatoin-mean` are retained exactly. Duplicate counts are additional
occurrences beyond the first, not deleted records. Page-blocks0's 79 feature /
69 full-row duplicates include coincident features with differing labels.

**Provenance limit:** user-supplied KEEL-format binary archives are verified as
files. No authenticated provider URL, release, dataset license, original UCI
class grouping/domain meaning or official split provenance is established.
Ecoli2/Ecoli4 and Yeast3/Yeast4 are related variants, not independent data sources.
Count agreement with a published table is useful corroboration, not provider
authentication. Do not guess missing metadata or equate KEEL software licensing
with each dataset's license.

## Fold validator and leakage checks

`verify_keel.py` identifies actual `*-k-itra.dat` / `*-k-itst.dat` pairs while
retaining identifiers and nested paths. It loads train/test separately, establishes
roles from training labels and reapplies those roles to test even if frequencies
reverse. It reports schema/vocabulary mismatches, incomplete/duplicated fold
parts, nonconsecutive IDs, multiple repeated-CV families, row-multiset mismatches,
feature/complete-row overlap and union-of-tests coverage against the full file.
It rejects unsupported one-class partitions rather than inventing a role.
It fits no preprocessing and concatenates no training/evaluation arrays.

The current report marks **all eight MISSING_FOLDS**, with coverage UNVERIFIED.
`VERIFIED_MULTISET` on a future archive would mean multiset/structural checks,
not authenticated official provenance. Identical-valued occurrences cannot be
assigned record identity without provider IDs: counts can reveal erroneous reuse
when multiplicities exceed full data but cannot prove distinct identical records
were not reused. Report ambiguity instead of deleting legitimate duplicates.
The complete file and fold copies count as one dataset, not extra independent runs.

## Stage 2 reaudit and test evidence

The existing `src/data_loader.py`, `src/imbalance_stats.py` and `main_stage2.py`
were inspected and exercised without modification. No supplied file failed.
All eight real-file integration checks passed locally, including nominal Sex,
feature order/types, original labels, encoding and independent expected row/count
checks. Existing tests cover KEEL directives, quoted/unquoted entries, numeric /
integer/nominal types, missing feature tokens, unusual labels and invalid classes.
No correctness bug was demonstrated, so no Stage 2 repair or API change was made.

Additional tests verify safe import failures/idempotency, artificial fold coverage
and mapping reversal, duplicate identity limits, missing-fold blocking and a
training-only smoke path. Artificial folds/estimators are **software fixtures**,
not regenerated official data or real benchmark evidence. On a Git-only clone,
eight real-data integrations skip clearly until the matching local assets are
imported; this cloud run executed all eight, with no skipped tests.

## Real-data UGO smoke status

`results/verification/ugo_smoke_summary.csv` (ignored) contains six rows:
Ecoli2, Yeast3, Ecoli4 × ROS/SMOTE, each
`BLOCKED_MISSING_VERIFIED_FIVE_FOLD`. Command exit code is 2. No real UGO fit,
scaling, synthetic count, noise/history, determinism or termination result is
reported. Full data and regenerated splits were **not** substituted.

After source-provided five-fold archives and provenance are supplied, rerun
verification then the smoke command. The runner checks the training checksum,
loads only the first verified five-fold training partition, fits scaling there,
checks original/label/shape/finite preservation and repeats seed 42. It never
opens the test file during generation. Future success would still be software
smoke evidence, not performance or uniquely faithful UGO reproduction.

## Offline import in Windows Antigravity

Keep your downloaded originals outside the clone. In PowerShell, after selecting
this repository folder and its Python 3.11 interpreter:

```powershell
$archiveDir = "C:\Users\YOUR_NAME\Downloads\rb_ugo_resources"
.\.venv\Scripts\python.exe scripts/import_resources.py `
  --dataset-zip "ecoli2=$archiveDir\ecoli2.zip" `
  --dataset-zip "glass6=$archiveDir\glass6.zip" `
  --dataset-zip "yeast3=$archiveDir\yeast3.zip" `
  --dataset-zip "page-blocks0=$archiveDir\page-blocks0.zip" `
  --dataset-zip "ecoli4=$archiveDir\ecoli4.zip" `
  --dataset-zip "yeast4=$archiveDir\yeast4.zip" `
  --dataset-zip "segment0=$archiveDir\segment0.zip" `
  --dataset-zip "abalone19=$archiveDir\abalone19.zip"
.\.venv\Scripts\python.exe scripts/import_resources.py `
  --paper-file "base=$archiveDir\1-s2.0-S095070512600612X-main.pdf"
.\.venv\Scripts\python.exe scripts/verify_keel.py
.\.venv\Scripts\python.exe scripts/ugo_smoke.py
.\.venv\Scripts\python.exe -m pytest -q
```

Adjust the directory and filenames to actual local resources. Other paper files
use `--paper-file "additional=PATH"`; smaller related-paper archives use
`--paper-zip PATH`. Linux uses `.venv/bin/python` with the same CLI arguments.
No internet or credentials are required by these scripts. Imports with equal
hashes are idempotent; conflicting existing contents are refused. Source ZIPs
stay unchanged; traversal, absolute/Windows escape paths, symlinks, size-limit
violations and file/case collisions are rejected. Verification regenerates its
own derived manifest/report; smoke regenerates its own summary.

Use `--root PATH` on import; verification exposes `--data-root`, `--archive-dir`,
`--manifest`, `--fold-report`; smoke exposes `--data-root`, `--fold-report`,
`--datasets`, `--output`. Each supports `--help`. Defaults resolve from the
script location, not `/workspace` or the terminal's current directory.
The current smoke exit 2 is the expected missing-fold blocker; do not bypass it.
No Windows runtime test was performed in this Linux cloud.
