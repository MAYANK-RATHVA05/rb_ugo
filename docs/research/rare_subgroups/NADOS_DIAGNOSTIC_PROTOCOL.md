# NADOS diagnostic on the frozen rare-subgroup contamination cases

Status: **Part 2 software complete; execution enabled, first run pending**. This protocol adds a
fixed NADOS baseline to the exact 90 deterministic cases from the completed
[contamination study](CONTAMINATION_PROTOCOL.md). It does not modify that
study, propose a new sampler, or report NADOS research results.

## Purpose

The completed study showed that the predeclared AW removal/retention tradeoff
screen did not trigger. NADOS offers a different existing mechanism: it first
decides whether an observed minority row is reliable enough to seed generation,
then weights eligible seeds by difficulty. This follow-up asks a narrower
diagnostic question:

> Does NADOS reduce synthetic exposure to deliberately mislabeled majority
> rows while also rejecting genuine rare-minority rows?

The implementation is the independent paper-aligned baseline documented in
[NADOS_BASELINE.md](NADOS_BASELINE.md). The paper provides no executable author
code, so results from this repository must be described as results from our
implementation, not an exact reproduction.

## Exact case reuse

The new configuration references the frozen parent scientific design by its
SHA-256 `adfdb2961963ff83b9552e84b06963689f938d853859d9bafc224106a08849db`
and completed run commit `975020ab3cc3ba5e15cfd29499c7ee5b776a0116`.
The loader validates both identities before exposing the parent generator.
Consequently the new runner uses the same:

- rare counts 4, 8 and 16;
- rare spreads 0.35 and 0.90;
- flipped-majority counts 0, 5 and 10;
- seeds 42 through 46;
- fitting, calibration and test populations and random streams;
- fitting-only `StandardScaler`, 100-tree random forest, and negative-only
  randomized-tie threshold calibration.

These factors give 3 × 2 × 3 × 5 = **90 paired cases**. Only NADOS is run, so
the maximum new budget is 90 classifier fits and 90 score rows. Existing
methods are not recomputed; a later audit may join NADOS rows to the already
published parent results by the exact case key.

## Fixed NADOS treatment

Each case uses a fresh sampler with `k=5`, reliability weight 0.5, difficulty
weight 0.5, reliability threshold 0.35, target ratio 1.0, minority label 1,
and the replicate seed. All original rows must remain in their original order.
The generated count must equal the observed class gap. Unsupported geometry is
recorded explicitly; there is no fallback sampler and no parameter selection.

The scaler is fitted on original fitting features only. NADOS and the random
forest receive only scaled fitting X and observed y. True fitting labels,
latent common/rare groups, corruption flags and stable IDs cannot enter the
training function. Calibration and test features are transformed only after
training returns.

## Saved diagnostics

For independent audit, each complete case checkpoint stores checksummed arrays
for fitting observed/true labels, group and corruption metadata, stable IDs,
calibration/test labels, test groups, tie uniforms, successful probability
vectors, and the numeric NADOS trace. The trace includes neighborhoods,
distances, support, reliability, difficulty, eligibility, seed probabilities,
and generated seed/partner/fraction identities.

After training, the evaluator reports eligibility separately for genuine
common, genuine rare, and corrupted observed-minority rows. It also counts
generated rows with clean/clean, clean/corrupt and corrupt/corrupt parents.
Those categories measure exposure to corrupted observed labels; they do not
identify the true class of a synthetic point.

## Frozen descriptive screen

The JSON freezes a conservative screen before any NADOS experiment. Further
method design is considered only if at least two distinct base settings show,
in at least four of five seeds, rare eligibility no greater than 0.50 and at
least 0.20 below common eligibility; rare recall is at least 0.05 below the
best existing control at comparable mean paired test FPR (difference at most
0.01); and corrupted-parent exposure is no more than half the AW retention
reference. These are descriptive engineering cutoffs, not significance tests
or novelty evidence. Failure of the screen means this route stops.

The runner deliberately does not calculate that cross-study conclusion. It
writes raw NADOS scores and eligibility diagnostics; a later audit must verify
case identity and combine them with the already published parent rows.

## Execution gate and commands

The configuration now has `execution_enabled: true` following the separate
authorization commit. Both the default command and `--check-only` validate
metadata and print the 90-case budget without generating
data, fitting models, or creating an output directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_rare_subgroups_nados --check-only
```

The authorization commit changed only the execution flag after all focused and
repository tests passed. The experiment has not yet run. Use the resumable
command below from a clean tracked checkout:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_rare_subgroups_nados --run --output results\rare-subgroups-nados-v1
```

Per-case JSON and NPZ files are atomic. On resume, every saved record, artifact,
array fingerprint, run manifest and source/config identity is verified before a
new model can be fitted. Corrupt or foreign checkpoints fail loudly.
