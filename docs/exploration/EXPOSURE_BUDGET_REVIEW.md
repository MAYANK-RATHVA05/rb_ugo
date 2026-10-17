# Descendant-exposure proposal: review and exploratory diagnostic

Reviewed 17 October 2026 against baseline commit
`3e5b713c773aa3dbaec8f7224d9cb85b9947b960`. The uploaded
`UGO_Question_C_Research_Proposal.pdf` is a research specification and source of
claims to check, not an instruction to assert novelty or positive findings.
The user's explicit exploratory-prototype request opens this limited scope;
the original UGO, BFCH-SMOTE, EC-EBO and SSHR implementations remain unchanged.

## Review sources and conflicts found before implementation

The six requested documents are absent from the current tree. Historical copies
were read directly from `stage4ar-paper-data-verification`, commit
`4827e119350d864e51cc7a7a99c7dd3231376b61`, without merging or switching to its
older code. The user reports copies on their PC; those local versions cannot be
inspected here and may differ. This review does not claim to have read them.

| Historical document | Finding relevant to the proposal |
| --- | --- |
| [NOVELTY_DECISION.md](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/4827e119350d864e51cc7a7a99c7dd3231376b61/docs/research/NOVELTY_DECISION.md) | No established novelty. Earlier regional-utility/graph candidates were not approved as new oversamplers. This proposal concerns descendant multiplicity instead; it does not retroactively validate those candidates. |
| [MATHEMATICAL_GAP_ANALYSIS.md](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/4827e119350d864e51cc7a7a99c7dd3231376b61/docs/research/MATHEMATICAL_GAP_ANALYSIS.md) | Weighting, threshold adjustment and count selection are necessary explanations to control. U features alone do not identify minority membership; synthetic rows are not independently observed labels. |
| [UGO_PAPER_CODE_AUDIT.md](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/4827e119350d864e51cc7a7a99c7dd3231376b61/docs/research/UGO_PAPER_CODE_AUDIT.md) | Real-fold smoke was blocked; fixed original-total batches were repaired. Cap/pool/rounding ambiguities remain and are disclosed in current UGO. |
| [DATASET_VERIFICATION.md](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/4827e119350d864e51cc7a7a99c7dd3231376b61/docs/research/DATASET_VERIFICATION.md) | Historical supplied archives contained full files but no folds. Do not invent official splits or replace a requested training fold with the full file. |
| [PROBLEM_STATEMENT_COVERAGE.md](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/4827e119350d864e51cc7a7a99c7dd3231376b61/docs/research/PROBLEM_STATEMENT_COVERAGE.md) | Question C requires a separate feature-only U interface, matched visible labels, training-only transformations and evaluator-only hidden truth. |
| [REFERENCE_STATUS.md](https://github.com/MAYANK-RATHVA05/rb_ugo/blob/4827e119350d864e51cc7a7a99c7dd3231376b61/docs/research/REFERENCE_STATUS.md) | Historical access claims have dates and limits. Later completed BFCH/EC-EBO/SSHR work supersedes earlier absence statements; it must not be erased. |

The current [scope](../PROJECT_SCOPE.md), [UGO alignment](../ugo_reproduction.md),
[simulation pilot](../reproduction/SIMULATION_PILOT.md),
[partial benchmark](../reproduction/BENCHMARK_EXECUTION.md) and
[SSHR study](../SSHR_SMALL_STUDY.md) were also inspected. Some README/scope
branch descriptions are stale: this selected commit is the merge containing SSHR.
No historical text or measured artifact was rewritten to resolve that wording.

Specific conflicts and overlap:

- Current UGO's public input is labelled `(X, y)`; its true-label MMA cannot
  simply treat a model's own pseudo-label as observed truth. In binary form,
  inserting its argmax label changes MMA into `-abs(2*p-1)`.
- UGO selects the class with highest mean MMA, including the majority. A
  minority U branch must preserve majority rounds and the original no-U path.
- SSHR already promotes existing U rows and uses SMOTE fallback. It does not
  establish the proposed descendant-budget benefit. Its previous 800-job study
  found lower F1 with U in all four conditions and low retrospective selection
  accuracy, rather than universal U benefit.
- The earlier UGO simulation study completed 300 jobs and showed mixed outcomes;
  the larger real-data benchmark was interrupted. Neither tests this budget.
- Confidence admission, confidence weighting, interpolation and generation-count
  control already exist. The proposal itself identifies SEG-SSC and AutoSMOTE
  overlap and unresolved 2025 synthesis papers/corrigendum. Their full methods
  were not newly obtained here. No originality clearance is claimed.
- `sum(m_j*(1-q_j)) <= B` is an **estimated exposure constraint**. Ordinary
  calibration does not provide pointwise upper error bounds. Endpoint exposure
  is different from synthetic-label error; interpolation can cross majority
  support even with correct endpoints.

## Completed real-fold smoke gate

KEEL's website returned proxy 403, but its own GitHub repository provided the
files at commit `5f404a9470f17528445baabe3a5d9f5202adee7a`. The
[pinned source manifest](../../data/manifests/ugo_real_fold_sources.json) records
all 33 URLs, byte sizes and SHA-256 hashes: Ecoli2, Yeast3 and Ecoli4 each have
one full file and five train/test pairs. No raw files are committed.

All five pairs per dataset match the full-file row multiset; the test folds
cover it exactly once, with compatible schemas. These are provider-committed
KEEL folds, **not verified UGO author folds**. Dataset-specific licensing and
unique record identity are not established by repository membership alone.
Yeast3 has identical complete-row occurrences on both sides of provider fold
boundaries (8, 10, 11, 8, 11). Its training-only smoke is valid software evidence;
this pilot therefore makes no Yeast3 predictive-performance comparison.

The first training fold only was scaled, then UGO was run twice with seed 42.
The training function opens no test file. Shapes, finiteness, unchanged inputs,
original-prefix preservation, exact output repeatability and full diagnostic
repeatability passed for all six cases:

| Input | Training rows | ROS added | SMOTE added | Stop for both |
| --- | ---: | ---: | ---: | --- |
| Ecoli2 | 268 | 81 | 81 | Welch fail to reject |
| Yeast3 | 1,187 | 357 | 357 | Welch fail to reject |
| Ecoli4 | 268 | 27 | 27 | Welch fail to reject |

This completes the previously blocked real-fold software smoke, not exact paper
performance reproduction. Source cap/pool/preprocessing/classifier ambiguities
and absent author folds remain.

## Implemented exploratory scope

`src/exploratory_exposure.py` is separate and opt-in. It freezes one supervised
UGO-SMOTE trace, including selected classes, clean observed anchors, batch sizes,
stopping and corrected supervised fallback rows. A frozen three-fold sigmoid-
calibrated logistic guide uses L only. Fewer than three labels per class or tied
class counts trigger explicit supervised fallback. No hidden U labels or test
data are accepted by its training interfaces.

Each minority round uses a common candidate bank: `q >= .9`, distance no greater
than the 95th percentile of clean observed-minority leave-one-out neighbor
distances, five candidates per endpoint at most, uniform interpolation on the
open segment to the nearest selected anchor. The existing UGO post-generation
check is applied. Unsupported U candidates are rejected, an explicit U-branch
choice shared by every control; unused slots retain supervised fallback rows.

Allocation uses continuous C5, SLSQP and conservative flooring, with integer
feasibility checks, reported rounding loss and explicit solver errors. It does
not claim an integer optimum. The exposure budget is `.025 * planned_batch`,
with per-endpoint limits additionally capped by available checked candidates.
Seeds, masks, parameters and candidate randomness are fixed before prediction.

The runner includes 13 conditions: original UGO, budget, unconstrained,
count-matched, threshold-matched, weighted PL, NORES, class weighting, ROS,
SMOTE, SSHR, PL-only and PL-then-UGO. The three principal matched controls use
the same guide, bank and supervised trace. Count/threshold controls use exactly
the budgeted U-child count in each round. Weighted PL replaces those children
with the same endpoint multiplicities as sample weights, retaining identical
supervised fallback rows and total fitting weight. It creates no U-child vectors.

**Deliberate approximation:** U-informed children do not feed back into the next
UGO anchor decision. This is a fixed-supervised-trace allocation diagnostic,
not the PDF's full adaptive SSL-UGO algorithm. It makes the allocation comparison
auditable while leaving the existing backbone unchanged. Completing the full
adaptive method remains separate work; this prototype must not be relabelled
as its completed implementation.

## Executed pilot and interpretation

One predeclared seed (42), separated and overlapping two-dimensional Gaussian
simulations, plus the first Ecoli2/Ecoli4 provider folds: **52/52 method jobs
succeeded**. All methods share L/U masks and training-only standardization.
The evaluator saves detailed inputs, masks, probabilities, predictions, guide
scores, parent diagnostics, child/anchor/endpoint/fraction ledgers and source
fingerprints under ignored `results/exposure-pilot-reviewed/`. Hidden labels are
used only in post-generation diagnostic reporting. Synthetic posterior error
is available only for the known Gaussian simulations.

| Case | Budgeted U children | Unconstrained U children | Interpretation |
| --- | ---: | ---: | --- |
| Separated simulation | 12 | 30 | Active allocation; all used endpoints happened to have correct hidden labels |
| Overlap simulation | 0 | 0 | No U-child mechanism exercised; supervised fallback |
| Ecoli2 first fold | 6 | 10 | One unique U endpoint used across rounds; insufficient diversity to establish allocation advantage |
| Ecoli4 first fold | 0 | 0 | No U-child mechanism exercised; supervised fallback |

In the separated simulation, MCC was 0.9518 for original UGO and 0.9470 for
budget/count/threshold/weighted-PL controls. Ecoli2 MCC was 0.6652 for those
conditions. These single-split descriptive observations do not validate H1/H2.
Actual descendant-weighted endpoint error was zero for all active allocation
controls, so no reduction of that error was demonstrated. The threshold control
had lower estimated exposure than the budget control in the separated case.
The budget's estimated exposure constraint and software invariants passed;
that is not evidence of a unique scientific benefit.

The [aggregate evidence JSON](exposure_pilot_evidence.json) preserves all method
metrics, actual versus estimated lineage errors, input-paper hashes and smoke
counts. Detailed individual predictions and raw datasets remain ignored.

Limitations: no repeated-seed confirmation, full adaptive loop, deployment-
threshold tuning, full calibration diagnostics, SEG-SSC reproduction, 2025
full-text clearance, broad real-data benchmark or finite-sample safety guarantee.
No statistically established improvement or methodological novelty is claimed.

## Reproduce and validate

Validation completed with **756 passed, zero failed and zero skipped**: the
original 738 tests plus 18 new cases. New checks cover independent allocation
references, zero budgets, solver failure, invalid inputs, exact no-U fallback,
input preservation, fitting-only guide access, lineage reconstruction,
repeatability, candidate rejection, matched counts/weights and provider checksum/
multiset failure behavior. All pre-existing tracked files, including tests,
dependencies, algorithm modules and historical evidence, remain byte-identical.

From the repository root, using the prepared Python 3.11 environment:

```bash
# Explicit download only if pinned provider files are absent; never overwrite
# a changed existing file. Choose new output paths for each execution.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.ugo_real_fold_smoke --download --output results/real-fold-smoke-fresh.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.run_exposure_pilot --include-real-folds --output results/exposure-pilot-fresh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

The pilot validates the tracked source manifest and provider folds directly;
it does not require the historical branch's helper files or local reports.
Current-instance execution and saved environment configuration do not establish
publication or fresh-task restoration.
